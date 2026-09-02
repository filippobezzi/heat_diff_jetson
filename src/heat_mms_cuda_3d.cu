#include "heat_3d.h"
#include <cuda_runtime.h>

#define MMS_L_DEF 1.0f
#define MMS_ALPHA_DEF 0.143f
#define RADIUS 1

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void init_field_cuda_kernel(const int nx, const int ny, const int nz,
                                       const float dx, const float dy, const float dz,
                                       const float L, const float alpha_val,
                                       float *__restrict__ alpha,
                                       float *__restrict__ u) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        float x = (i + 1.0f) * dx;
        float y = (j + 1.0f) * dy;
        float z = (k + 1.0f) * dz;
        float pi_l = (float)M_PI / L;
        size_t idx = IDX3(i, j, k, nx, ny);
        alpha[idx] = alpha_val;
        u[idx] = sinf(pi_l * x) * sinf(pi_l * y) * sinf(pi_l * z);
    }
}

// Optimal 2.5D Shared Memory + Register Sliding Window MMS Stencil Kernel (Dynamic Shared Memory with on-the-fly harmonic mean)
__global__ void solve_stencil_cuda_optimized_kernel(const int nx, const int ny, const int nz,
                                                    const float inv_dx2, const float inv_dy2, const float inv_dz2,
                                                    const float *__restrict__ alpha,
                                                    const float *__restrict__ u,
                                                    float *__restrict__ u_next) {
    extern __shared__ float s_mem[];

    int tile_x = blockDim.x;
    int tile_y = blockDim.y;
    int pitch = tile_x + 2 * RADIUS;
    int slice = pitch * (tile_y + 2 * RADIUS);
    // allocate one contiguous array for u and alpha in shared memory
    // [ (0, ...,  slice - 1), (slice, ..., 2 * slice - 1) ]
    float *s_u     = s_mem;
    float *s_alpha = &s_mem[slice];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = blockIdx.x * tile_x + tx;
    int j = blockIdx.y * tile_y + ty;

    int sx = tx + RADIUS;
    int sy = ty + RADIUS;

    // Registers for sliding window along Z
    float u_bot = 0.0f;
    float u_curr = 0.0f;
    float u_top = (i < nx && j < ny && nz > 0) ? u[IDX3(i, j, 0, nx, ny)] : 0.0f;

    float a_bot = 0.0f;
    float a_curr = 0.0f;
    float a_top = (i < nx && j < ny && nz > 0) ? alpha[IDX3(i, j, 0, nx, ny)] : 0.0f;

    for (int k = 0; k < nz; k++) {
        // Slide registers along Z
        u_bot = u_curr;
        u_curr = u_top;
        u_top = (k + 1 < nz && i < nx && j < ny) ? u[IDX3(i, j, k + 1, nx, ny)] : 0.0f;

        a_bot = a_curr;
        a_curr = a_top;
        a_top = (k + 1 < nz && i < nx && j < ny) ? alpha[IDX3(i, j, k + 1, nx, ny)] : a_curr;

        // Load 2D XY slice into shared memory
        if (i < nx && j < ny) {
            s_u[sy * pitch + sx]     = u_curr;
            s_alpha[sy * pitch + sx] = a_curr;
        } else {
            s_u[sy * pitch + sx]     = 0.0f;
            s_alpha[sy * pitch + sx] = a_curr;
        }

        // Halos with Dirichlet u=0 boundary condition
        if (tx == 0) {
            s_u[sy * pitch + 0]     = (i > 0 && j < ny) ? u[IDX3(i - 1, j, k, nx, ny)] : 0.0f;
            s_alpha[sy * pitch + 0] = (i > 0 && j < ny) ? alpha[IDX3(i - 1, j, k, nx, ny)] : a_curr;
        }
        if (tx == tile_x - 1) {
            s_u[sy * pitch + (tile_x + 1)]     = (i + 1 < nx && j < ny) ? u[IDX3(i + 1, j, k, nx, ny)] : 0.0f;
            s_alpha[sy * pitch + (tile_x + 1)] = (i + 1 < nx && j < ny) ? alpha[IDX3(i + 1, j, k, nx, ny)] : a_curr;
        }
        if (ty == 0) {
            s_u[0 * pitch + sx]     = (j > 0 && i < nx) ? u[IDX3(i, j - 1, k, nx, ny)] : 0.0f;
            s_alpha[0 * pitch + sx] = (j > 0 && i < nx) ? alpha[IDX3(i, j - 1, k, nx, ny)] : a_curr;
        }
        if (ty == tile_y - 1) {
            s_u[(tile_y + 1) * pitch + sx]     = (j + 1 < ny && i < nx) ? u[IDX3(i, j + 1, k, nx, ny)] : 0.0f;
            s_alpha[(tile_y + 1) * pitch + sx] = (j + 1 < ny && i < nx) ? alpha[IDX3(i, j + 1, k, nx, ny)] : a_curr;
        }

        __syncthreads();

        // Compute 3D 7-point stencil with on-the-fly harmonic means
        if (i < nx && j < ny) {
            size_t idx = IDX3(i, j, k, nx, ny);

            float a_left_val  = s_alpha[sy * pitch + (sx - 1)];
            float a_right_val = s_alpha[sy * pitch + (sx + 1)];
            float a_front_val = s_alpha[(sy - 1) * pitch + sx];
            float a_back_val  = s_alpha[(sy + 1) * pitch + sx];
            float a_bot_val   = (k > 0) ? a_bot : a_curr;
            float a_top_val   = (k < nz - 1) ? a_top : a_curr;

            float ax_left  = (i > 0)      ? harmonic_mean(a_curr, a_left_val)  : a_curr;
            float ax_right = (i < nx - 1) ? harmonic_mean(a_curr, a_right_val) : a_curr;

            float ay_front = (j > 0)      ? harmonic_mean(a_curr, a_front_val) : a_curr;
            float ay_back  = (j < ny - 1) ? harmonic_mean(a_curr, a_back_val)  : a_curr;

            float az_bot   = (k > 0)      ? harmonic_mean(a_curr, a_bot_val)   : a_curr;
            float az_top   = (k < nz - 1) ? harmonic_mean(a_curr, a_top_val)   : a_curr;

            float u_xm = (i > 0)      ? s_u[sy * pitch + (sx - 1)] : 0.0f;
            float u_xp = (i < nx - 1) ? s_u[sy * pitch + (sx + 1)] : 0.0f;
            float u_ym = (j > 0)      ? s_u[(sy - 1) * pitch + sx] : 0.0f;
            float u_yp = (j < ny - 1) ? s_u[(sy + 1) * pitch + sx] : 0.0f;
            float u_zm = (k > 0)      ? u_bot                      : 0.0f;
            float u_zp = (k < nz - 1) ? u_top                      : 0.0f;

            float f_x_left  = ax_left  * (u_curr - u_xm) * inv_dx2;
            float f_x_right = ax_right * (u_xp - u_curr) * inv_dx2;

            float f_y_front = ay_front * (u_curr - u_ym) * inv_dy2;
            float f_y_back  = ay_back  * (u_yp - u_curr) * inv_dy2;

            float f_z_bot   = az_bot   * (u_curr - u_zm) * inv_dz2;
            float f_z_top   = az_top   * (u_zp - u_curr) * inv_dz2;

            u_next[idx] = u_curr + (f_x_right - f_x_left + f_y_back - f_y_front + f_z_top - f_z_bot);
        }

        __syncthreads();
    }
}

double l2norm(int nx, int ny, int nz, float dx, float dy, float dz,
              float L, float alpha, float t_final, const float *u) {
    double sum_sq = 0.0;
    float pi_l = (float)M_PI / L;
    float decay = expf(-3.0f * alpha * pi_l * pi_l * t_final);

    for (int k = 0; k < nz; k++) {
        float z = (k + 1.0f) * dz;
        for (int j = 0; j < ny; j++) {
            float y = (j + 1.0f) * dy;
            for (int i = 0; i < nx; i++) {
                float x = (i + 1.0f) * dx;
                size_t idx = IDX3(i, j, k, nx, ny);

                float u_exact = sinf(pi_l * x) * sinf(pi_l * y) * sinf(pi_l * z) * decay;
                float delta_u = u[idx] - u_exact;
                sum_sq += (double)(delta_u * delta_u);
            }
        }
    }

    size_t total_cells = (size_t)nx * (size_t)ny * (size_t)nz;
    return sqrt(sum_sq / (double)total_cells);
}

__global__ void reduction_averages_cuda_kernel(const int nx, const int ny, const int nz,
                                               const float *__restrict__ u,
                                               float *__restrict__ d_block_tot) {
    extern __shared__ float s_tot[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = (tz * blockDim.y + ty) * blockDim.x + tx;
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;

    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;
    int k = blockIdx.z * blockDim.z + tz;

    float local_tot = (i < nx && j < ny && k < nz) ? u[IDX3(i, j, k, nx, ny)] : 0.0f;
    s_tot[tid] = local_tot;
    __syncthreads();

    for (int stride = threads_per_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_tot[tid] += s_tot[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        int block_id = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        d_block_tot[block_id] = s_tot[0];
    }
}

__global__ void aggregate_averages_cuda_kernel(const int num_blocks, const int step,
                                              const size_t total_cells,
                                              const float *__restrict__ d_block_tot,
                                              float *__restrict__ d_avg_evo) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double sum_tot = 0.0;
        for (int b = 0; b < num_blocks; b++) {
            sum_tot += (double)d_block_tot[b];
        }
        d_avg_evo[step] = (float)(sum_tot / (double)total_cells);
    }
}

int main(int argc, char *argv[]) {
    int n = 32;
    int nsteps = 0;
    int block_size = 256;
    int diag_stride = 0;
    char output_csv[256] = "../data/mms_cuda_n32.csv";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) nsteps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--block_size") == 0 && i + 1 < argc) block_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--diag_stride") == 0 && i + 1 < argc) diag_stride = atoi(argv[++i]);
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) strncpy(output_csv, argv[++i], sizeof(output_csv)-1);
    }

    float L = MMS_L_DEF;
    float alpha_val = MMS_ALPHA_DEF;
    float dx = L / (float)(n + 1);
    float dy = dx, dz = dx;

    float r = 0.80f / 6.0f; // Safety factor 0.80 of the 3D stability limit dt <= dx^2/(6*alpha)
    float dt = r * (dx * dx) / alpha_val;
    float tau = (L * L) / (3.0f * (float)(M_PI * M_PI) * alpha_val);

    if (nsteps <= 0) nsteps = (int)ceilf((1.5f * tau) / dt);
    float t_final = nsteps * dt;

    if (diag_stride <= 0) diag_stride = (nsteps > 500) ? (nsteps / 500) : 1;

    float inv_dx2 = dt / (dx * dx);
    float inv_dy2 = dt / (dy * dy);
    float inv_dz2 = dt / (dz * dz);

    size_t total_cells = (size_t)n * (size_t)n * (size_t)n;
    size_t grid_bytes = total_cells * sizeof(float);

    dim3 blockDimInit(16, 8, 2);
    dim3 gridDimInit((n + blockDimInit.x - 1) / blockDimInit.x,
                     (n + blockDimInit.y - 1) / blockDimInit.y,
                     (n + blockDimInit.z - 1) / blockDimInit.z);
    int num_blocks = gridDimInit.x * gridDimInit.y * gridDimInit.z;

    dim3 blockDim2D;
    get_2d_tile_dims(block_size, &blockDim2D);
    dim3 gridDim2D((n + blockDim2D.x - 1) / blockDim2D.x,
                   (n + blockDim2D.y - 1) / blockDim2D.y);

    int pitch = blockDim2D.x + 2 * RADIUS;
    int slice = pitch * (blockDim2D.y + 2 * RADIUS);
    size_t shared_mem_bytes_stencil = 2 * slice * sizeof(float);

    float *d_alpha, *d_u, *d_u_next;
    CUDA_CHECK(cudaMalloc((void **)&d_alpha, grid_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_u, grid_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_u_next, grid_bytes));

    float *d_block_tot, *d_avg_evo;
    CUDA_CHECK(cudaMalloc((void **)&d_block_tot, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_avg_evo, (nsteps + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_avg_evo, 0, (nsteps + 1) * sizeof(float)));

    init_field_cuda_kernel<<<gridDimInit, blockDimInit>>>(n, n, n, dx, dy, dz, L, alpha_val, d_alpha, d_u);
    CUDA_CHECK(cudaGetLastError());

    reduction_averages_cuda_kernel<<<gridDimInit, blockDimInit, 256 * sizeof(float)>>>(n, n, n, d_u, d_block_tot);
    CUDA_CHECK(cudaGetLastError());
    aggregate_averages_cuda_kernel<<<1, 256>>>(num_blocks, 0, total_cells, d_block_tot, d_avg_evo);
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < nsteps; t++) {
        solve_stencil_cuda_optimized_kernel<<<gridDim2D, blockDim2D, shared_mem_bytes_stencil>>>(
            n, n, n, inv_dx2, inv_dy2, inv_dz2, d_alpha, d_u, d_u_next);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % diag_stride == 0) {
            reduction_averages_cuda_kernel<<<gridDimInit, blockDimInit, 256 * sizeof(float)>>>(
                n, n, n, d_u_next, d_block_tot);
            CUDA_CHECK(cudaGetLastError());

            aggregate_averages_cuda_kernel<<<1, 256>>>(
                num_blocks, t + 1, total_cells, d_block_tot, d_avg_evo);
            CUDA_CHECK(cudaGetLastError());
        }

        float *tmp = d_u;
        d_u = d_u_next;
        d_u_next = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float solve_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&solve_time_ms, start, stop));
    double elapsed_sec = (double)solve_time_ms / 1000.0;

    float *h_u = (float *)malloc(grid_bytes);
    CUDA_CHECK(cudaMemcpy(h_u, d_u, grid_bytes, cudaMemcpyDeviceToHost));

    float *h_avg_evo = (float *)malloc((nsteps + 1) * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_avg_evo, d_avg_evo, (nsteps + 1) * sizeof(float), cudaMemcpyDeviceToHost));

    double l2_err = l2norm(n, n, n, dx, dy, dz, L, alpha_val, t_final, h_u);

    printf("========================================================\n");
    printf("  3D Method of Manufactured Solutions (MMS) - CUDA GPU\n");
    printf("========================================================\n");
    printf("Grid: %d x %d x %d (%zu cells) | Block: %d (%dx%d) | dx: %.5f | dt: %.6f s\n",
           n, n, n, total_cells, block_size, blockDim2D.x, blockDim2D.y, dx, dt);
    printf("Diffusivity alpha: %.4f | Characteristic Time: %.4f s | Steps: %d\n", alpha_val, tau, nsteps);
    printf("Update Factor r: %.4f | Diagnostic stride: %d\n", r, diag_stride);
    printf("--------------------------------------------------------\n");
    printf("Results:\n");
    printf("  Final Simulation Time : %.4f s\n", t_final);
    printf("  L2-Norm Error         : %.6E\n", l2_err);
    printf("  Solve Wall-Clock Time : %.4f s\n", elapsed_sec);
    printf("========================================================\n\n");

    if (output_csv[0] != '\0') {
        FILE *fp = fopen(output_csv, "w");
        if (fp) {
            fprintf(fp, "step,time,u_avg,u_exact_avg\n");
            float pi_l = (float)M_PI / L;
            for (int t = 0; t <= nsteps; t += diag_stride) {
                float curr_t = (float)t * dt;
                float u_exact_val = h_avg_evo[0] * expf(-3.0f * alpha_val * pi_l * pi_l * curr_t);
                fprintf(fp, "%d,%.6f,%.6f,%.6f\n", t, curr_t, h_avg_evo[t], u_exact_val);
            }
            fclose(fp);
            printf("MMS evolution saved to %s\n\n", output_csv);
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_alpha));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaFree(d_block_tot));
    CUDA_CHECK(cudaFree(d_avg_evo));
    free(h_u);
    free(h_avg_evo);

    return 0;
}
