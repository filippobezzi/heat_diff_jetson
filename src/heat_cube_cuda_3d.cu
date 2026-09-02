#include "heat_3d.h"
#include <cuda_runtime.h>

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
                                        const float cube_x_min, const float cube_x_max,
                                        const float cube_y_min, const float cube_y_max,
                                        const float cube_z_min, const float cube_z_max,
                                        const float alpha_cube, const float alpha_water,
                                        const float t_cube_0, const float t_water_0,
                                        float *__restrict__ alpha,
                                        float *__restrict__ u) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        float x = (i + 0.5f) * dx;
        float y = (j + 0.5f) * dy;
        float z = (k + 0.5f) * dz;
        size_t idx = IDX3(i, j, k, nx, ny);

        if (x >= cube_x_min && x <= cube_x_max &&
            y >= cube_y_min && y <= cube_y_max &&
            z >= cube_z_min && z <= cube_z_max) {
            alpha[idx] = alpha_cube;
            u[idx] = t_cube_0;
        } else {
            alpha[idx] = alpha_water;
            u[idx] = t_water_0;
        }
    }
}

#define RADIUS 1

// Optimal 2.5D Shared Memory + Register Sliding Window Stencil Kernel with On-the-Fly Harmonic Means
__global__ void solve_stencil_cuda_kernel(const int nx, const int ny, const int nz,
                                           const float inv_dx2, const float inv_dy2, const float inv_dz2,
                                           const float *__restrict__ alpha,
                                           const float *__restrict__ u,
                                           float *__restrict__ u_next) {
    extern __shared__ float s_mem[];

    int tile_x = blockDim.x;
    int tile_y = blockDim.y;
    int pitch = tile_x + 2 * RADIUS;
    int slice = pitch * (tile_y + 2 * RADIUS);
    // allocate one contigous array for u and alpha in shared memory
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
        a_top = (k + 1 < nz && i < nx && j < ny) ? alpha[IDX3(i, j, k + 1, nx, ny)] : 0.0f;

        // Load 2D XY slice into shared memory
        if (i < nx && j < ny) {
            s_u[sy * pitch + sx]     = u_curr;
            s_alpha[sy * pitch + sx] = a_curr;
        } else {
            s_u[sy * pitch + sx]     = 0.0f;
            s_alpha[sy * pitch + sx] = 0.0f;
        }

        // Load X Halos
        if (tx == 0) {
            s_u[sy * pitch + 0]     = (i > 0 && j < ny) ? u[IDX3(i - 1, j, k, nx, ny)] : 0.0f;
            s_alpha[sy * pitch + 0] = (i > 0 && j < ny) ? alpha[IDX3(i - 1, j, k, nx, ny)] : 0.0f;
        }
        if (tx == tile_x - 1) {
            s_u[sy * pitch + (tile_x + 1)]     = (i + 1 < nx && j < ny) ? u[IDX3(i + 1, j, k, nx, ny)] : 0.0f;
            s_alpha[sy * pitch + (tile_x + 1)] = (i + 1 < nx && j < ny) ? alpha[IDX3(i + 1, j, k, nx, ny)] : 0.0f;
        }

        // Load Y Halos
        if (ty == 0) {
            s_u[0 * pitch + sx]     = (j > 0 && i < nx) ? u[IDX3(i, j - 1, k, nx, ny)] : 0.0f;
            s_alpha[0 * pitch + sx] = (j > 0 && i < nx) ? alpha[IDX3(i, j - 1, k, nx, ny)] : 0.0f;
        }
        if (ty == tile_y - 1) {
            s_u[(tile_y + 1) * pitch + sx]     = (j + 1 < ny && i < nx) ? u[IDX3(i, j + 1, k, nx, ny)] : 0.0f;
            s_alpha[(tile_y + 1) * pitch + sx] = (j + 1 < ny && i < nx) ? alpha[IDX3(i, j + 1, k, nx, ny)] : 0.0f;
        }

        __syncthreads();

        // Compute 3D 7-point heterogeneous stencil with on-the-fly interface harmonic means
        if (i < nx && j < ny) {
            size_t idx = IDX3(i, j, k, nx, ny);

            float ax_left  = (i > 0)      ? harmonic_mean(a_curr, s_alpha[sy * pitch + (sx - 1)]) : 0.0f;
            float ax_right = (i < nx - 1) ? harmonic_mean(a_curr, s_alpha[sy * pitch + (sx + 1)]) : 0.0f;

            float ay_front = (j > 0)      ? harmonic_mean(a_curr, s_alpha[(sy - 1) * pitch + sx]) : 0.0f;
            float ay_back  = (j < ny - 1) ? harmonic_mean(a_curr, s_alpha[(sy + 1) * pitch + sx]) : 0.0f;

            float az_bot   = (k > 0)      ? harmonic_mean(a_curr, a_bot) : 0.0f;
            float az_top   = (k < nz - 1) ? harmonic_mean(a_curr, a_top) : 0.0f;

            float f_x_left  = ax_left  * (s_u[sy * pitch + sx] - s_u[sy * pitch + (sx - 1)]) * inv_dx2;
            float f_x_right = ax_right * (s_u[sy * pitch + (sx + 1)] - s_u[sy * pitch + sx]) * inv_dx2;

            float f_y_front = ay_front * (s_u[sy * pitch + sx] - s_u[(sy - 1) * pitch + sx]) * inv_dy2;
            float f_y_back  = ay_back  * (s_u[(sy + 1) * pitch + sx] - s_u[sy * pitch + sx]) * inv_dy2;

            float f_z_bot   = az_bot   * (u_curr - u_bot) * inv_dz2;
            float f_z_top   = az_top   * (u_top - u_curr) * inv_dz2;

            u_next[idx] = u_curr + (f_x_right - f_x_left + f_y_back - f_y_front + f_z_top - f_z_bot);
        }	// 48 FLOPS

        __syncthreads();
    }
}

__global__ void reduction_averages_cuda_kernel(const int nx, const int ny, const int nz,
                                               const float dx, const float dy, const float dz,
                                               const float cube_x_min, const float cube_x_max,
                                               const float cube_y_min, const float cube_y_max,
                                               const float cube_z_min, const float cube_z_max,
                                               const float *__restrict__ u,
                                               float *__restrict__ d_block_tot,
                                               float *__restrict__ d_block_cube) {
    extern __shared__ float s_mem[];	// dynamic memory allocation
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    // allocate one contigous array two compute two sums
    // [ (0, ...,  threads_per_block - 1), (threads_per_block, ..., 2 * threads_per_block - 1) ]
    float *s_tot  = s_mem;
    float *s_cube = &s_mem[threads_per_block];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = (tz * blockDim.y + ty) * blockDim.x + tx;

    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;
    int k = blockIdx.z * blockDim.z + tz;

    float local_tot = 0.0f;
    float local_cube = 0.0f;

    if (i < nx && j < ny && k < nz) {
        size_t idx = IDX3(i, j, k, nx, ny);
        float val = u[idx];
        float x = (i + 0.5f) * dx;
        float y = (j + 0.5f) * dy;
        float z = (k + 0.5f) * dz;

        local_tot = val;
        if (x >= cube_x_min && x <= cube_x_max &&
            y >= cube_y_min && y <= cube_y_max &&
            z >= cube_z_min && z <= cube_z_max) {
            local_cube = val;
        }
    }

    s_tot[tid] = local_tot;
    s_cube[tid] = local_cube;
    __syncthreads();

    for (int stride = threads_per_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_tot[tid]  += s_tot[tid + stride];
            s_cube[tid] += s_cube[tid + stride];
        }
        __syncthreads();
    }
    // assign local sum to thread 0 in each block
    if (tid == 0) {
        int block_id = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        d_block_tot[block_id]  = s_tot[0];
        d_block_cube[block_id] = s_cube[0];
    }
}

__global__ void aggregate_averages_cuda_kernel(const int num_blocks, const int step,
                                              const size_t total_cells, const int n_cube,
                                              const float *__restrict__ d_block_tot,
                                              const float *__restrict__ d_block_cube,
                                              float *__restrict__ d_avg_tot_evo,
                                              float *__restrict__ d_avg_cube_evo) {

    // assign task to only one thread to avoid block synchronization
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double sum_tot = 0.0;
        double sum_cube = 0.0;
        for (int b = 0; b < num_blocks; b++) {
            sum_tot  += (double)d_block_tot[b];
            sum_cube += (double)d_block_cube[b];
        }
        d_avg_tot_evo[step]  = (float)(sum_tot / (double)total_cells);
        d_avg_cube_evo[step] = (n_cube > 0) ? (float)(sum_cube / (double)n_cube) : 0.0f;
    }
}



void save_evolution_to_csv(const char *filename, const int nsteps, const float dt,
                           const float *u_avg_total, const float *u_avg_cube, const int stride) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open %s for writing\n", filename);
        return;
    }
    fprintf(fp, "step,time,u_avg_total,u_avg_cube\n");
    for (int t = 0; t <= nsteps; t += stride) {
        float current_time = t * dt;
        fprintf(fp, "%d,%.6f,%.6f,%.6f\n", t, current_time, u_avg_total[t], u_avg_cube[t]);
    }
    fclose(fp);
    printf("Evolution saved to %s\n", filename);
}

int main(int argc, char *argv[]) {
    SimulationConfig cfg;
    init_default_config(&cfg, 64);
    parse_arguments(argc, argv, &cfg);

    size_t total_cells = (size_t)cfg.nx * (size_t)cfg.ny * (size_t)cfg.nz;
    size_t grid_bytes = total_cells * sizeof(float);
    size_t evo_bytes  = (cfg.nsteps + 1) * sizeof(float);

    int n_cube = 0;
    for (int k = 0; k < cfg.nz; k++) {
        float z = (k + 0.5f) * cfg.dz;
        for (int j = 0; j < cfg.ny; j++) {
            float y = (j + 0.5f) * cfg.dy;
            for (int i = 0; i < cfg.nx; i++) {
                float x = (i + 0.5f) * cfg.dx;
                if (x >= cfg.cube_x_min && x <= cfg.cube_x_max &&
                    y >= cfg.cube_y_min && y <= cfg.cube_y_max &&
                    z >= cfg.cube_z_min && z <= cfg.cube_z_max) {
                    n_cube++;
                }
            }
        }
    }

    dim3 blockDim;
    get_3d_block_dims(cfg.block_size, &blockDim);
    dim3 gridDim((cfg.nx + blockDim.x - 1) / blockDim.x,
                 (cfg.ny + blockDim.y - 1) / blockDim.y,
                 (cfg.nz + blockDim.z - 1) / blockDim.z);

    dim3 blockDim2D;
    get_2d_tile_dims(cfg.block_size, &blockDim2D);
    dim3 gridDim2D((cfg.nx + blockDim2D.x - 1) / blockDim2D.x,
                   (cfg.ny + blockDim2D.y - 1) / blockDim2D.y);

    int num_blocks = gridDim.x * gridDim.y * gridDim.z;
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    size_t shared_mem_bytes_red = 2 * threads_per_block * sizeof(float);// unsigned 64-bit integer to avoid overflow

    size_t shared_mem_bytes_stencil = 2 * (blockDim2D.x + 2 * RADIUS) * (blockDim2D.y + 2 * RADIUS) * sizeof(float);

    printf("========================================================\n");
    printf("  3D Heat Transfer Solver - CUDA Parallel GPU (2.5D Tiling)\n");
    printf("========================================================\n");
    printf("Domain: %.1f x %.1f x %.1f mm | Grid: %d x %d x %d (%zu cells)\n",
           cfg.lx, cfg.ly, cfg.lz, cfg.nx, cfg.ny, cfg.nz, total_cells);
    printf("Spatial spacing: dx=%.4f mm, dy=%.4f mm, dz=%.4f mm\n", cfg.dx, cfg.dy, cfg.dz);
    printf("Time stepping: dt=%.6f s | Total steps: %d | Time scale: %.2f s\n",
           cfg.dt, cfg.nsteps, cfg.time_scale);
    printf("CUDA 2.5D Stencil Config: Tile (%d x %d) | Grid (%d x %d)\n",
           blockDim2D.x, blockDim2D.y, gridDim2D.x, gridDim2D.y);
    printf("Cube cells: %d (%.2f%% of volume)\n", n_cube, 100.0f * (float)n_cube / (float)total_cells);
    printf("--------------------------------------------------------\n");

    float *d_alpha;
    float *d_u, *d_u_next;
    float *d_avg_total_evo, *d_avg_cube_evo;
    float *d_block_tot, *d_block_cube;

    CUDA_CHECK(cudaMalloc((void **)&d_alpha, grid_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_u, grid_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_u_next, grid_bytes));

    CUDA_CHECK(cudaMalloc((void **)&d_avg_total_evo, evo_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_avg_cube_evo, evo_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_block_tot, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_block_cube, num_blocks * sizeof(float)));

    init_field_cuda_kernel<<<gridDim, blockDim>>>(
        cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz,
        cfg.cube_x_min, cfg.cube_x_max, cfg.cube_y_min, cfg.cube_y_max, cfg.cube_z_min, cfg.cube_z_max,
        cfg.alpha_cube, cfg.alpha_water, cfg.t_cube_0, cfg.t_water_0,
        d_alpha, d_u);
    CUDA_CHECK(cudaGetLastError());

    reduction_averages_cuda_kernel<<<gridDim, blockDim, shared_mem_bytes_red>>>(
        cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz,
        cfg.cube_x_min, cfg.cube_x_max, cfg.cube_y_min, cfg.cube_y_max, cfg.cube_z_min, cfg.cube_z_max,
        d_u, d_block_tot, d_block_cube);
    CUDA_CHECK(cudaGetLastError());

    aggregate_averages_cuda_kernel<<<1, 256>>>(
        num_blocks, 0, total_cells, n_cube,
        d_block_tot, d_block_cube, d_avg_total_evo, d_avg_cube_evo);
    CUDA_CHECK(cudaGetLastError());

    const float inv_dx2 = cfg.dt / (cfg.dx * cfg.dx);
    const float inv_dy2 = cfg.dt / (cfg.dy * cfg.dy);
    const float inv_dz2 = cfg.dt / (cfg.dz * cfg.dz);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int t = 0; t < cfg.nsteps; t++) {
        solve_stencil_cuda_kernel<<<gridDim2D, blockDim2D, shared_mem_bytes_stencil>>>(
            cfg.nx, cfg.ny, cfg.nz, inv_dx2, inv_dy2, inv_dz2,
            d_alpha, d_u, d_u_next);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % cfg.stride == 0 || (t + 1) == cfg.nsteps) {
            reduction_averages_cuda_kernel<<<gridDim, blockDim, shared_mem_bytes_red>>>(
                cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz,
                cfg.cube_x_min, cfg.cube_x_max, cfg.cube_y_min, cfg.cube_y_max, cfg.cube_z_min, cfg.cube_z_max,
                d_u_next, d_block_tot, d_block_cube);
            CUDA_CHECK(cudaGetLastError());

            aggregate_averages_cuda_kernel<<<1, 256>>>(
                num_blocks, t + 1, total_cells, n_cube,
                d_block_tot, d_block_cube, d_avg_total_evo, d_avg_cube_evo);
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

    char cfg_label[64];
    snprintf(cfg_label, sizeof(cfg_label), "block=%d (tile %dx%d)",
             cfg.block_size, blockDim2D.x, blockDim2D.y);
    print_performance("CUDA GPU", cfg_label, total_cells, cfg.nsteps, elapsed_sec);

    float *h_avg_total = (float *)malloc(evo_bytes);
    float *h_avg_cube  = (float *)malloc(evo_bytes);

    CUDA_CHECK(cudaMemcpy(h_avg_total, d_avg_total_evo, evo_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_avg_cube, d_avg_cube_evo, evo_bytes, cudaMemcpyDeviceToHost));

    save_evolution_to_csv(cfg.output_csv, cfg.nsteps, cfg.dt, h_avg_total, h_avg_cube, cfg.stride);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_alpha));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaFree(d_avg_total_evo));
    CUDA_CHECK(cudaFree(d_avg_cube_evo));
    CUDA_CHECK(cudaFree(d_block_tot));
    CUDA_CHECK(cudaFree(d_block_cube));
    free(h_avg_total);
    free(h_avg_cube);

    return 0;
}
