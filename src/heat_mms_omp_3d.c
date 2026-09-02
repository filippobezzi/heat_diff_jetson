#include "heat_3d.h"
#include <omp.h>

#define MMS_L_DEF 1.0f
#define MMS_ALPHA_DEF 0.143f

float exact_field(float x, float y, float z, float t, float alpha, float L) {
    float pi_l = (float)M_PI / L;
    return sinf(pi_l * x) * sinf(pi_l * y) * sinf(pi_l * z) * expf(-3.0f * alpha * pi_l * pi_l * t);
}

void init_field(int nx, int ny, int nz, float dx, float dy, float dz, float L,
                float alpha_val, float *alpha, float *u) {
    float pi_l = (float)M_PI / L;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                float z = (k + 1.0f) * dz;
                float y = (j + 1.0f) * dy;
                float x = (i + 1.0f) * dx;
                size_t idx = IDX3(i, j, k, nx, ny);
                alpha[idx] = alpha_val;
                u[idx] = sinf(pi_l * x) * sinf(pi_l * y) * sinf(pi_l * z);
            }
        }
    }
}

void compute_alpha(int nx, int ny, int nz, const float *alpha,
                   float *alpha_x, float *alpha_y, float *alpha_z) {
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                size_t idx = IDX3(i, j, k, nx, ny);
                float a_curr = alpha[idx];

                if (i < nx - 1) {
                    alpha_x[idx] = harmonic_mean(a_curr, alpha[IDX3(i + 1, j, k, nx, ny)]);
                } else {
                    alpha_x[idx] = a_curr;
                }
                if (j < ny - 1) {
                    alpha_y[idx] = harmonic_mean(a_curr, alpha[IDX3(i, j + 1, k, nx, ny)]);
                } else {
                    alpha_y[idx] = a_curr;
                }
                if (k < nz - 1) {
                    alpha_z[idx] = harmonic_mean(a_curr, alpha[IDX3(i, j, k + 1, nx, ny)]);
                } else {
                    alpha_z[idx] = a_curr;
                }
            }
        }
    }
}

void solve_stencil(int nx, int ny, int nz,
                   float inv_dx2, float inv_dy2, float inv_dz2,
                   const float *__restrict__ alpha_x,
                   const float *__restrict__ alpha_y,
                   const float *__restrict__ alpha_z,
                   const float *__restrict__ u,
                   float *__restrict__ u_next) {
    #pragma omp parallel for collapse(3) schedule(static)
    for (int kz = 0; kz < nz; kz += OMP_TILE_Z) {
        for (int jy = 0; jy < ny; jy += OMP_TILE_Y) {
            for (int ix = 0; ix < nx; ix += OMP_TILE_X) {
                const int k_end = (kz + OMP_TILE_Z < nz) ? (kz + OMP_TILE_Z) : nz;
                const int j_end = (jy + OMP_TILE_Y < ny) ? (jy + OMP_TILE_Y) : ny;
                const int i_end = (ix + OMP_TILE_X < nx) ? (ix + OMP_TILE_X) : nx;

                for (int k = kz; k < k_end; k++) {
                    for (int j = jy; j < j_end; j++) {
                        for (int i = ix; i < i_end; i++) {
                            size_t idx = IDX3(i, j, k, nx, ny);
                            float u_curr = u[idx];

                            float f_x_left  = (i > 0)      ? alpha_x[IDX3(i - 1, j, k, nx, ny)] * (u_curr - u[IDX3(i - 1, j, k, nx, ny)]) * inv_dx2 : alpha_x[idx] * u_curr * inv_dx2;
                            float f_x_right = (i < nx - 1) ? alpha_x[idx]                        * (u[IDX3(i + 1, j, k, nx, ny)] - u_curr) * inv_dx2 : alpha_x[idx] * (0.0f - u_curr) * inv_dx2;

                            float f_y_front = (j > 0)      ? alpha_y[IDX3(i, j - 1, k, nx, ny)] * (u_curr - u[IDX3(i, j - 1, k, nx, ny)]) * inv_dy2 : alpha_y[idx] * u_curr * inv_dy2;
                            float f_y_back  = (j < ny - 1) ? alpha_y[idx]                        * (u[IDX3(i, j + 1, k, nx, ny)] - u_curr) * inv_dy2 : alpha_y[idx] * (0.0f - u_curr) * inv_dy2;

                            float f_z_bot   = (k > 0)      ? alpha_z[IDX3(i, j, k - 1, nx, ny)] * (u_curr - u[IDX3(i, j, k - 1, nx, ny)]) * inv_dz2 : alpha_z[idx] * u_curr * inv_dz2;
                            float f_z_top   = (k < nz - 1) ? alpha_z[idx]                        * (u[IDX3(i, j, k + 1, nx, ny)] - u_curr) * inv_dz2 : alpha_z[idx] * (0.0f - u_curr) * inv_dz2;

                            u_next[idx] = u_curr + (f_x_right - f_x_left + f_y_back - f_y_front + f_z_top - f_z_bot);
                        }
                    }
                }
            }
        }
    }
}

double l2norm(int nx, int ny, int nz, float dx, float dy, float dz,
              float L, float alpha, float t_final, const float *u) {
    double sum_sq = 0.0;
    float pi_l = (float)M_PI / L;
    float decay = expf(-3.0f * alpha * pi_l * pi_l * t_final);

    #pragma omp parallel for collapse(3) reduction(+:sum_sq) schedule(static)
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                float z = (k + 1.0f) * dz;
                float y = (j + 1.0f) * dy;
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

float compute_domain_average(int nx, int ny, int nz, const float *u) {
    double sum = 0.0;
    size_t total = (size_t)nx * (size_t)ny * (size_t)nz;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t i = 0; i < total; i++) {
        sum += (double)u[i];
    }
    return (float)(sum / (double)total);
}

int main(int argc, char *argv[]) {
    int n = 32;
    int nsteps = 0;
    int num_threads = 4;
    int diag_stride = 0;
    char output_csv[256] = "../data/mms_omp_3d.csv";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) nsteps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--diag_stride") == 0 && i + 1 < argc) diag_stride = atoi(argv[++i]);
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) strncpy(output_csv, argv[++i], sizeof(output_csv)-1);
    }

    omp_set_num_threads(num_threads);

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

    float *alpha   = (float *)malloc(grid_bytes);
    float *alpha_x = (float *)calloc(total_cells, sizeof(float));
    float *alpha_y = (float *)calloc(total_cells, sizeof(float));
    float *alpha_z = (float *)calloc(total_cells, sizeof(float));
    float *u       = (float *)malloc(grid_bytes);
    float *u_next  = (float *)malloc(grid_bytes);
    float *evo_avg = (float *)malloc((nsteps + 1) * sizeof(float));
    float *evo_exact = (float *)malloc((nsteps + 1) * sizeof(float));

    init_field(n, n, n, dx, dy, dz, L, alpha_val, alpha, u);
    compute_alpha(n, n, n, alpha, alpha_x, alpha_y, alpha_z);

    float u_0_avg = compute_domain_average(n, n, n, u);
    evo_avg[0] = u_0_avg;
    evo_exact[0] = u_0_avg;

    double t_start = get_wall_time();

    for (int t = 0; t < nsteps; t++) {
        solve_stencil(n, n, n, inv_dx2, inv_dy2, inv_dz2, alpha_x, alpha_y, alpha_z, u, u_next);
        float *tmp = u; u = u_next; u_next = tmp;

        if ((t + 1) % diag_stride == 0) {
            float curr_t = (t + 1) * dt;
            float pi_l = (float)M_PI / L;
            evo_avg[t + 1] = compute_domain_average(n, n, n, u);
            evo_exact[t + 1] = u_0_avg * expf(-3.0f * alpha_val * pi_l * pi_l * curr_t);
        }
    }

    double t_end = get_wall_time();
    double l2_err = l2norm(n, n, n, dx, dy, dz, L, alpha_val, t_final, u);

    printf("========================================================\n");
    printf("  3D Method of Manufactured Solutions (MMS) - OpenMP CPU\n");
    printf("========================================================\n");
    printf("Grid: %d x %d x %d (%zu cells) | Threads: %d | dx: %.5f | dt: %.6f s\n",
           n, n, n, total_cells, num_threads, dx, dt);
    printf("Diffusivity alpha: %.4f | Characteristic Time: %.4f s | Steps: %d\n", alpha_val, tau, nsteps);
    printf("Update Factor r: %.4f | Diagnostic stride: %d\n", r, diag_stride);
    printf("--------------------------------------------------------\n");
    printf("Results:\n");
    printf("  Final Simulation Time : %.4f s\n", t_final);
    printf("  L2-Norm Error         : %.6E\n", l2_err);
    printf("  Solve Wall-Clock Time : %.4f s\n", t_end - t_start);
    printf("========================================================\n");

    FILE *fp = fopen(output_csv, "w");
    if (fp) {
        fprintf(fp, "step,time,u_avg,u_exact_avg\n");
        for (int t = 0; t <= nsteps; t += diag_stride) {
            fprintf(fp, "%d,%.6f,%.6f,%.6f\n", t, t * dt, evo_avg[t], evo_exact[t]);
        }
        fclose(fp);
        printf("MMS evolution saved to %s\n\n", output_csv);
    }

    free(alpha); free(alpha_x); free(alpha_y); free(alpha_z);
    free(u); free(u_next); free(evo_avg); free(evo_exact);
    return 0;
}
