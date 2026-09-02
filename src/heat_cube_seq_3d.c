#include "heat_3d.h"

void init_field(const SimulationConfig *cfg, float *alpha, float *u) {
    for (int k = 0; k < cfg->nz; k++) {
        float z = (k + 0.5f) * cfg->dz;
        for (int j = 0; j < cfg->ny; j++) {
            float y = (j + 0.5f) * cfg->dy;
            for (int i = 0; i < cfg->nx; i++) {
                float x = (i + 0.5f) * cfg->dx;
                size_t idx = IDX3(i, j, k, cfg->nx, cfg->ny);

                if (x >= cfg->cube_x_min && x <= cfg->cube_x_max &&
                    y >= cfg->cube_y_min && y <= cfg->cube_y_max &&
                    z >= cfg->cube_z_min && z <= cfg->cube_z_max) {
                    alpha[idx] = cfg->alpha_cube;
                    u[idx] = cfg->t_cube_0;
                } else {
                    alpha[idx] = cfg->alpha_water;
                    u[idx] = cfg->t_water_0;
                }
            }
        }
    }
}

void compute_alpha(const SimulationConfig *cfg, const float *alpha,
                   float *alpha_x, float *alpha_y, float *alpha_z) {
    for (int k = 0; k < cfg->nz; k++) {
        for (int j = 0; j < cfg->ny; j++) {
            for (int i = 0; i < cfg->nx; i++) {
                size_t idx = IDX3(i, j, k, cfg->nx, cfg->ny);
                float a_curr = alpha[idx];

                if (i < cfg->nx - 1) {
                    alpha_x[idx] = harmonic_mean(a_curr, alpha[IDX3(i + 1, j, k, cfg->nx, cfg->ny)]);
                }
                if (j < cfg->ny - 1) {
                    alpha_y[idx] = harmonic_mean(a_curr, alpha[IDX3(i, j + 1, k, cfg->nx, cfg->ny)]);
                }
                if (k < cfg->nz - 1) {
                    alpha_z[idx] = harmonic_mean(a_curr, alpha[IDX3(i, j, k + 1, cfg->nx, cfg->ny)]);
                }
            }
        }
    }
}

void compute_averages(const SimulationConfig *cfg, const float *u,
                      float *u_avg_total, float *u_avg_cube, int n_cube) {
    double sum_total = 0.0;
    double sum_cube = 0.0;

    for (int k = 0; k < cfg->nz; k++) {
        float z = (k + 0.5f) * cfg->dz;
        for (int j = 0; j < cfg->ny; j++) {
            float y = (j + 0.5f) * cfg->dy;
            for (int i = 0; i < cfg->nx; i++) {
                float x = (i + 0.5f) * cfg->dx;
                size_t idx = IDX3(i, j, k, cfg->nx, cfg->ny);
                float val = u[idx];

                sum_total += (double)val;
                if (x >= cfg->cube_x_min && x <= cfg->cube_x_max &&
                    y >= cfg->cube_y_min && y <= cfg->cube_y_max &&
                    z >= cfg->cube_z_min && z <= cfg->cube_z_max) {
                    sum_cube += (double)val;
                }
            }
        }
    }

    size_t total_cells = (size_t)cfg->nx * (size_t)cfg->ny * (size_t)cfg->nz;
    *u_avg_total = (float)(sum_total / (double)total_cells);
    *u_avg_cube  = (n_cube > 0) ? (float)(sum_cube / (double)n_cube) : 0.0f;
}

void solve_stencil(const SimulationConfig *cfg,
                   const float *__restrict__ alpha_x,
                   const float *__restrict__ alpha_y,
                   const float *__restrict__ alpha_z,
                   const float *__restrict__ u,
                   float *__restrict__ u_next) {
    const int nx = cfg->nx;
    const int ny = cfg->ny;
    const int nz = cfg->nz;

    const float inv_dx2 = cfg->dt / (cfg->dx * cfg->dx);
    const float inv_dy2 = cfg->dt / (cfg->dy * cfg->dy);
    const float inv_dz2 = cfg->dt / (cfg->dz * cfg->dz);

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                size_t idx = IDX3(i, j, k, nx, ny);
                float u_curr = u[idx];

                float f_x_left  = (i > 0)      ? alpha_x[IDX3(i - 1, j, k, nx, ny)] * (u_curr - u[IDX3(i - 1, j, k, nx, ny)]) * inv_dx2 : 0.0f;
                float f_x_right = (i < nx - 1) ? alpha_x[idx]                        * (u[IDX3(i + 1, j, k, nx, ny)] - u_curr) * inv_dx2 : 0.0f;

                float f_y_front = (j > 0)      ? alpha_y[IDX3(i, j - 1, k, nx, ny)] * (u_curr - u[IDX3(i, j - 1, k, nx, ny)]) * inv_dy2 : 0.0f;
                float f_y_back  = (j < ny - 1) ? alpha_y[idx]                        * (u[IDX3(i, j + 1, k, nx, ny)] - u_curr) * inv_dy2 : 0.0f;

                float f_z_bot   = (k > 0)      ? alpha_z[IDX3(i, j, k - 1, nx, ny)] * (u_curr - u[IDX3(i, j, k - 1, nx, ny)]) * inv_dz2 : 0.0f;
                float f_z_top   = (k < nz - 1) ? alpha_z[idx]                        * (u[IDX3(i, j, k + 1, nx, ny)] - u_curr) * inv_dz2 : 0.0f;

                u_next[idx] = u_curr + (f_x_right - f_x_left + f_y_back - f_y_front + f_z_top - f_z_bot);
            }
        }
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

    float *alpha   = (float *)malloc(grid_bytes);
    float *alpha_x = (float *)calloc(total_cells, sizeof(float));
    float *alpha_y = (float *)calloc(total_cells, sizeof(float));
    float *alpha_z = (float *)calloc(total_cells, sizeof(float));
    float *u       = (float *)malloc(grid_bytes);
    float *u_next  = (float *)malloc(grid_bytes);

    float *avg_total_evo = (float *)malloc((cfg.nsteps + 1) * sizeof(float));
    float *avg_cube_evo  = (float *)malloc((cfg.nsteps + 1) * sizeof(float));

    if (!alpha || !alpha_x || !alpha_y || !alpha_z || !u || !u_next || !avg_total_evo || !avg_cube_evo) {
        fprintf(stderr, "Memory allocation failed for grid size %d^3\n", cfg.nx);
        return EXIT_FAILURE;
    }

    init_field(&cfg, alpha, u);
    compute_alpha(&cfg, alpha, alpha_x, alpha_y, alpha_z);

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

    printf("========================================================\n");
    printf("  3D Heat Transfer Solver - Sequential C Baseline\n");
    printf("========================================================\n");
    printf("Domain: %.1f x %.1f x %.1f mm | Grid: %d x %d x %d (%zu cells)\n",
           cfg.lx, cfg.ly, cfg.lz, cfg.nx, cfg.ny, cfg.nz, total_cells);
    printf("Spatial spacing: dx=%.4f mm, dy=%.4f mm, dz=%.4f mm\n", cfg.dx, cfg.dy, cfg.dz);
    printf("Time stepping: dt=%.6f s | Total steps: %d | Time scale: %.2f s\n",
           cfg.dt, cfg.nsteps, cfg.time_scale);
    printf("Memory: %.2f MB for temperature fields\n", 2.0f * (float)grid_bytes / (1024.0f * 1024.0f));
    printf("--------------------------------------------------------\n");
    printf("Initialized 3D domain. Cube cells: %d (%.2f%% of total volume)\n",
           n_cube, 100.0f * (float)n_cube / (float)total_cells);

    compute_averages(&cfg, u, &avg_total_evo[0], &avg_cube_evo[0], n_cube);

    double t_start = get_wall_time();

    for (int t = 0; t < cfg.nsteps; t++) {
        solve_stencil(&cfg, alpha_x, alpha_y, alpha_z, u, u_next);

        if ((t + 1) % cfg.stride == 0 || (t + 1) == cfg.nsteps) {
            compute_averages(&cfg, u_next, &avg_total_evo[t + 1], &avg_cube_evo[t + 1], n_cube);
        }

        float *tmp = u;
        u = u_next;
        u_next = tmp;
    }

    double t_end = get_wall_time();
    double elapsed_sec = t_end - t_start;

    print_performance("Sequential CPU", "1 thread", total_cells, cfg.nsteps, elapsed_sec);

    save_evolution_to_csv(cfg.output_csv, cfg.nsteps, cfg.dt, avg_total_evo, avg_cube_evo, cfg.stride);

    free(alpha);
    free(alpha_x);
    free(alpha_y);
    free(alpha_z);
    free(u);
    free(u_next);
    free(avg_total_evo);
    free(avg_cube_evo);

    return 0;
}
