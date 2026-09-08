#ifndef HEAT_3D_H
#define HEAT_3D_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Physical Constants
#define ALPHA_W_DEF 0.143f   // Water thermal diffusivity [mm^2/s]
#define ALPHA_C_DEF 111.0f   // Copper thermal diffusivity [mm^2/s]
#define T_W_0_DEF   20.0f    // Initial water temperature [°C]
#define T_C_0_DEF   80.0f    // Initial copper cube temperature [°C]

// Domain Default Dimension (mm)
#define L_DEF 100.0f

// MMS Defaults
#define MMS_L_DEF     1.0f
#define MMS_ALPHA_DEF 0.143f
#define PI_L(L)       ((float)M_PI / (float)(L))

// CPU L1/L2 Cache Blocking Defaults
#define OMP_TILE_X 64
#define OMP_TILE_Y 16
#define OMP_TILE_Z 16

// 3D 1D-Flattened Indexing Macro: Row-major in X
// Layout: index = (k * ny + j) * nx + i
/*
       _______________ (nx,ny,0)
      /ny            /|
     / |            / |
  j /  |           /  |
   /0   <-i->  nx /   |
  /______________/    |
  |0   |         |    |
  |    |_________|____|
  |    /(0,ny,nk)|    |    /(nx,ny,nk)
k |   /          |   /
  |  /           |  /
  |nk            | /
  |______________|/
                 (nx,0,nk)
*/

#define IDX3(i, j, k, nx, ny) (((size_t)(k) * (size_t)(ny) + (size_t)(j)) * (size_t)(nx) + (size_t)(i))

// Harmonic Mean for Conservative Interface Diffusivity
#ifdef __CUDACC__ // make callable for both C CPU and CUDA GPU
__host__ __device__
#endif
static inline float harmonic_mean(float a, float b) {
    if (a + b <= 0.0f) return 0.0f;
    return (2.0f * a * b) / (a + b);
}

#ifdef __CUDACC__
static inline void get_3d_block_dims(int total_threads, dim3 *blockDim) {
    if (total_threads <= 64) {
        *blockDim = dim3(32, 2, 1);
    } else if (total_threads <= 128) {
        *blockDim = dim3(32, 4, 1);
    } else if (total_threads <= 256) {
        *blockDim = dim3(32, 8, 1);
    } else if (total_threads <= 512)
        *blockDim = dim3(32, 16, 1); // 512 threads
    } else {
        *blockDim = dim3(32, 32, 1); // 1024
    }
}

static inline void get_2d_tile_dims(int total_threads, dim3 *tileDim) {
    int tile_x = 32, tile_y = 16;
    if (total_threads <= 64)         { tile_x = 32; tile_y = 2;  }
    else if (total_threads <= 128)   { tile_x = 32; tile_y = 4;  }
    else if (total_threads <= 256)   { tile_x = 32; tile_y = 8;  }
    else if (total_threads <= 512)   { tile_x = 32; tile_y = 16; } // 512 threads
    else                             { tile_x = 32; tile_y = 32; } // 1024 threads
    *tileDim = dim3(tile_x, tile_y);
}
#endif

// High-resolution timer (nanosecond precision)
static inline double get_wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Simulation Configuration Struct
typedef struct {
    int nx, ny, nz;
    float lx, ly, lz;
    float dx, dy, dz;
    float alpha_water;
    float alpha_cube;
    float t_water_0;
    float t_cube_0;
    float cube_x_min, cube_x_max;
    float cube_y_min, cube_y_max;
    float cube_z_min, cube_z_max;
    float time_scale;
    float dt;
    int nsteps;
    int stride;
    int num_threads; // OpenMP
    int block_size;  // CUDA
    char output_csv[256];
} SimulationConfig;

static inline void init_default_config(SimulationConfig *cfg, int n_cells) {
    cfg->nx = n_cells;
    cfg->ny = n_cells;
    cfg->nz = n_cells;

    cfg->lx = L_DEF;
    cfg->ly = L_DEF;
    cfg->lz = L_DEF;

    cfg->dx = cfg->lx / (float)cfg->nx;
    cfg->dy = cfg->ly / (float)cfg->ny;
    cfg->dz = cfg->lz / (float)cfg->nz;

    cfg->alpha_water   = ALPHA_W_DEF;
    cfg->alpha_cube    = ALPHA_C_DEF;
    cfg->t_water_0     = T_W_0_DEF;
    cfg->t_cube_0      = T_C_0_DEF;

    // Centered 20x20x20 mm Copper Cube
    cfg->cube_x_min = 40.0f; cfg->cube_x_max = 60.0f;
    cfg->cube_y_min = 40.0f; cfg->cube_y_max = 60.0f;
    cfg->cube_z_min = 40.0f; cfg->cube_z_max = 60.0f;

    // 3D FTCS Stability Limit: dt <= dx^2 / (6 * alpha_max)
    float alpha_max = (cfg->alpha_cube > cfg->alpha_water) ? cfg->alpha_cube : cfg->alpha_water;
    cfg->dt = 0.80f * (cfg->dx * cfg->dx) / (6.0f * alpha_max); // Safety factor 0.80 (80% of 3D FTCS limit)

    // Characteristic Time Scale: tau ~ L_w^2 / (pi^2 * alpha_w)
    float water_span = (cfg->lx - (cfg->cube_x_max - cfg->cube_x_min)) / 2.0f;
    cfg->time_scale = (water_span * water_span) / ((float)(M_PI * M_PI) * cfg->alpha_water);
    cfg->nsteps = (int)ceilf(cfg->time_scale / cfg->dt);

    cfg->stride = 10;
    cfg->num_threads = 4;
    cfg->block_size = 256;
    snprintf(cfg->output_csv, sizeof(cfg->output_csv), "../data/avg_u_evo_3d.csv");
}

static inline void parse_arguments(int argc, char *argv[], SimulationConfig *cfg) {
    int user_steps = 0;
    float user_time = 0.0f;

    /* Pass 1: --n only.
     * init_default_config() re-derives the whole configuration (dx, dt, nsteps)
     * AND resets every other field to its default value.
     */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            int user_n = atoi(argv[++i]);
            if (user_n > 0 && user_n != cfg->nx) {
                init_default_config(cfg, user_n);
            }
        }
    }

    /* Pass 2 */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            i++; /* already consumed in pass 1 */
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            user_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--time") == 0 && i + 1 < argc) {
            user_time = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--stride") == 0 && i + 1 < argc) {
            cfg->stride = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg->num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--block_size") == 0 && i + 1 < argc) {
            cfg->block_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            strncpy(cfg->output_csv, argv[++i], sizeof(cfg->output_csv) - 1);
            cfg->output_csv[sizeof(cfg->output_csv) - 1] = '\0';
        }
    }

    if (user_time > 0.0f) {
        cfg->time_scale = user_time;
        cfg->nsteps = (int)ceilf(cfg->time_scale / cfg->dt);
    } else if (user_steps > 0) {
        cfg->nsteps = user_steps;
        cfg->time_scale = cfg->nsteps * cfg->dt;
    }

    if (cfg->stride < 1) cfg->stride = 1;
}

/* ---------------------------------------------------------------------------
 * Performance reporting - shared by all three backends so that the numbers are
 * produced by identical arithmetic and are therefore directly comparable.
 *
 * FLOPs per lattice-point update: 6 fluxes x (1 sub + 2 mul) = 18, plus 5
 * add/sub to combine them and 1 final add = 24.
 * On-the-fly harmonic mean computation on GPU raises that number to 48.
 *
 * Effective Bandwidth: the true DRAM traffic  depends on cache and shared-memory
 * reuse. To be calculate after compilation with a profiler
 * (nvprof --metrics gld_throughput, gst_throughput).
 * Given our implementation, the two theoretical bounds are:
 * CPU:
 *   lower  =  1 u-read + 3 alpha-reads + 1 write  = 5 floats = 20 B  (perfect reuse)
 *   upper  =  7 u-reads + 6 alpha-reads + 1 write = 14 floats = 56 B (no reuse)
 * GPU:
 *   lower  =  1 u-read + 1 alpha-reads + 1 write  = 3 floats = 12 B  (perfect reuse)
 *   upper  =  7 u-reads + 7 alpha-reads + 1 write = 15 floats = 60 B (no reuse)
 * ------------------------------------------------------------------------- */

#ifdef __CUDACC__
// CUDA GPU: on-the-fly harmonic means (48 FLOP/cell, 12 B/cell compulsory)
#define FLOPS_PER_CELL       48.0
#define BYTES_PER_CELL_MIN   12.0
#else
// CPU (Sequential / OpenMP): precomputed alpha_x/y/z (24 FLOP/cell, 20 B/cell compulsory)
#define FLOPS_PER_CELL       24.0
#define BYTES_PER_CELL_MIN   20.0
#endif

static inline void print_performance(const char *backend, const char *config,
                                     size_t total_cells, int nsteps,
                                     double elapsed_sec) {
    double glups = ((double)total_cells * (double)nsteps) / (elapsed_sec * 1e9);
    printf("========================================================\n");
    printf("Simulation Completed Successfully!\n");
    printf("  Backend                   : %s\n", backend);
    printf("  Configuration             : %s\n", config);
    printf("  Solve Wall-Clock Time     : %.4f seconds\n", elapsed_sec);
    printf("  Throughput (GLUPS)        : %.4f GLUPS\n", glups);
    printf("  Compute Performance       : %.4f GFLOPS (%.0f FLOP/cell)\n",
           glups * FLOPS_PER_CELL, FLOPS_PER_CELL);
    printf("  Effective Memory Bandwidth: %.4f GB/s\n",
           glups * BYTES_PER_CELL_MIN);
    printf("========================================================\n");
}

#endif // HEAT_3D_H
