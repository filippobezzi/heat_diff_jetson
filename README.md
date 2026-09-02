# 3D Heterogeneous Heat Diffusion on NVIDIA Jetson Nano

[![Language](https://img.shields.io/badge/Language-C%20%2F%20CUDA%20%2F%20Python-blue.svg)](https://github.com/filippobezzi/heat_diff_jetson)
[![Target](https://img.shields.io/badge/Hardware-NVIDIA%20Jetson%20Nano-76B900.svg)](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
[![Course](https://img.shields.io/badge/Physics-Modern%20Computing-orange.svg)]()

High-performance scientific computing simulation of heterogeneous 3D transient thermal diffusion on the **NVIDIA Jetson Nano** edge computing architecture (4-core ARM Cortex-A57 CPU + 128-core Maxwell GPU).

---

## 📌 Physical Problem & Governing Equations

The system models unsteady, three-dimensional heat conduction through a heterogeneous medium governed by the variable-diffusivity heat equation:

$$\frac{\partial u(\mathbf{x}, t)}{\partial t} = \nabla \cdot \Big(\alpha(\mathbf{x}) \nabla u(\mathbf{x}, t)\Big) + S(\mathbf{x}, t)$$

### Thermal Configuration
* **Domain:** Insulated cubic container of dimensions $L_x = L_y = L_z = 100\text{ mm}$ with adiabatic (Neumann $\nabla u \cdot \mathbf{n} = 0$) boundary conditions.
* **Medium:** Water bath at initial temperature $T_{\text{water}} = 20^\circ\text{C}$ ($\alpha_{\text{water}} = 1.43 \times 10^{-7}\text{ m}^2/\text{s}$).
* **Heat Source:** Centered solid copper cube of dimensions $20 \times 20 \times 20\text{ mm}$ at initial temperature $T_{\text{cube}} = 80^\circ\text{C}$ ($\alpha_{\text{Cu}} = 1.11 \times 10^{-4}\text{ m}^2/\text{s}$).
* **Diffusivity Ratio:** $\frac{\alpha_{\text{Cu}}}{\alpha_{\text{water}}} \approx 776\times$, creating steep thermal gradients across interfaces.

### Conservative Interface Discretization
To strictly enforce heat flux continuity across material discontinuities, inter-cell conductivities are computed using **harmonic means**:

$$\alpha_{i+1/2, j, k} = \frac{2 \alpha_{i, j, k} \alpha_{i+1, j, k}}{\alpha_{i, j, k} + \alpha_{i+1, j, k}}$$

The explicit FTCS update is bounded by the 3D von Neumann numerical stability limit:

$$\Delta t \le \frac{\Delta x^2}{6 \cdot \max(\alpha)}$$

---

## 🚀 Implementation Architectures

| Implementation | Backend | Memory Access Strategy | Arithmetic Cost | Compulsory DRAM Traffic |
| :--- | :--- | :--- | :--- | :--- |
| **Sequential** | C (`gcc -O3`) | Precomputed face arrays ($\alpha_x, \alpha_y, \alpha_z$) | 24 FLOP/cell | 20 Bytes/cell |
| **Multi-Threaded** | OpenMP (`-fopenmp`) | 3D pencil cache tiling ($64 \times 16 \times 16$) | 24 FLOP/cell | 20 Bytes/cell |
| **GPU Accelerated**| CUDA (`nvcc -O3`) | 2.5D register sliding window + 2D shared memory halo | 48 FLOP/cell (on-the-fly) | 12 Bytes/cell |

### Key Optimizations:
1. **Sequential Baseline (C):** Precomputes directional interface harmonic averages during initialization, converting 6 on-the-fly floating-point divisions per cell into 3 memory reads.
2. **OpenMP CPU Solver:** Employs 3D cache pencil blocking (`OMP_TILE_X=64, OMP_TILE_Y=16, OMP_TILE_Z=16`) to retain working sets entirely within the Cortex-A57's 2 MB shared L2 cache, eliminating memory bus streaming stalls.
3. **CUDA GPU Solver:** Implements a 2.5D sliding-window stencil kernel:
   - Maintains $z$-axis neighbors ($u_{k-1}, u_k, u_{k+1}$) in on-chip GPU hardware registers.
   - Loads $x$- and $y$-halos into 2D shared memory (`__shared__`).
   - Computes harmonic interface averages on the fly to trade cheap arithmetic for precious DRAM bandwidth (increasing Arithmetic Intensity from $1.20 \to 4.00\text{ FLOP/Byte}$).

---

## 🔬 Mathematical Verification (MMS)

Code verification is performed using the **Method of Manufactured Solutions (MMS)** on a 3D trigonometric test field:

$$u_{\text{exact}}(x,y,z,t) = e^{-3\pi^2 t} \sin(\pi x)\sin(\pi y)\sin(\pi z)$$

Grid convergence tests across grid resolutions $N \in \{16, 32, 64, 128\}$ confirm second-order spatial accuracy ($\mathcal{O}(h^2)$) and machine-precision equivalence between CPU and GPU backends.

---

## 📊 Benchmark Results (NVIDIA Jetson Nano)

* **Platform:** NVIDIA Jetson Nano Developer Kit (4× ARM Cortex-A57 @ 1.43 GHz, 128-core Maxwell GPU @ 921 MHz, 4 GB unified LPDDR4 @ 25.6 GB/s).

### Grid Scaling & Throughput Summary

| Backend | Threads / Block | Time ($N=128^3$) | Throughput | GFLOPS | Compulsory Bandwidth | Speedup vs Seq |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sequential CPU** | 1 thread | 0.428 s | 0.489 GLUPS | 11.75 GFLOPS | 9.79 GB/s | **1.00×** |
| **OpenMP CPU** | 4 threads | 0.123 s | 1.705 GLUPS | 40.92 GFLOPS | 34.10 GB/s (cached) | **3.48×** |
| **CUDA GPU** | 256 th/block | 0.024 s | 8.738 GLUPS | 419.42 GFLOPS | 104.85 GB/s (cached) | **17.83×** |

---

## 📂 Repository Structure

```
heat_diff_jetson/
├── src/                          # C and CUDA source codes
│   ├── Makefile                  # Build script for host compiler
│   ├── Makefile_jetson           # Build script optimized for Jetson Nano (aarch64 / SM_53)
│   ├── heat_3d.h                 # Core header, struct configs, and performance print routines
│   ├── heat_cube_seq_3d.c        # Sequential C implementation
│   ├── heat_cube_omp_3d.c        # OpenMP CPU implementation (tiled & untiled)
│   ├── heat_cube_cuda_3d.cu      # 2.5D tiled CUDA GPU implementation
│   ├── heat_mms_seq_3d.c         # MMS validation (Sequential)
│   ├── heat_mms_omp_3d.c         # MMS validation (OpenMP)
│   ├── heat_mms_cuda_3d.cu       # MMS validation (CUDA)
│   ├── benchmark_sweep.py        # Automated benchmark harness
│   ├── test_mms.py               # Spatial grid convergence runner
│   ├── plot_results.py           # Publication-quality plotting suite (300 DPI)
│   └── utils.md                  # Hardware details and profiling instructions
│
├── data/                         # CSV datasets
│   ├── avg_u_evo_3d.csv          # 500s physical temperature evolution dataset
│   ├── benchmarks_reduced.csv    # Benchmark timings for N=128^3
│   ├── benchmarks_large.csv      # Benchmark timings for N=384^3
│   └── mms_convergence_3d.csv    # L2 error convergence data
│
├── fig/                          # Publication figures
│   ├── temperature_evolution.png # Temperature evolution over 500 seconds
│   ├── parameter_sweep_times.png # Parameter sweep times (OpenMP threads & CUDA block size)
│   ├── speedup_ideal.png         # Speedup curves vs theoretical linear / BW limits
│   ├── speedup_grid_sizes.png    # Scaling across small vs large grids
│   ├── mms_test.png              # Spatial convergence plot (O(h^2) slope)
│   ├── effective_bandwidth.png   # Measured memory bandwidth
│   └── roofline_model.png        # Roofline operational points
│
└── presentation_FINAL.md         # Final presentation slides
```

---

## 🛠️ Build & Execution Instructions

### Prerequisites
* **C Compilers:** `gcc` with OpenMP support (`-fopenmp`)
* **CUDA Toolkit:** `nvcc` with Compute Capability 5.3+ (`-arch=sm_53`)
* **Python Environment:** Python 3.8+ with `numpy`, `pandas`, `matplotlib`

```bash
# Clone the repository
git clone https://github.com/filippobezzi/heat_diff_jetson.git
cd heat_diff_jetson/src
```

### 1. One-Shot Reproduction
To compile all targets, execute the MMS convergence suite, run the full benchmark parameter sweeps, and generate all figures:

```bash
make results
```

### 2. Manual Step-by-Step Execution

#### Build Targets
```bash
make all        # Builds production binaries (heat_cube_seq_3d, heat_cube_omp_3d, heat_cube_cuda_3d, MMS targets)
make cpu        # Builds only CPU targets (if running on a machine without NVCC)
```

#### Run Solvers Individually
```bash
# Sequential C Baseline (Grid 64^3, 500 seconds physical time)
./heat_cube_seq_3d --n 64 --time 500 --stride 200 --output ../data/avg_u_evo_3d.csv

# OpenMP CPU Solver (4 threads)
./heat_cube_omp_3d --n 64 --time 500 --stride 200 --threads 4 --output ../data/avg_u_evo_3d.csv

# CUDA GPU Solver (256 threads per block)
./heat_cube_cuda_3d --n 64 --time 500 --stride 200 --block_size 256 --output ../data/avg_u_evo_3d.csv
```

#### MMS Convergence Verification
```bash
python3 test_mms.py
```

#### Benchmark Parameter Sweeps
```bash
python3 benchmark_sweep.py
```

#### Generate Plots
```bash
python3 plot_results.py --plot all
```

---

## 📄 License
Academic project for the **Modern Computing for Physics (MCP)** course at the University of Padua (UniPD). Distributed under the MIT License.
