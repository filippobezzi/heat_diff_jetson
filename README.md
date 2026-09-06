# 3D Heterogeneous Heat Diffusion on NVIDIA Jetson Nano

[![Language](https://img.shields.io/badge/Language-C%20%2F%20CUDA%20%2F%20Python-blue.svg)](https://github.com/filippobezzi/heat_diff_jetson)
[![Target](https://img.shields.io/badge/Hardware-NVIDIA%20Jetson%20Nano-76B900.svg)](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
[![Course](https://img.shields.io/badge/Physics-Modern%20Computing-orange.svg)]()

High-performance scientific computing simulation of heterogeneous 3D transient thermal diffusion on the **NVIDIA Jetson Nano** edge computing architecture (4-core ARM Cortex-A57 CPU + 128-core Maxwell GPU).

---

## Physical Problem & Governing Equations

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

$$\Delta t \le \frac{\Delta x^2}{6 \cdot \max(\alpha)}, \qquad \Delta t = 0.80\,\frac{\Delta x^2}{6\,\alpha_{\text{Cu}}}, \quad \Delta x = \frac{L}{N}$$

Since $\Delta t \propto N^{-2}$, the number of steps to a fixed physical time scales as $N^2$ and total work as $\mathcal{O}(N^5)$.

---

## Implementation Architectures

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

## Mathematical Verification (MMS)

Code verification is performed using the **Method of Manufactured Solutions (MMS)** on a 3D trigonometric test field:

$$u_{\text{exact}}(x,y,z,t) = \sin\!\left(\frac{\pi x}{L}\right)\sin\!\left(\frac{\pi y}{L}\right)\sin\!\left(\frac{\pi z}{L}\right)\exp\!\left(-\frac{3\alpha\pi^2 t}{L^2}\right)$$

run with $L = 1$, uniform $\alpha = 0.143$, Dirichlet $u = 0$ on all faces, and $h = L/(N+1)$.

Grid convergence tests across grid resolutions $N \in \{16, 32, 64, 128\}$ confirm second-order spatial accuracy ($\mathcal{O}(h^2)$) and machine-precision equivalence between CPU and GPU backends.

---

## Benchmark Results (NVIDIA Jetson Nano)

* **Platform:** NVIDIA Jetson Nano Developer Kit (4× ARM Cortex-A57 @ 1.43 GHz, 128-core Maxwell GPU @ 921 MHz, 4 GB unified LPDDR4 @ 25.6 GB/s).

### Throughput Summary — $N = 128^3$, 100 steps, 3 repeats (mean ± σ)

Source of record: [`data/benchmarks_reduced.csv`](data/benchmarks_reduced.csv).
GFLOPS uses 24 FLOP/cell on CPU and 48 on GPU; bandwidth is the **compulsory**
traffic (20 B/cell CPU, 12 B/cell GPU), so it is a per-backend utilisation
figure and not a cross-backend comparison. Use GLUPS or wall-clock time for that.

| Backend | Threads / Block | Time (s) | Throughput | GFLOPS | $B_{\text{eff}}$ | % of 25.6 GB/s | Speedup vs Seq |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Sequential CPU** | 1 thread, untiled | 9.9223 ± 0.2741 | 0.02113 GLUPS | 0.507 | 0.423 GB/s | 1.65 % | **1.000×** |
| **OpenMP CPU** | 1 thread, tiled | 8.9536 ± 0.0596 | 0.02343 GLUPS | 0.562 | 0.468 GB/s | 1.83 % | **1.108×** |
| **OpenMP CPU** | 2 threads | 4.8226 ± 0.0868 | 0.04350 GLUPS | 1.044 | 0.870 GB/s | 3.40 % | **2.057×** |
| **OpenMP CPU** | 4 threads | 2.9353 ± 0.0168 | 0.07147 GLUPS | 1.715 | 1.429 GB/s | 5.58 % | **3.380×** |
| **OpenMP CPU** | **8 threads** | **2.8356 ± 0.0638** | **0.07397 GLUPS** | **1.776** | **1.480 GB/s** | **5.78 %** | **3.499×** |
| **OpenMP CPU** | 16 threads | 2.9119 ± 0.0118 | 0.07200 GLUPS | 1.728 | 1.440 GB/s | 5.63 % | **3.408×** |
| **CUDA GPU** | 64 th/block (32×2) | 1.0812 ± 0.0006 | 0.19393 GLUPS | 9.310 | 2.328 GB/s | 9.09 % | **9.177×** |
| **CUDA GPU** | 128 th/block (32×4) | 0.9904 ± 0.0006 | 0.21173 GLUPS | 10.164 | 2.541 GB/s | 9.93 % | **10.018×** |
| **CUDA GPU** | **256 th/block (32×8)** | **0.9501 ± 0.0004** | **0.22070 GLUPS** | **10.595** | **2.649 GB/s** | **10.35 %** | **10.443×** |
| **CUDA GPU** | 512 th/block (32×16) | 0.9629 ± 0.0004 | 0.21777 GLUPS | 10.454 | 2.614 GB/s | 10.21 % | **10.304×** |
| **CUDA GPU** | 1024 th/block (32×32) | 1.0506 ± 0.0003 | 0.19963 GLUPS | 9.582 | 2.395 GB/s | 9.36 % | **9.444×** |

Optimal CUDA block size is **256** (tile 32×8); the spread across the block sweep is 13.8 %.

### Large-Grid Scaling — $N = 384^3$ (56.6 M cells, 27× the cells)

Source of record: [`data/benchmarks_large.csv`](data/benchmarks_large.csv).

| Backend | Threads / Block | Time (s) | Throughput | Speedup vs Seq |
| :--- | :--- | ---: | ---: | ---: |
| **Sequential CPU** | 1 thread, untiled | 290.022 ± 0.405 | 0.01950 GLUPS | **1.000×** |
| **OpenMP CPU** | 8 threads | 77.751 ± 0.108 | 0.07280 GLUPS | **3.730×** |
| **CUDA GPU** | 256 th/block | 25.878 ± 0.004 | 0.21880 GLUPS | **11.207×** |

CUDA throughput is flat to **0.86 %** (0.22070 → 0.21880 GLUPS) across the 27×
increase in cell count: the 2.5D scheme's reuse is block-local and therefore
size-independent. The *speedup ratios* rise (OpenMP 3.499× → 3.730×, CUDA
10.443× → 11.207×) because the denominator moves: every backend loses
throughput at 56.6 M cells, but the untiled sequential loop loses 7.73 %
against CUDA's 0.86 %. Quote the GLUPS column, not the ratio, for claims about
the parallel codes themselves.

### Profiled Kernel Traffic (`nvprof`, block 256)

| Metric | Value |
| :--- | ---: |
| `gld_throughput` / `gst_throughput` | 2.838 / 0.903 GB/s |
| Measured DRAM traffic | 16.57 B/cell (4.14 accesses/cell) |
| Measured CGMA / arithmetic intensity | 11.59 FLOP/access / 2.897 FLOP/B |
| `gld_efficiency` | 79.61 % |
| `achieved_occupancy` (`--maxrregcount=32`) | 0.9852 |
| `sysmem_read/write_throughput` | 0 (unified memory, no PCIe traffic) |

Every derivation behind these tables — formulas, explicit arithmetic, roofline
placement, tile working sets, the $\mathcal{O}(N^5)$ cost law — is written out in
[`COMPUTATIONS.md`](COMPUTATIONS.md).

---

## Repository Structure

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
│   └── plot_results.py           # Publication-quality plotting suite (300 DPI)
│
├── data/                         # CSV datasets (source of record for every quoted number)
│   ├── avg_u_evo_3d.csv          # 500 s physical temperature evolution (N=64, CUDA, 170,400 steps)
│   ├── benchmarks_reduced.csv    # Benchmark timings for N=128^3 (5 thread counts, 5 block sizes)
│   ├── benchmarks_large.csv      # Benchmark timings for N=384^3
│   ├── mms_convergence_3d.csv    # L2 error and observed order vs grid spacing
│   ├── mms_u_evo_3d.csv          # MMS field average vs analytic solution
│   └── mms_{seq,omp,cuda}_n{16,32,64,128}.csv   # Per-backend, per-resolution MMS traces
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
├── COMPUTATIONS.md               # Reference sheet: every metric, formula and explicit calculation
└── presentation_FINAL.md         # Final presentation slides
```

---

## Build & Execution Instructions

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

## License
Academic project for the **Modern Computing for Physics (MCP)** course at the University of Padua (UniPD). Distributed under the MIT License.
