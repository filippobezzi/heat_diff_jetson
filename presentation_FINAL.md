%title: Computational Limits of the Jetson Nano - 3D Heat Diffusion
%author: Filippo Bezzi
%date: 2026-09-08

-> # Computational Limits of the NVIDIA Jetson Nano <-
-> ### 3D Heterogeneous Heat Diffusion from Ground Zero ### <-

<br>

-> Modern Computing for Physics (MCP) <-
-> Final Project <-

<br>

A complete workflow, from a plain sequential CPU implementation to a
GPU-targeted one, for a problem whose cost I could estimate in advance.

-> The aim was not to show that the GPU wins, but to find out <-
-> which resource actually sets the limit, and to check it. <-

--------------------------------------------------------------------------------

# Outline

* **1. The task** - a copper cube quenched in water, and why the
  material contrast makes it interesting.
* **2. Numerics** - conservative discretisation, stability, cost scaling.
* **3. The platform** - Tegra X1 ceilings derived before coding.
* **4. Three implementations** - sequential C, OpenMP, CUDA.
* **5. Verification** - convergence and a conservation law.
* **6. Benchmarks** - figures of merit vs threads, block size, grid size.
* **7. Limiting factors** - what I predicted, and what I measured.

<br>

```
   Sequential C  ->  OpenMP directives  ->  CUDA (coalesced -> tiled)
```

Set-up and methods first; every measured number appears in section 6.
Benchmark point: N = 128^3, 100 steps, 3 repeats, mean +/- sigma.

--------------------------------------------------------------------------------

# The Physical Task

```
        Domain: 100 x 100 x 100 mm  (zero-flux / insulated walls)
       +---------------------------------------------------------+
       |  WATER   alpha_w = 0.143 mm^2/s     T_0 = 20 C          |
       |                                                         |
       |                 +-------------+                         |
       |                 |   COPPER    |   20 x 20 x 20 mm       |
       |                 |  alpha_Cu = |   T_0 = 80 C            |
       |                 |  111 mm^2/s |                         |
       |                 +-------------+                         |
       |                                                         |
       |          alpha ratio = 111 / 0.143 = 776                |
       +---------------------------------------------------------+
```

* The cube is **0.8 %** of the volume, with a diffusivity **776x**
  larger than the water surrounding it.

* A single-material cube would have reduced to a Laplacian benchmark.
  The material jump is what forces a conservative discretisation and,
  as we will see, a very small timestep.

Relevance: the material contrast, not the grid size, is what sets the
cost of this simulation. Everything that follows descends from that 776.

--------------------------------------------------------------------------------

# Why the Divergence Form Is Not Optional

* With alpha varying in space, the equation to discretise is

```
      du/dt = div( alpha(x) grad(u) )      NOT      alpha * lap(u)
```

* Expanding the divergence produces an extra `grad(alpha) . grad(u)`
  term. At the copper/water face `grad(alpha)` is a 776x jump, so
  dropping it would produce unphysical results (leak of energy).

* I discretise **fluxes on cell faces**, with a harmonic mean:

```
      alpha_{i+1/2} = 2 * alpha_i * alpha_{i+1} / (alpha_i + alpha_{i+1})
```

  * Harmonic, not arithmetic: two conductors **in series**, so the face
    conductance is dominated by the poorer of the two materials.
  * An arithmetic mean would let the copper dominate the interface and
    over-conduct heat into the water.

* Where those six face values come from is an **implementation** choice,
  and the two backend families answer it differently in line with their
  different needs: the CPU precomputes them once into `alpha_x`, `alpha_y`, `alpha_z`;
  the GPU rebuilds them inside the kernel. I return to this later.

Relevance: a physics requirement that costs either three extra arrays
and six extra memory accesses per cell, or the arithmetic to rebuild
them. That trade-off is the origin of the traffic problem on the next
slide, and of the main design decision in the CUDA version.

--------------------------------------------------------------------------------

# The 3D 7-Point Stencil: Counting Operations and Accesses

```
                                 u[i,j,k-1]   (Up: k-1)
                                     o
                                     |  
                                     |     o u[i,j+1,k]   (Back: j+1)
                                     |    /  
                        alpha_z[i,j,k-1] /
                                     |  alpha_y[i,j,k]
                                     | /
                                     |/
u[i-1,j,k] o----alpha_x[i-1,j,j]---- O ----alpha_x[i,j,k]----o u[i+1,j,k]   
(Left: i-1)                         /|                         (Right: i+1)
                                   / |
                      alpha_y[i,j,k] |
                                 /   |
                                /  alpha_z[i,j,k]
                               o     |               
         (Front: j-1) u[i,j-1,k]     o
                                     u[i,j,k+1]   (Down: k+1)
                                                 
```


```
   F_x^R = alpha_x[i,j,k]   * ( u[i+1,j,k] - u[i,j,k] ) * invdx2 <- 1 sub, 2 mul
   F_x^L = alpha_x[i-1,j,k] * ( u[i,j,k] - u[i-1,j,k] )     ... x 6 faces

   COMPUTE : 6 x 3 + 5 combine + 1 update      =  24 FP operations / cell
   ACCESSES: 7(1) u-reads + 6(3) alpha-reads + 1 write = 14(5) accesses / cell
                                                 = 56(20) Bytes / cell (FP32)

   EFFECTIVE MEMORY BANDWIDTH := minimal access (cache reuse) / cell
                               = 20 * GLUPS * Bytes / cell
   Giga-Lattice-Updates-per-Second = #Updates x 10^9 / elapsed time (s)
```

* **CGMA = 24 / 5 = 4.8 FP operations per (compulsory) global memory accesses.**
    Theoretical CGMA = 36.84 FLOP / access

Relevance: This given an insight into implementation - low CGMA means the task is
  memory-bound and every optimisation should target memory accesses, not operations.


--------------------------------------------------------------------------------

# Stability: the Copper Sets the Timestep for Every Cell

* Von Neumann analysis (insert a Fourier mode `u ~ g^n exp(i k.x)` and
  require `|g| <= 1`, so no mode can grow unstable) gives,
  for explicit Forard-Time Central-Space discretization (FTCS) in 3D:

```
      dt <= h^2 / ( 6 * alpha_max ) = h^2 / ( 6 * alpha_copper )
```

  I run at 80 % of this bound: `dt = 0.80 h^2 / (6 alpha_Cu)`, keeping a
  20 % margin against the FP32 round-off in the coefficients.

* `alpha_max` is a **global** maximum, so the 99.2 % of the grid that is
  water is integrated 776x more finely than its own physics requires.

* The two relaxation times, `tau = L^2 / (pi^2 alpha)`:

```
    Copper cube  (L = 20 mm) :  tau_Cu = 0.37 s
    Water shell  (L = 40 mm) :  tau_w  = 1134 s     (~19 minutes)

    separation = 3105, i.e. about 3.5 decades
```

Relevance: this is a **stiff** problem, meaning there is a massive disparity
 between the fast and slow physical time scales. The stiffness is the dominant
 cost driver here.

--------------------------------------------------------------------------------

# The O(N^5) Cost Wall

* Refining the grid costs twice over: since `dt ~ h^2`, halving h gives
  8x the cells (space) **and** 4x the steps (time).

```
      work  ~  N^3 (cells)  x  N^2 (steps)  =  O(N^5)
```

* Inverted, this answers a more useful question: for a given wall-clock
  budget, how much *physical* time can I simulate? With `LUPS` the
  measured lattice-update rate and `r = 0.80` the stability factor,

```
   +-----------------------------------------------------------------+
   |                                                                 |
   |   T_phys,max  =  t_wall  x  ( r L^2 LUPS / 6 alpha )  x  N^-5   |
   |                                                                 |
   +-----------------------------------------------------------------+
```

* Two consequences I can state before measuring anything:

  * the only free parameter on the right is `LUPS`, so **all three
    backends obey the same N^-5 law** and differ only by a prefactor;
  * a factor 3 in resolution costs a factor 3^5 = 243 in reachable
    physical time. Refinement cannot be considered a matter of waiting
    longer.

* I evaluate this with the measured throughput in the results section,
  where it also fixes the grid and duration of the evolution run.

Relevance: this is a limiting factor. It let me choose the benchmark
  configurations deliberately rather than by trial.

--------------------------------------------------------------------------------

# The Platform: One Chip, One Memory Pool

```
       +---------------------------------------------------------+
       |        NVIDIA JETSON NANO SoC  (Tegra X1 / T210)        |
       +----------------------------+----------------------------+
                                    |
       +----------------------------+----------------------------+
       |      4 GB UNIFIED LPDDR4 DRAM  (shared CPU & GPU)       |
       |        64-bit bus @ 1600 MHz  ->  peak 25.6 GB/s        |
       +----------------------------+----------------------------+
                                    |
                    +---------------+---------------+
                    |                               |
         +--------------------+          +--------------------+
         | 4x Cortex-A57 CPU  |          |   1 SMP, Maxwell   |
         |  4 cores, no SIMD  |          |   128 CUDA cores   |
         |   2 MB shared L2   |          |  compute cap. 5.3  |
         |  32 KB L1D / core  |          |  256 KB L2 cache   |
         +--------------------+          +--------------------+
```

Relevance: each hardware block directly motivates a specific implementation choice:
  * 4 GB => hard upper ceiling on the maximum grid dimensions N^3;
  * Small CPU/GPU caches => explicit spatial cache blocking and 2.5D tiling;
  * 128 cores on 1 SM => latency must be hidden via warp occupancy and coalescing;
  * Unified 25.6 GB/s bus => establishes the ultimate performance bottleneck for both engines.


--------------------------------------------------------------------------------

# Ceilings Derived Before Writing Code

* **Peak FP32 (GPU):**

```
    128 CUDA cores x 2 FLOP/cycle (FMA) x 0.921 GHz  =  236 GFLOP/s
    (deviceQuery reports a 900 MHz base clock -> 230 GFLOP/s;
     the distinction does not matter for the argument below)
```

* **Peak bandwidth:** `8 B x 1600 MHz x 2 (DDR) = 25.6 GB/s`

* **The ceiling that actually binds**, from the CGMA of 1.71:

```
    max FLOPS = ( 25.6 GB/s / 4 B ) x CGMA = 6.4 x 1.71 = 11.0 GFLOP/s
```

* So the memory system caps the naive formulation at **11.0 of**
  **236 GFLOP/s - 4.7 % of the compute peak**.

* **Machine balance:** `236 / 25.6 = 9.22 FLOP/Byte`. Below that ratio
  no kernel can be compute-limited. My naive stencil sits at
  `24/56 = 0.43`, a factor 21 below it.

* **FP64:** on Maxwell sm_53 the FP64 issue rate is 1/32 of FP32.
  Everything is FP32; FP64 would have cost a factor 32 for no benefit.

Relevance: To raise the ceiling I must raise CGMA, and there are 
 two ways to do it - reuse data instead of re-reading it,
 or **do more arithmetic per byte fetched**. I ended up
 using both.

--------------------------------------------------------------------------------

# One SMP: Latency Hiding Is the Whole Game

```
       +----------------------------------------------------------+
       |    MAXWELL STREAMING MULTIPROCESSOR  (the Nano has 1)    |
       +----------------------------------------------------------+
       |  * 128 CUDA cores (SPs), warp size 32                    |
       |  * 4 warp schedulers  ->  zero-overhead thread switching |
       |  * Max resident threads : 2048   (64 warps)              |
       |  * Register file        : 64K x 32-bit  (256 KB)         |
       |  * Shared memory        : 64 KB / SMP, 48 KB per block   |
       +----------------------------------------------------------+
```

* With **one** SMP there is nowhere for extra blocks to go:
  a large grid simply queues.

* A global memory access costs ~400-800 clock cycles; an FP operation
  costs ~1. The scheduler hides that gap by switching to another
  eligible warp at zero overhead - but only if eligible warps exist.

```
    The rule: overcommit the SMP with warps.
```

--------------------------------------------------------------------------------

# The Memory Hierarchy Suggested the Algorithm

```
   Latency          Space              Scope        Size
  -----------      -------            --------     ----------------
  ~0 cycles        Registers          thread       256 KB / SMP
  ~10-100 cycles   Shared memory      block        48 KB / block
  ~20 cycles       L2 cache           grid         256 KB
  400-800 cycles   Global (LPDDR4)    grid         4 GB shared pool
```

* A 7-point stencil naively reads every `u` value **seven times**: once as the centre
  cell, twice as an X/Y/Z neighbour. 

* 2.5D Tiling maps these 7 accesses directly onto the memory hierarchy:
```
    pay the 400-800 cycle global access ONCE per value,
    then serve the other two uses from shared memory and registers:
```
   -> XY accesses from SHARED MEMORY (~20 cycles)
   -> Z accesses from REGISTERS (0 cycles)

* The XY neighbours are needed by **other threads** in the block, so
  they belong in shared memory. The Z neighbours are needed by **the**
  **same thread at a later k**, so they belong in registers.

Relevance: 2.5D tiling is the first of the two ways to raise
 CGMA - reuse.

--------------------------------------------------------------------------------

# Memory Budget: Choosing the Grid Sizes

* The two backend families (CPU/GPU) keep a different number of FP32 arrays alive,
  because they treat the interface diffusivity differently:

```
   CPU  (seq, OpenMP)   u, u_next, alpha, alpha_x, alpha_y, alpha_z
                        6 arrays  ->  24 * N^3 Bytes

   CUDA                 u, u_next, alpha
                        3 arrays  ->  12 * N^3 Bytes
```

```
   Grid            Cells        CPU (6 arr)   CUDA (3 arr)   Verdict
  -------------   -----------   -----------   ------------   -----------
   N =  64^3       0.26 M          6.3 MB        3.1 MB      evolution run
   N = 128^3       2.10 M         50.3 MB       25.2 MB      sweep point
   N = 384^3      56.62 M          1.36 GB       0.68 GB     benchmark test
   N = 512^3     134.22 M          3.22 GB       1.61 GB     CPU: OOM
```

* The 4 GB is the *whole system* pool - kernel, display server and page
  cache all take a share.

Relevance: N = 384 is the largest grid **all three** backends can hold,
so it is the common stress point. The GPU alone could reach 512^3.

--------------------------------------------------------------------------------

# Raising CGMA the Second Way: Arithmetic for Bandwidth


* The CPU version precomputes the interface diffusivity arrays once and
  reads them per cell. On the GPU, I do the opposite: store only the raw
  `alpha` field and rebuild the six harmonic means **on-the-fly inside the**
  **kernel**, from values already in shared memory and registers.

```
                          reads/cell    FLOP/cell    CGMA
   -------------------   ------------  -----------  -------
   precomputed faces      7 u + 6 a        24        1.71
   on-the-fly means       7 u + 7 a        48        3.20
```

* One extra `alpha` read, but the arithmetic doubles: each harmonic mean
  is `2ab/(a+b)`, i.e. *2 multiplies*, *1 add* and **1 division**, six times
  per cell.

* On this machine that is a good trade in one direction only:

  * The GPU has a machine balance of 9.22 FLOP/B and (naively) sits at 0.43.
    It has arithmetic units idle while waiting on memory, so paying FLOPs
    to avoid bytes moves it *towards* the ridge point.

  * Cortex-A57 CPU has limited scalar ALUs and expensive division. Spending
    24 extra FLOPs per cell would degrade throughput, so the CPU keeps
    the precomputed arrays.

Relevance: The three backends now differ algorithmically.

--------------------------------------------------------------------------------

# One Data Layout for Three Backends

```
   Row-major linearisation:  IDX3(i,j,k) = (k * Ny + j) * Nx + i

           _______________ (nx,ny,0)
          /ny            /|
         / |            / |
      j /  |           /  |        i (X): contiguous in memory
       /0   <--i--> nx/   |       j (Y): strides by Nx
      /______________/    |        k (Z): strides by Nx*Ny
      |0   |         |    |
      |    |_________|____|
      |   /          |   /
    k |  /           |  /
      | /            | /
      |______________|/
```

* A 64 B cache line holds 16 consecutive floats along `i`.
* A warp is 32 consecutive threads, which I map onto `i`.

Relevance: the CPU cache line and the GPU warp want the same thing -
 neighbouring threads touching neighbouring addresses.

--------------------------------------------------------------------------------

# Benchmarking Methodology

* **The metric that stays comparable is GLUPS**, lattice updates per
  second. It counts `cells x steps`, which is identical in all three
  backends, so wall-clock time and speedup are fair comparisons.

```
    GLUPS  = cells x steps / (time x 1e9)         common to all backends
```

* **The metrics that are NOT directly comparable across backends** are the two
  derived from GLUPS, because the GPU kernel does different arithmetic
  on a different array set:

```
                     FLOP/cell    traffic range B/cell
    Sequential, OMP      24            20  ..  56
    CUDA                 48            12  ..  60
```

  Quoting raw GFLOPS alone would falsely inflate GPU speedup by
  2x due to on-the-fly arithmetic. So I evaluate GFLOPS and Bandwidth
  against each backend's **Roofline Model** (% of hardware peak).


* **Timing Protocol:**
  * CPU: `clock_gettime(CLOCK_MONOTONIC)` around the solve loop.
  * GPU: `cudaEventRecord` (excludes asynchronous host launch overhead).
  * Timers isolate the solver: allocation, init, and I/O excluded.
  * Platform pinned: `MAXN` power mode, `jetson_clocks`, 3 repeats.

Relevance: GLUPS measures time-to-solution for the physics; Roofline efficiency
 measures how well each implementation exploited its respective hardware.

--------------------------------------------------------------------------------

# Backend 1: Sequential C Baseline

* Plain triple loop, innermost on `i`, no tiling, no threads, `-O3`.

* **Model:** 56 B/cell, 24 FLOP/cell, CGMA = 1.71.

--------------------------------------------------------------------------------

# Backend 2: OpenMP Worksharing with Tiling

```
  +-------------------------------------------------------+
  | FULL 3D GRID: Nx x Ny x Nz                            |
  |                                                       |
  |   +-----------------------+   Tile configuration:     |
  |   | TILE                  |   * OMP_TILE_X = 64       |
  |   | 64 x 32 x 16 cells    |   * OMP_TILE_Y = 32       |
  |   | = 32,768 cells        |   * OMP_TILE_Z = 16       |
  |   +-----------------------+                           |
  +-------------------------------------------------------+
```

```
    #pragma omp parallel for collapse(3) schedule(static)
```

* **`collapse(3)`** over the three tile loops: on its own the outer
  tile loop has only Nz/16 = 8 iterations at N = 128, which is coarse
  for 4 threads. Collapsing merges the three into one iteration space
  and removes the over-decomposition question.
* **`schedule(static)`** because every cell costs exactly the same:
  the load is balanced by construction, so dynamic scheduling would
  only add run-time overhead for nothing.
* **No false sharing.** Each thread writes a disjoint tile, and since
  `OMP_TILE_X = 64` floats = 256 B, tile boundaries fall on cache-line
  boundaries - 64 B fixed chunks of transferred memory each exclusively
  owned by one core. Nothing is shared write-side, no cache line sloshes.


--------------------------------------------------------------------------------

# How Much Does One Tile Actually Weigh?

```
   Tile = 64 x 16 x 16 = 32,768 cells.  Per tile the loop touches:

   u        read, with 1-cell halo : 66 x 18 x 18  = 85.5 KB
   alpha_x  read (idx and i-1)     : 65 x 16 x 16  = 65.5 KB
   alpha_y  read (idx and j-1)     : 64 x 17 x 16  = 68.0 KB
   alpha_z  read (idx and k-1)     : 64 x 16 x 17  = 68.0 KB
   u_next   written                : 64 x 16 x 16  = 64.0 KB
                                                    ----------
   WORKING SET PER TILE                             ~ 348.53 KB
```

* **1 thread:** 350 KiB fits inside the 2 MB shared L2. Tiling works.
* **4 threads:** 4 x 350 KB = **1.4 MB > 2 MB**. Fits as well.
* ** 8 threads:** 8 x 350 KB = **2.8 MB > 2 MB**.  The eight tiles
  begin to evict one another, so the tiling that helps at P <= 4 works
  partly against itself at P = 8.


Relevance: If P = 4 does not perform better than 8 something
 else is going on.


--------------------------------------------------------------------------------

# Backend 3, Step 1: Coalesced Global Access

```
  Warp threads (32):   T0    T1    T2    T3   ...   T30   T31
                       |     |     |     |           |     |
  Array indices (i):   i    i+1   i+2   i+3   ...   i+30  i+31
                       v     v     v     v           v     v
  Global DRAM:       [ 4B ][ 4B ][ 4B ][ 4B ] ... [ 4B ][ 4B ]
                     +---------------------------------------+
                          combined into ONE 128-Byte transaction
```

* When threads of a warp access neighbouring addresses the GPU combines
  them into a single transaction. So `threadIdx.x` is bound strictly to
  the spatial coordinate `i`, which is the contiguous axis.

* Mapping `threadIdx.x -> k` instead would issue **~32 separate** 
  **bus transactions** per warp.

--------------------------------------------------------------------------------

# Backend 3, Step 2: 2.5D Tiling with a Register Window

```
    Shared memory holds ONE XY slice of BOTH fields, plus halos:
    pitch = tile_x + 2*RADIUS = 32 + 2 = 34 floats

    [halo Y]  +---------------------------------------+
              | h |         s_u  and  s_alpha     | h |
    [slice k] | a |        (threadIdx.x,          | a |
              | l |         threadIdx.y)          | l |
              | o |         32 x 4 threads        | o |
              | X |         (128 threads)         | X |
    [halo Y]  +---------------------------------------+

    One dynamic allocation, split in two:  s_alpha = &s_mem[slice]
```

```
    The Z direction lives in 6 registers per thread and slides:

      for (k = 0; k < nz; k++) {
          u_bot = u_curr;  u_curr = u_top;
          a_bot = a_curr;  a_curr = a_top;
          u_top = u[IDX3(i,j,k+1)];      // the only two
          a_top = alpha[IDX3(i,j,k+1)];  // global reads
          ...
          __syncthreads();               // XY tiles are ready
      }
```

* **"Streaming" along Z.** Each `u` and each `alpha` value is read from
  global memory **once**, as `*_top`, then reused twice more as `*_curr`
  and `*_bot` at no memory cost because it never leaves the register file.
* XY halos are loaded by the boundary threads into shared memory.
  Z needs no halo and no `__syncthreads()`, since that reuse is
  register-local rather than shared between threads.
* `alpha` is staged the same way as `u`, so the six harmonic means
  can be built from shared memory and registers only.

Relevance: For a 32 x 4 tile: 
 `Global-Memory-Trasferred-per-Cell = 2 x (1 + 2/32 + 2/4) + 1 = 4.13`
 accesses per cell, i.e. **CGMA 3.20 -> 11.6**.

--------------------------------------------------------------------------------

# Sizing the Block: Shared Memory vs Warp Residency

* **SMP Shared Memory vs. Block Allocation:**
  * Hardware Budget: 64 KB total SRAM per SM.
  * Block Allocation: Each block requests only 1.06 to 9.03 KB for its 2D XY slice.

```
 #threads  XY      smem         Y halo reads      #blocks for 2048 threads
  /block  tile     /block       /cell             -> total shared mem
 ------  -------   ----------   ---------------   -----------------------
    64   32 x 2     1.06 KB        2/2 = 1.00        32  ->  34.0 KB
   128   32 x 4     1.59 KB        2/4 = 0.50        16  ->  25.5 KB
   256   32 x 8     2.66 KB        2/8 = 0.25         8  ->  21.2 KB
   512   32 x 16    4.78 KB       2/16 = 0.12         4  ->  19.1 KB
  1024   32 x 32    9.03 KB       2/32 = 0.06         2  ->  18.1 KB
```
  where `smem/block = 2 * tile_x * (tile_y + 2) * 4 B / 1024 B / KB`
  and for 100% occupancy (2048 threads) the number of resident blocks
  is `#blocks for 2048 threads = 2048 threads / #threads/block`

* Neither exhausts the 64 KB budget, so shared memory is not the binding constraint.

* **What actually limits residency** is the register file:

```
   achieved_occupancy = 0.4998    ->  1024 of 2048 resident threads

   64K registers / 2048 threads = 32 registers per thread budget.
   The stencil - 3D indexing, 6 harmonic means, 6 fluxes - needs more ~64
   so only half the thread slots can be filled whatever the block size.
```

Relevance: Compiling with fixed `-maxrregcount=32` achieves **0 bytes spilled** to
 local memory, boosts `achieved_occupancy` from **50% -> 98%**, and drops solve time.

--------------------------------------------------------------------------------

# Backend 3: Reduction Without Leaving the Device

* I need the domain averages `<u>_total` and `<u>_cube` every `stride`
  steps. Doing that on the host would put a synchronous transfer inside
  the time loop, which would degrade performance.

* Threads can synchronise only within a block, so a block-level tree
  reduction produces **one output per block** - and those partials must
  then be combined either on the CPU or by a second kernel. I chose the
  second kernel, so nothing returns to the host until the run ends.

```
  [1] each thread writes its cell into shared memory
      s_tot[tid] = u[idx];   s_cube[tid] = in_cube ? u[idx] : 0

  [2] tree reduction, log2(B) steps, active threads kept consecutive
      for (stride = B/2; stride > 0; stride >>= 1) {
          if (tid < stride) s_tot[tid] += s_tot[tid + stride];
          __syncthreads();
      }

  [3] thread 0 writes one partial sum per block -> d_block_tot[]

  [4] aggregate_averages_cuda_kernel<<<1,256>>> sums the partials
      in FP64, straight into the device evolution array
```

* Halving the stride from `B/2` downwards keeps the active threads in
  the front of the array, so whole warps retire - no *thread-divergence*.
* Both reductions share **one** dynamic allocation
  (`s_cube = &s_mem[threads_per_block]`) to test for different params.
* Stage 4 accumulates in FP64. Summing 2.1 M FP32 values loses
  significance, and with a single thread the 1/32 FP64 rate is free.

--------------------------------------------------------------------------------

# Verification 1: Method of Manufactured Solutions

* Before benchmarking, the solver has to be correct. I use a case with
  an exact analytic solution:

```
   u(x,y,z,t) = sin(pi x/L) sin(pi y/L) sin(pi z/L) exp(-3 alpha pi^2 t/L^2)

   L = 1,  alpha = 0.143 uniform,  Dirichlet u = 0 on all faces
```

* It satisfies `du/dt = alpha lap(u)` exactly, so every deviation is
  discretisation error plus round-off.

```
    N     h           L2 (seq)     L2 (omp)     L2 (cuda)   order
   ----  ----------  -----------  -----------  -----------  ------
    16   0.058824    4.1986e-04   4.1986e-04   4.1985e-04     -
    32   0.030303    1.0597e-04   1.0597e-04   1.0597e-04    2.076
    64   0.015385    2.6576e-05   2.6576e-05   2.6573e-05    2.040
   128   0.007752    6.4598e-06   6.4598e-06   6.4585e-06    2.064
```

* The error falls **4x per halving of h**; observed order **2.04-2.08**,
  the expected O(h^2).
* Sequential and OpenMP agree to all 7 digits, which also says the
  worksharing introduced no race condition. CUDA differs only in the
  5th digit.

--------------------------------------------------------------------------------

# [FIGURE: fig/mms_test.png]

-> ## All three backends on the theoretical slope <-

* Log-log L2 error against grid spacing h. The three curves are
  indistinguishable and parallel to the reference O(h^2) line.

* **What this rules out:**

  * wrong stencil coefficients - the slope would decrease;
  * a race condition in the OpenMP version - the points would scatter
    between repeats;
  * broken halo logic in CUDA - the green points would drift at large
    N, where the number of tiles grows.

* **What it does not test:** MMS runs at *uniform* alpha, so the
  harmonic mean is executed but never actually has to discriminate
  between two materials.

--------------------------------------------------------------------------------

# [FIGURE: fig/parameter_sweep_times.png]

-> ## Figures of merit vs threads and block size <-

```
   Backend / config          Time (s)          GLUPS    GFLOPS  (FLOP/cell)
  ------------------------  ----------------  -------  ------------------
   Sequential (untiled)      9.788 +/- 0.137   0.0214   0.514      (24)
   OpenMP  1 thread          9.392 +/- 0.410   0.0224   0.537      (24)
   OpenMP  2 threads         4.831 +/- 0.146   0.0434   1.043      (24)
   OpenMP  4 threads         2.878 +/- 0.036   0.0729   1.749      (24)
   OpenMP  8 threads         2.862 +/- 0.027   0.0733   1.759      (24)
   CUDA  block  64           0.905 +/- 0.002   0.2317  11.122      (48)
   CUDA  block 128           0.901 +/- 0.007   0.2328  11.174      (48)
   CUDA  block 256           0.933 +/- 0.005   0.2248  10.789      (48)
   CUDA  block 512           1.000 +/- 0.001   0.2097  10.066      (48)
   CUDA  block 1024          
```

* **The GLUPS column is the comparison**

* **OpenMP** improves to n = 4 and then flattens: 2.878 s against
  2.862 s at n = 8 is a 0.6 % difference against ~1 % scatter, so P = 4
  and P = 8 are indistinguishable.

* **CUDA peaks at block 128** (tile 32 x 4) and then declines
  monotonically to 512. Probably, this is due to the trade-off mentioned on
  the block sizing slide: 
  * 64 pays a full extra read per cell in Y halo,
  * 512 needs 4.78 KB of shared memory per block and cannot keep enough
  blocks resident. 
  * 128 sits at the minimum of the two costs.

* Note: in the previous 24 FLOPS kernel spread was flat, now it is *~10%*.

--------------------------------------------------------------------------------

# [FIGURE: fig/speedup_ideal.png]

-> ## Speedup against the ideal linear curve <-

* **Left - OpenMP against the ideal linear curve S(n) = n:**

```
    n = 1 :  1.00x        n = 2 :  1.94x   (E = 0.97)
    n = 4 :  3.26x        n = 8 :  3.28x   (E = 0.41)
```

  The two contributions separate: tiling alone gives 1.04x
  (9.79 s -> 9.39 s at one thread) and worksharing gives the remaining
  3.26x, for **3.40x** over the sequential baseline.

* **Right - CUDA against the bandwidth-implied ceiling:**

```
    measured best                              = 10.86x  (block 128)
```

* The curve leaves the ideal line sharply between n = 2 and n = 4. The
  next slide asks whether Amdahl's law accounts for that.

--------------------------------------------------------------------------------

# Does Amdahl's Law Explain That Departure?

* The run is a **fixed problem size with increasing thread count**, so
  this is strong scaling and Amdahl's law is the model to test:

```
      S(n) = 1 / ( (1 - p) + p/n )          E(n) = S(n) / n
```

* Fitting the parallel fraction `p` independently at each thread count,
  from the left panel of the previous figure:

```
    n     S(n)     E(n)      p implied by S(n)
   ---   ------   ------    -------------------
    2     1.94     0.97           0.969
    4     3.26     0.82           0.925
    8     3.28     0.41           0.795
```

* **A single `p` does not describe the data.** With p = 0.969 fitted at
  n = 2, Amdahl predicts S(4) = 3.66 and S(8) = 6.55; I measure 3.26 and
  3.28. The curve falls off faster than any serial fraction allows.

* So the limiting factor is not the serial fraction. Two candidates,
  and the data separates them:

  * **n = 8 is not oversubscription damage** - it is simply the same
    performance as n = 4, which is what four physical cores with one
    hardware thread each should give. Nothing is lost, nothing is gained.
  * **n = 4 is where the real ceiling sits**, and the tile-weight slide
    made the falsifiable prediction: 4 x 713 KiB = 2.79 MiB of tiles
    against a 2 MiB L2. The bandwidth slide tests it further.

Relevance: this is the useful part of a scaling study - locating where
you leave the ideal curve, and then being obliged to explain why.

--------------------------------------------------------------------------------

# [FIGURE: fig/speedup_grid_sizes.png]

-> ## Does the speedup survive at N = 384^3 (1.36 GB)? <-

```
   Backend            N = 128^3      N = 384^3      change
  ----------------   -----------    -----------    --------
   Sequential           1.00x          1.00x         ---
   OpenMP  (8 th)       3.42x          3.47x         +1 %
   CUDA (blk 128)      10.86x         10.00x         -8 %
```

```
   Wall clock at N = 384^3, 100 steps (56.6 M cells):
     Sequential    283.02 +/- 0.16 s
     OpenMP 8 th    81.59 +/- 0.09 s
     CUDA blk 128   28.31 +/- 0.29 s
```

* **OpenMP is size-independent** to within 1 %: the tiles are a fixed
  size, so each thread sees the same problem regardless of N.
* **CUDA loses only 8 %** of its advantage over a 27x increase in cell
  count. The 2.5D scheme depends on reuse *inside* a block, in shared
  memory and registers, and that reuse is independent of the total
  working set. Only the residual cross-block reuse through the 256 KB L2
  degrades, and it was never carrying much of the load.
* Reproducibility is good at this size on all three backends
  (sigma <= 1 %), which is what a 28-283 s run should look like on a
  board with pinned clocks.

--------------------------------------------------------------------------------
# What the O(N^5) Law Costs in Practice

* Now that the throughput is measured, the cost law from the set-up can
  be evaluated. Using the best CUDA rate (0.2328 GLUPS) and r = 0.80:

```
   T_phys,max  =  t_wall  x  ( r L^2 LUPS / 6 alpha )  x  N^-5

   N     dt (s)      physical seconds per HOUR of GPU wall clock
  ----  ----------  ---------------------------------------------
    64   2.93e-03            9376 s          (2.6 h of physics)
   128   7.33e-04             293 s
   384   8.15e-05             1.0 s
```

* The N^-5 law is directly visible: 64 -> 128 is a factor 2 in
  resolution and a factor **32** in reachable physical time; 128 -> 384
  is a factor 3 and a factor **293**, against 3^5 = 243 corrected by the
  0.2328/0.200 throughput ratio, which predicts 283.

* **This is what fixes the evolution run.** To watch the cube actually
  cool I need several hundred seconds of physics, and:

```
   500 s of physics costs      N = 64  :    192 s of GPU time
                               N = 128 :    1.7 hours
                               N = 384 :     20 days
```

* So the field-evolution figure is run at **N = 64 for 500 s**, which
  costs about three minutes. Reaching the 1134 s water relaxation time
  at N = 384 would take roughly **7 months** of continuous GPU time.

Relevance: the grid and duration of the physics run were chosen by this
calculation, not by trial. It is also the clearest limiting factor in
the project - the scheme, not the hardware, is what puts a converged
384^3 run out of reach.
--------------------------------------------------------------------------------
# [FIGURE: fig/temperature_evolution.png]

-> ## Verification 2: a conservation law <-

Run at N = 64 for 500 s, as justified on the previous slide.

* Cube average: 80.0 C -> 26.24 C over 500 s of simulated time.
* Domain average: flat.

```
   <u>_total (t = 0)        =  20.395508 C
   <u>_total (t = 499.7 s)  =  20.390408 C
   ---------------------------------------------
   drift = 0.0051 C = 2.5e-4 relative,
   over 170,400 timesteps with zero-flux walls
```

* With insulated boundaries the domain mean is a **conserved quantity**,
  so this is a test and not merely a plot. A 2.5e-4 drift over 1.7e5
  FP32 updates is ~1.5e-9 per step: round-off, not scheme error.

* **This is the test that validates the interface treatment**, and it is
  the only one that does. MMS runs at uniform alpha, where a harmonic
  mean of two equal values is trivially correct. Here the two materials
  differ by 776x, and with an arithmetic mean - or with the
  non-conservative `alpha * lap(u)` form - flux leaks at the copper face
  and this line visibly slopes.

* It also validates the **on-the-fly** harmonic means specifically, since
  this run is the CUDA backend. The GPU rebuilds the face conductances
  from raw `alpha` and still conserves energy to round-off.

* Physics sanity check: at t = 500 s the cube is still at 26 C, well
  above the 20.4 C equilibrium, consistent with the 1134 s water
  relaxation time derived in the set-up.

--------------------------------------------------------------------------------

# [FIGURE: fig/effective_bandwidth.png]

-> ## Utilisation against the 25.6 GB/s spec <-

Each bar spans that backend's own traffic bracket: [20 .. 56] B/cell for
the CPU backends, [12 .. 60] B/cell for CUDA.

```
   Configuration      Effective BW (GB/s)     % of 25.6 GB/s spec
  ----------------   ---------------------   ---------------------
   Seq (1 th)            0.43 -  1.20             1.7 -  4.7 %
   OMP (1 th)            0.45 -  1.25             1.7 -  4.9 %
   OMP (2 th)            0.87 -  2.43             3.4 -  9.5 %
   OMP (4 th)            1.46 -  4.08             5.7 - 15.9 %
   OMP (8 th)            1.47 -  4.10             5.7 - 16.0 %
   CUDA (blk 128)        2.79 - 13.97            10.9 - 54.6 %
```

* This is where my expectation broke. I had read the OpenMP plateau at
  n = 4 as the cores saturating the memory controller - but 4 threads
  reach at most **4.1 GB/s, 16 % of spec**. That is not saturation.
* The 25.6 GB/s figure is a **CPU + GPU aggregate**. What the A57
  cluster alone can sustain is much lower, and I never measured it.
* **The missing experiment:** a STREAM triad on this board would give
  the real CPU ceiling. Until then, "saturated" is an assumption I
  cannot defend, and I would rather say so.
* The CUDA bracket is wide because the two limits are far apart. The
  profiler closes it on the last results slide.

--------------------------------------------------------------------------------

# [FIGURE: fig/roofline_model.png]

-> ## Roofline: below the roof, not on it <-

The roofline is CGMA plotted against bytes instead of accesses
(`I = CGMA / 4 B`), which puts both ceilings on one axis.

```
   Ridge point = 236 / 25.6 = 9.22 FLOP/B.

     backend        I (FLOP/B)   roof = I x 25.6   measured   % of roof
    ------------   -----------  -----------------  ---------  ----------
     Sequential      0.43            11.0 GFLOPS      0.51        5 %
     OpenMP 8 th     0.43            11.0 GFLOPS      1.76       16 %
     CUDA blk 128    2.59 *          66.3 GFLOPS     11.17       17 %

     * measured with the profiler, not assumed - see next slide
```

* The sequential roof, 11.0 GFLOPS, is exactly the CGMA calculation from
  the ceilings slide - the same limit reached two ways.
* **The CUDA point moved right, not just up.** Trading arithmetic for
  bandwidth raised I from 0.43 to 2.59, so the kernel is now a factor
  3.6 below the ridge point instead of a factor 21. It is still
  memory-side, but far less so.
* All three points still sit **6x to 20x below their own memory roof**.
  Were bandwidth the binding constraint they would lie *on* the sloped
  line, and they do not.
* So the roofline is useful here in the negative: it excludes both
  ceilings, and what remains is latency and warp residency - which is
  what the 50 % occupancy figure says independently.

--------------------------------------------------------------------------------

# Closing the Bracket: Profiling the Kernel

The course tools are `htop` and `jtop`; to get per-kernel numbers I used
`nvprof`, which is a step beyond the lectures.

```
   nvprof --metrics gld_throughput,gst_throughput,achieved_occupancy,
                    sysmem_read_throughput,sysmem_write_throughput
          ./heat_cube_cuda_3d --n 128 --steps 100 --block_size 128
```

```
   solve_stencil_cuda_kernel, averaged over 100 invocations:

   gld_throughput  = 3.494 GB/s      gst_throughput = 0.962 GB/s
   ------------------------------------------------------------
   load/store ratio = 3.63
   1 store = 4 B/cell   ->   loads = 14.5 B/cell
   total requested traffic = 18.5 B/cell = 4.63 accesses/cell

   CGMA_measured = 48 / 4.63 = 10.4     (naive: 3.20)
```

* **Cross-check against the design.** The model for a 32 x 4 tile is
  `2 x (1 + 2/32 + 2/4) + 1 = 4.13` accesses/cell against **4.63
  measured**, 12 % apart. The gap is the halo again: the `tx == 0` and
  `ty == 0` loads execute on a full warp with one active lane, so they
  cost more than the one element they fetch.
* Together the two optimisations take the kernel from **60 B/cell to
  18.5 B/cell**, and the memory-imposed ceiling from 11.0 GFLOPS
  (precomputed, no reuse) to **66 GFLOPS**.
* `achieved_occupancy = 0.4998` - exactly half the thread slots, and
  unchanged from the previous kernel version, confirming that the
  register file and not shared memory sets residency.
* `sysmem_read/write_throughput = 0` confirms zero PCIe traffic, as
  expected on unified memory.
* The reduction kernel reaches **0.96 occupancy** and runs 11 times
  against the stencil's 100, so it is not on the critical path.

Relevance: replacing a modelled bracket with a measured number, then
checking it against my own design estimate, is the part of this project
I would keep if I had to throw the rest away.

--------------------------------------------------------------------------------

# Summary of the Parameter Sweeps

```
+-------------------+------------------+-------------------------------------+
| Dimension         | Range tested     | Observed trend                      |
+-------------------+------------------+-------------------------------------+
| Grid size N       | 128, 384         | CUDA 10.86x -> 10.00x (-8 %); OMP   |
|                   | (64 for physics) | 3.42x -> 3.47x. Block-local reuse   |
|                   |                  | is size-independent, so both hold.  |
+-------------------+------------------+-------------------------------------+
| OpenMP threads n  | 1, 2, 4, 8       | E = 0.97, 0.82, 0.41. Leaves the    |
|                   |                  | ideal curve between n=2 and n=4,    |
|                   |                  | faster than any single serial       |
|                   |                  | fraction allows. n=8 = n=4.         |
+-------------------+------------------+-------------------------------------+
| CUDA block size B | 64, 128, 256,    | Peak at 128 (32x4). 64 pays a full  |
|                   | 512              | extra Y-halo read per cell; 512     |
|                   |                  | needs 4.8 KiB smem/block. 10 %      |
|                   |                  | spread: second-order, not flat.     |
+-------------------+------------------+-------------------------------------+
| Interface alpha   | precomputed vs   | on-the-fly on GPU: +1 read, +24     |
|                   | on-the-fly       | FLOP, CGMA 1.71 -> 3.20 naive.      |
+-------------------+------------------+-------------------------------------+
| Precision         | FP32 throughout  | FP64 costs a factor 32 on sm_53.    |
+-------------------+------------------+-------------------------------------+
```

Relevance: the shapes matter as much as the values. Block size became a
real trade-off only once shared memory held two fields; before that it
was flat, and a flat sweep is itself the finding that the dimension is
not a limiting factor.

--------------------------------------------------------------------------------

-> # Limiting Factors: What I Found <-

* **1. The physics wall is O(N^5), and it is absolute.**
  One hour of GPU time buys 9376 s of simulated time at N = 64, 293 s at
  N = 128, and 1 s at N = 384. That law, not the hardware, is what fixed
  the configuration of every run in this project.

* **2. Correctness holds across all three backends.**
  Observed order 2.04-2.08 on the manufactured solution, and the domain
  mean conserved to 2.5e-4 over 170,400 steps with a 776x material jump.

* **3. Both CGMA routes paid off, and the second one more than the first.**
  Reuse (2.5D tiling) and extra arithmetic (on-the-fly harmonic means)
  together took the kernel from 60 to **18.5 B/cell measured**, CGMA
  3.20 -> 10.4, and **10.86x** over the sequential baseline.

* **4. I was wrong about the bottleneck, and that is the useful part.**
  I expected bandwidth saturation on the CPU and shared-memory pressure
  on the GPU. In fact every backend runs 6x-20x below its own memory
  roof, the CPU never exceeds 16 % of the bandwidth spec, and the GPU
  sits at exactly 50 % occupancy limited by the **register file**. On a
  single-SMP board, warp residency is the currency.

* **5. The three experiments I would run next:**
  * STREAM triad, to replace the assumed 25.6 GB/s CPU ceiling.
  * `-maxrregcount`, to trade register spills against warp residency -
    now more pressing, since the harmonic means added register pressure.
  * OMP tile 32x16x8, to test the L2 oversubscription hypothesis.

<br>

-> Run: `mdp presentation.md`  |  figures in `fig/`, data in `data/` <-
