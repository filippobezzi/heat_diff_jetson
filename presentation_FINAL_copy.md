%title: Computational Limits of the Jetson Nano - 3D Heat Diffusion
%author: Filippo Bezzi
%date: 2026-09-08

-> # Computational Limits of the NVIDIA Jetson Nano <-
-> ### 3D Heterogeneous Heat Diffusion from Ground Zero ### <-

<br>

-> Modern Computing for Physics (MCP) <-
-> Final Project <-

<br>

--------------------------------------------------------------------------------

# 1. Outline

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
   Sequential C  ->  OpenMP directives  ->  CUDA
```

Set-up and methods first; every measured number appears in section 6.
Benchmark point: N = 128^3, 100 steps, 3 repeats, mean +/- sigma.

--------------------------------------------------------------------------------

# 2. The Physical Task

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
  The material jump is what forces a conservative discretisation and
  ... a very small timestep.

Relevance: the material contrast, not the grid size, is what sets the
cost of this simulation. Everything that follows descends from that 776.

--------------------------------------------------------------------------------

# 3. Why the Divergence Form Is Not Optional

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
  different needs: the CPU precomputes them once into `alpha_x`,
  `alpha_y`, `alpha_z`;
  the GPU rebuilds them inside the kernel. I return to this later.

--------------------------------------------------------------------------------

# 4. The 3D 7-Point Stencil: Counting Operations and Accesses

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
   F_x^R = alpha_x[i,j,k]  * (u[i+1,j,k] - u[i,j,k]) * invdx2  <- 1 sub, 2 mul
   F_x^L = alpha_x[i-1,j,k] * ( u[i,j,k] - u[i-1,j,k] )     ... x 6 faces

   COMPUTE : 6 x 3 + 5 combine + 1 update      =  24 FP operations / cell
   ACCESSES: 7(1) u-reads + 6(3) alpha-reads + 1 write = 14(5) accesses / cell
                                                 = 56(20) Bytes / cell (FP32)

   EFFECTIVE MEMORY BANDWIDTH := minimal access (cache reuse) / cell
                               = 20 * GLUPS * Bytes / cell
```

* **CGMA = 24 / 5 = 4.8 FP operations per compulsory memory access.**
    Theoretical CGMA = 36.84 FLOP / access

Relevance: this gives the insight that drove the implementation - a low
  CGMA means the task is memory-bound, so optimisation should
  target memory accesses, not operations.


--------------------------------------------------------------------------------

# 5. Stability: the Copper Sets the Timestep for Every Cell

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

# 6. The O(N^5) Cost Wall

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

* Two consequences:

  * the only free parameter on the right is `LUPS`, so **all three**
   **backends obey the same N^-5 law** and differ only by a prefactor;
  * a factor 3 in resolution costs a factor 3^5 = 243 in reachable
   physical time. Refinement cannot be considered a matter of waiting
   longer.

* I evaluate this with the measured throughput in the results section,
  where it also fixes the grid and duration of the evolution run.

--------------------------------------------------------------------------------

# 7. The Platform: One Chip, One Memory Pool

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
         |   (64 B line)      |          |   (32 B line)      |
         +--------------------+          +--------------------+

```

Relevance: each hardware block motivates a specific implementation
 choice:
  * 4 GB => hard upper ceiling on the maximum grid dimensions N^3;
  * Small CPU/GPU caches => explicit spatial cache blocking and 2.5D tiling;
  * 128 cores on 1 SMP => latency hidden via occupancy and coalescing;
  * Unified 25.6 GB/s bus => the ultimate ceiling for both engines.


--------------------------------------------------------------------------------

# 8. Ceilings Derived Before Writing Code

* **Peak FP32 (GPU):**

```
    128 CUDA cores x 2 FLOP/cycle (FMA) x 0.921 GHz  =  236 GFLOP/s
```

* **Peak bandwidth:** `8 B x 1600 MHz x 2 (DDR) = 25.6 GB/s`

* **The ceiling that actually binds**, from CGMA:

```
    max FLOPS = ( 25.6 GB/s / 4 B ) x CGMA = 6.4 x CGMA

      no reuse at all  CGMA = 24/14 = 1.71  ->  11.0 GFLOP/s   4.7 %
      perfect reuse    CGMA = 24/ 5 = 4.80  ->  30.7 GFLOP/s  13.0 %
```

* So even with a perfect cache the memory system caps the naive
  formulation at **30.7 of 236 GFLOP/s - 13 % of the compute peak**.

* **Machine balance:** `236 / 25.6 = 9.22 FLOP/Byte`, or 36.8 FLOP per
  4-byte access (CGMA). Below that no kernel can be compute-limited. The
  compulsory stencil sits at `24/20 = 1.20 FLOP/B`, a factor 7.7 below.

* **FP64:** on Maxwell sm_53 the FP64 issue rate is 1/32 of FP32.
  Everything is FP32; FP64 would have cost a factor 32 for no benefit.

Relevance: To raise the ceiling I must raise CGMA, and there are 
 two ways to do it - reuse data instead of re-reading it,
 or **do more arithmetic per byte fetched**. I used both.

--------------------------------------------------------------------------------

# 9. One SMP: Latency Hiding Is the Whole Game

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

# 10. The Memory Hierarchy Suggested the Algorithm

```
   Latency          Space              Scope        Size
  -----------      -------            --------     ----------------
  ~0 cycles        Registers          thread       256 KB / SMP
  ~10-100 cycles   Shared memory      block        48 KB / block
  ~20 cycles       L2 cache           grid         256 KB
  400-800 cycles   Global (LPDDR4)    grid         4 GB shared pool
```

* A 7-point stencil naively reads every `u` value **seven times**: once
  as the centre cell, twice as an X/Y/Z neighbour.

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
 CGMA reuse.

--------------------------------------------------------------------------------

# 11. Memory Budget: Choosing the Grid Sizes

* The two backend families keep a different number of FP32 arrays alive,
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

# 12. Raising CGMA the Second Way: Arithmetic for Bandwidth


* The CPU version precomputes the interface diffusivity arrays once and
  reads them per cell. On the GPU, I do the opposite: store only the raw
  `alpha` field and rebuild the six harmonic means **on-the-fly inside the**
  **kernel**, from values already in shared memory and registers.

```
                          reads/cell    FLOP/cell    CGMA
   -------------------   ------------  -----------  -------
   precomputed faces     7(1)u + 6(3)a     24        4.80
   on-the-fly means      7(1)u + 7(1)a     48       16.00

```

* One extra `alpha` read, but the arithmetic doubles: each harmonic mean
  is `2ab/(a+b)`, i.e. *2 multiplies*, *1 add* and **1 division**, six times
  per cell.

* On this machine that is a good trade in one direction only:

  * The GPU has a machine balance of 9.22 FLOP/B and (naively) sits at 4.8.
    It has arithmetic units idle while waiting on memory, so paying FLOPs
    to avoid bytes moves it *towards* the ridge point.

  * Cortex-A57 CPU has limited scalar ALUs and expensive division. Spending
    24 extra FLOPs per cell would degrade throughput, so the CPU keeps
    the precomputed arrays.

Relevance: The three backends now differ algorithmically.

--------------------------------------------------------------------------------

# 13. One Data Layout for Three Backends

```
   Row-major linearisation:  IDX3(i,j,k) = (k * Ny + j) * Nx + i

           _______________ (nx,ny,0)
          /ny            /|
         / |            / |
      j /  |           /  |        i (X): contiguous in memory
       /0   <--i--> nx/   |        j (Y): strides by Nx
      /______________/    |        k (Z): strides by Nx*Ny
      |0   |         |    |
      |    |_________|____|
      |   /          |   /
    k |  /           |  /
      | /            | /
      |______________|/

```

This data layout ensures **CPU spatial locality** and **GPU memory coalescing**:
* CPU: A 64 B cache line fetches 16 contiguous floats along `i` (4 B x 16 = 64 B).
* GPU: A 32-thread warp accesses 32 contiguous floats along `i` (4 B x 32 = 128 B coalesced).

Relevance: Row-major contiguous layout along `i` satisfies both architectures 
 simultaneously - the CPU cache line and the GPU warp want the same thing.

--------------------------------------------------------------------------------

# 14. Benchmarking Methodology

* **The metric that stays comparable is GLUPS**, lattice updates per
  second. It counts `cells x steps`, which is identical in all three
  backends, so wall-clock time and speedup are fair comparisons.

```
    GLUPS  = cells x steps / (time x 1e9)         common to all backends
```

* **The metrics that are NOT directly comparable across backends** are the two
  derived from GLUPS, because the GPU kernel does different
  arithmetic on a different array set:

```
                     FLOP/cell    traffic range B/cell
    Sequential, OMP      24            20  ..  56
    CUDA                 48            12  ..  60
```

  Quoting raw GFLOPS alone would falsely inflate GPU speedup by
  2x due to on-the-fly arithmetic. So I evaluate GFLOPS and Bandwidth
  against each backend's **Roofline Model** (% of hardware peak):
  * It bounds the maximum achievable performance (GFLOPS) as a function
   of arithmetic intensity (FLOPs/B)
  * The Ridge Point `I_ridge = CGMA / 4 B / access` is the minimum arithmetic intensity required 
   to reach peak compute throughput.

* **Timing Protocol:**
  * CPU: `clock_gettime(CLOCK_MONOTONIC)` around the solve loop.
  * GPU: `cudaEventRecord` (excludes asynchronous host launch overhead).
  * Timers isolate the solver: allocation, init, and I/O excluded.

Relevance: GLUPS measures time-to-solution for the physics; roofline
 efficiency measures how well each implementation exploited its respective hardware.

--------------------------------------------------------------------------------

# 15. Backend 1: Sequential C Baseline

* Plain triple loop, innermost on `i`, no tiling, no threads, `-O3`.

* **Model:** 20 B/cell, 24 FLOP/cell, CGMA = 4.80.

Relevance: Boring.

--------------------------------------------------------------------------------

# 16. Backend 2: OpenMP Worksharing with Tiling

```
  +-------------------------------------------------------+
  | FULL 3D GRID: Nx x Ny x Nz                            |
  |                                                       |
  |   +-----------------------+   Tile configuration:     |
  |   | TILE                  |   * OMP_TILE_X = 64       |
  |   | 64 x 16 x 16 cells    |   * OMP_TILE_Y = 16       |
  |   | = 16,384 cells        |   * OMP_TILE_Z = 16       |
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

# 17. How Much Does One Tile Actually Weigh?

```
   Tile = 64 x 16 x 16 = 16,384 cells.  Per tile the loop touches:

   u        read, with 1-cell halo : 66 x 18 x 18  =  83.53 KiB
   alpha_x  read (idx and i-1)     : 65 x 16 x 16  =  65.00 KiB
   alpha_y  read (idx and j-1)     : 64 x 17 x 16  =  68.00 KiB
   alpha_z  read (idx and k-1)     : 64 x 16 x 17  =  68.00 KiB
   u_next   written                : 64 x 16 x 16  =  64.00 KiB
                                                     -----------
   WORKING SET PER TILE               356,896 B    = 348.53 KiB
```

* **1 thread:**  348.5 KiB, 17 % of the 2 MiB shared L2.
* **4 threads:** 4 x 348.5 KiB = **1.36 MiB < 2 MiB**. Still fits, at
  68 % of the cache.
* **8 threads:** 8 x 348.5 KiB = **2.72 MiB > 2 MiB**. The eight tiles
  begin to evict one another - but there are only 4 physical cores, so
  n = 8 is oversubscription anyway and the two effects are confounded.

* A `64 x 32 x 16` tile weighs 683.8 KiB, so four of them (**2.67 MiB**)
  do not fit.


--------------------------------------------------------------------------------

# 18. Backend 3, Step 1: Coalesced Global Access

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

# 19. Backend 3, Step 2: 2.5D Tiling with a Register Window

```
    Shared memory holds ONE XY slice of BOTH fields, plus halos:
    pitch = tile_x + 2*RADIUS = 32 + 2 = 34 floats
    slice = pitch * (tile_y + 2*RADIUS)

    [halo Y]  +---------------------------------------+
              | h |         s_u  and  s_alpha     | h |
    [slice k] | a |        (threadIdx.x,          | a |
              | l |         threadIdx.y)          | l |
              | o |         32 x 8 threads        | o |
              | X |         (256 threads)         | X |
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

Spoiler: the 32 x 8 tile the model will perform best. Counting the over-fetch
 ratios, it gives `accesses/cell = 2 x (1 + 2/32 + 2/8) + 1 = 3.63`, 
 against **4.14 measured** by the profiler.

--------------------------------------------------------------------------------


# 20. Sizing the Block: Shared Memory vs Warp Residency

* **SMP Shared Memory vs. Block Allocation:**
  * Hardware Budget: 64 KB total SRAM per SM.
  * Block allocation: 1.06 to 9.03 KB per block for its 2D XY slice.

```
 #threads  XY      smem       total halo      resident blocks
  /block  tile     /block     reads/cell      for 2048 threads -> smem
 ------  -------   --------   -------------   ------------------------
    64   32 x 2     1.06 KB     2 x 1.06        32  ->  34.0 KB
   128   32 x 4     1.59 KB     2 x 0.56        16  ->  25.5 KB
   256   32 x 8     2.66 KB     2 x 0.31         8  ->  21.2 KB
   512   32 x 16    4.78 KB     2 x 0.19         4  ->  19.1 KB
  1024   32 x 32    9.03 KB     2 x 0.125        2  ->  18.1 KB
```
  `smem/block = 2 x (tile_x + 2) x (tile_y + 2) x 4 B`, the leading 2
  because **both** `u` and `alpha` are staged. Halo reads per cell are
  `2 x (2/tile_x + 2/tile_y)`, also doubled for the same reason.

* No configuration exhausts the 64 KB budget, so **shared memory is not**
  **the binding constraint**. The register file was:

```
   Xptxas -v, same source, only the flag differs:

     default          : 64 registers, 0 bytes spill stores / loads
     -maxrregcount=32 : 32 registers, 0 bytes spill stores / loads
             
            (total # reg)
   64 regs th -> 65536/64 = 1024 resident threads ->  50 %  (measured 0.4997)
   32 regs th -> 65536/32 = 2048 resident threads -> 100 %  (measured 0.9852)

```

* The compiler was using 64 registers per thread because it could,
  not because it needed to: **zero spills in both builds**, so the 32 cap is
  free. Every block size reaches full residency.

--------------------------------------------------------------------------------

# 21. Backend 3: Reduction Without Leaving the Device

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
      in FP64, into the device evolution array
```

* Halving the stride from `B/2` downwards keeps the active threads in
  the front of the array. Whole warps retire - no *thread-divergence*.
* Both reductions share **one** dynamic allocation
  (`s_cube = &s_mem[threads_per_block]`) to test for different blockDims.
* Stage 4 accumulates in FP64. Summing 2.1 M FP32 values loses
  significance, and with a single thread the 1/32 FP64 rate is free.

--------------------------------------------------------------------------------

# 22. Verification 1: Method of Manufactured Solutions

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
    16   0.058824    4.1986e-04   4.1986e-04   4.1986e-04     -
    32   0.030303    1.0597e-04   1.0597e-04   1.0597e-04    2.076
    64   0.015385    2.6697e-05   2.6697e-05   2.6695e-05    2.034
   128   0.007752    6.7844e-06   6.7844e-06   6.7837e-06    1.999
```

* The error falls **4x per halving of h**; observed order **2.08, 2.03,  2.00**,
  in line with the expected O(h^2) as the grid refines.
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

--------------------------------------------------------------------------------

# [FIGURE: fig/parameter_sweep_times.png]

-> ## Figures of merit vs threads and block size <-

```
   Backend / config          Time (s)            GLUPS   GFLOPS  BW GB/s
  ------------------------  -----------------   ------  -------  -------
   Sequential (untiled)      9.9223 +/- 0.2741  0.02113   0.507   0.423
   OpenMP   1 thread         8.9536 +/- 0.0596  0.02343   0.562   0.468
   OpenMP   2 threads        4.8226 +/- 0.0868  0.04350   1.044   0.870
   OpenMP   4 threads        2.9353 +/- 0.0168  0.07147   1.715   1.429
   OpenMP   8 threads        2.8356 +/- 0.0638  0.07397   1.776   1.480  <-
   OpenMP  16 threads        2.9119 +/- 0.0118  0.07200   1.728   1.440
   CUDA  block   64          1.0812 +/- 0.0006  0.19393   9.310   2.327
   CUDA  block  128          0.9904 +/- 0.0006  0.21173  10.164   2.541
   CUDA  block  256          0.9501 +/- 0.0004  0.22070  10.594   2.648  <-
   CUDA  block  512          0.9629 +/- 0.0004  0.21777  10.454   2.613
   CUDA  block 1024          1.0506 +/- 0.0003  0.19963   9.582   2.396

   GFLOPS uses 24 FLOP/cell on CPU, 48 on GPU; BW uses the compulsory
   traffic, 20 B/cell on CPU and 12 B/cell on GPU.
```

* **GLUPS** are used to compare across backends.

* **OpenMP** improves to n = 4 and then flattens: 2.9353 s at n = 4,
  2.8356 s at n = 8, 2.9119 s at n = 16 - a 3.4 % band, and the n=4 /
  n=8 gap is only 1.5 sigma. Beyond four threads nothing changes, which
  is what four physical cores with one hardware thread each should give.

* **CUDA peaks at block 256** (tile 32 x 8), 0.9501 s, and the spread
  across the sweep is **10.6 %**. 
  Two effects:
  * below 256 the Y-halo dominates - block 64 pays `2 x 1.06` extra
    reads per cell against `2 x 0.31` at block 256;
  * above 256 the halo keeps shrinking, but resident blocks fall from
    8 to 4 => each barrier now spans 16 warps instead of 8.

* Note: with the (naive) 24-FLOP kernel this sweep was flat to 3 %.
  Staging a second array in shared memory is what made block size a
  real parameter.

--------------------------------------------------------------------------------

# [FIGURE: fig/speedup_ideal.png]

-> ## Speedup against the ideal linear curve <-

* **Left - OpenMP against the ideal linear curve S(n) = n:**

```
    n =  1 :  1.00x              n =  2 :  1.86x   (E = 0.93)
    n =  4 :  3.05x  (E = 0.76)  n =  8 :  3.16x   (E = 0.39)
    n = 16 :  3.08x  (E = 0.19)
```

  The two contributions separate: tiling alone gives 1.108x (or more
  given the overhead due to calling of OpenMP methods) 
  Worksharing clearly gives remaining advantage:
  3.158x over 1 thread, **3.499x** over the sequential baseline.

* **Right - CUDA against the bandwidth-implied ceiling:**

```
    measured best                              = 10.44x  (block 256)
```

* The curve leaves the ideal line sharply between n = 2 and n = 4. The
  next slide asks whether Amdahl's law accounts for that.

--------------------------------------------------------------------------------

# [FIGURE: fig/speedup_grid_sizes.png]

-> ## Does the speedup survive at N = 384^3 (1.36 GB)? <-

```
   Backend            N = 128^3      N = 384^3      change
  ----------------   -----------    -----------    --------
   Sequential           1.00x          1.00x         ---
   OpenMP  (8 th)       3.50x          3.73x         +6.6 %
   CUDA (blk 256)      10.44x         11.21x         +7.3 %
```

```
   Wall clock at N = 384^3, 100 steps (56.6 M cells):
     Sequential    290.022 +/- 0.405 s      (sigma = 0.14 %)
     OpenMP 8 th    77.751 +/- 0.108 s      (sigma = 0.14 %)
     CUDA blk 256   25.878 +/- 0.004 s      (sigma = 0.02 %)

   Throughput, N = 128 -> N = 384:
     Sequential    0.02113 -> 0.01950 GLUPS   -7.73 %
     OMP 8 th      0.07397 -> 0.07280 GLUPS   -1.58 %
     CUDA blk 256  0.22070 -> 0.21880 GLUPS   -0.86 %
```

* **The speedups do not degrade with size - they improve slightly.**
  The gain is in the denominator, not the numerator: every backend loses
  throughput at 56.6 M cells, but the sequential baseline loses **9x
  more** than CUDA (-7.7 % against -0.86 %). The untiled triple loop is
  the only one whose reuse depends on the total working set.

* **CUDA throughput is flat to 0.86 %** across a **27x** increase in cell
  count. The 2.5D scheme reuses data *inside* a block, in shared memory and registers,
  and that reuse does not care how large the total working set is.

Relevance: a change from the naive kernel, which lost 25 % going to
  N = 384 because it read three `alpha` arrays from global memory and
  depended on L2 to catch them. Rebuilding the harmonic means from a
  single staged array removed that dependence.

--------------------------------------------------------------------------------

# 28. What the O(N^5) Law Costs in Practice

* Now that the throughput is measured, the cost law from the set-up can
  be evaluated. Using the best CUDA rate (0.22070 GLUPS) and r = 0.80,
  the prefactor is K = 3600 x (0.80 x 100^2 x 0.22070e9) / (6 x 111.0)
  = 9.5439e12:

```
   T_phys,max  =  t_wall  x  ( r L^2 LUPS / 6 alpha )  x  N^-5

   N     dt (s)      physical seconds per HOUR of GPU wall clock
  ----  ----------  ---------------------------------------------
    64   2.933e-03           8888 s          (2.47 h of physics)
   128   7.332e-04            277.8 s
   384   8.146e-05              1.133 s     (at its own 0.21880 GLUPS)
```

* The N^-5 law is directly visible: 64 -> 128 is a factor 2 in
  resolution and a factor **32.0** in reachable physical time;
  128 -> 384 is a factor 3 and a measured factor **245.1**, against
  3^5 x (0.22070/0.21880) = **245.1** predicted.

* **This is what fixes the evolution run.** To watch the cube actually
  cool I need several hundred seconds of physics, and:

```
   500 s of physics costs      N = 64  :    202.5 s of GPU time
                               N = 128 :    6480 s   (1.80 hours)
                               N = 384 :    1.588e6 s (18.4 days)

   tau_w = 1134 s costs        N = 64  :    459 s     (7.7 minutes)
                               N = 128 :    14,698 s  (4.08 hours)
                               N = 384 :    3.603e6 s (41.7 days)
```

* So the field-evolution figure is run at **N = 64 for 500 s**, which
  costs about three and a half minutes. Reaching the 1134 s water
  relaxation time at N = 384 would take **41.7 days** of continuous
  GPU time.

--------------------------------------------------------------------------------

# [FIGURE: fig/temperature_evolution.png]

-> ## Verification 2: a conservation law <-

Run at N = 64 for 500 s.

* Cube average: 80.0 C -> 26.24 C over 500 s of simulated time.
* Domain average: flat.

```
   <u>_total (t = 0)        =  20.395508 C
   <u>_total (t = 499.7 s)  =  20.390408 C
   ---------------------------------------------
   drift = 0.0051 C = 2.5e-4 relative,
   over 170,400 timesteps
```

* With insulated boundaries the domain mean is a **conserved quantity**.
  A 2.5e-4 drift over 1.7e5 FP32 updates is ~1.5e-9 per step: round-off error.

* Physics sanity check: at t = 500 s the cube is still at 26 C, well
  above the 20.4 C equilibrium, consistent with the 1134 s water
  relaxation time derived in the set-up.

--------------------------------------------------------------------------------

# [FIGURE: fig/effective_bandwidth.png]

-> ## Utilisation against the 25.6 GB/s spec <-


```
   BW_eff = GLUPS x B_compulsory      B_compulsory = 20 B/cell (CPU)
                                                     12 B/cell (GPU)
```

```
   Configuration      BW_eff (GB/s)     % of 25.6 GB/s spec
  ----------------   ---------------   ---------------------
   Seq (1 th)             0.4227               1.65 %
   OMP (1 th)             0.4685               1.83 %
   OMP (2 th)             0.8699               3.40 %
   OMP (4 th)             1.4289               5.58 %
   OMP (8 th)             1.4797               5.78 %
   CUDA (blk 256)         2.6484              10.35 %
```

* Equivalently, as a rate of lattice updates against what the memory
  system allows for each algorithm:

```
   GLUPS_max = 25.6 / B_compulsory    CPU 1.280   GPU 2.133  GLUPS

   best OMP  0.07397 / 1.280 =  5.78 %
   best CUDA 0.22070 / 2.133 = 10.35 %    <- same numbers, one unit
```

* OpenMP n = 4: the cores do not saturate the memory.
* The 25.6 GB/s figure is a **CPU + GPU aggregate**. What the A57
  cluster alone can sustain is much lower, and I never measured it.

--------------------------------------------------------------------------------

# [FIGURE: fig/roofline_model.png]

-> ## Roofline: well below the roof <-

The roofline is CGMA plotted against bytes instead of accesses
(`I = CGMA / 4 B`), which puts both ceilings on one axis.

```
   Ridge point = 236 / 25.6 = 9.22 FLOP/B.
   Points plotted at the compulsory intensity, I = FLOP / B_compulsory:

     backend        I (FLOP/B)   roof = I x 25.6   measured   % of roof
    ------------   -----------  -----------------  ---------  ----------
     Sequential     24/20 = 1.20     30.7 GFLOPS     0.507      1.65 %
     OpenMP 8 th    24/20 = 1.20     30.7 GFLOPS     1.776      5.78 %
     CUDA blk 256   48/12 = 4.00    102.4 GFLOPS    10.594     10.35 %

   Profiler cross-check on the CUDA point:
     I_measured = 48 / 16.57 B = 2.90 FLOP/B -> roof 74.2 GFLOPS -> 14.3 %
```

* **The CUDA point moved right.**
* The "% of roof" column is numerically identical to the "% of peak
  bandwidth" column on the previous slide. That is not a coincidence:
  `GFLOPS/roof = (I x BW_eff)/(I x BW_peak) = BW_eff/BW_peak`. The two
  slides are one measurement seen in two units.
* All three points sit **10x to 61x below their own memory roof**
  (60.6x sequential, 17.3x OpenMP 8, 9.7x CUDA).

Relevance: the roofline excludes the compute ceiling *and* the bandwidth ceiling.

--------------------------------------------------------------------------------

# 32. Closing the Bracket: Profiling the Kernel

To get per-kernel numbers I used `nvprof`:

```
   nvprof --metrics gld_throughput,gst_throughput,achieved_occupancy,
                    sysmem_read_throughput,sysmem_write_throughput
          ./heat_cube_cuda_3d --n 128 --steps 100 --block_size 256
```

```
   solve_stencil_cuda_kernel, averaged over 100 invocations:

   gld_throughput  = 2.838 GB/s      gst_throughput = 0.903 GB/s
   ------------------------------------------------------------
   load/store ratio = 3.143
   1 store = 4 B/cell   ->   loads = 12.57 B/cell
   total requested traffic = 16.57 B/cell = 4.14 accesses/cell

   CGMA_measured = 48 / 4.14 = 11.6
   gld_efficiency     = 79.61 %
   achieved_occupancy = 0.9852
```

* **Cross-check against the design.** The model for a 32 x 8 tile is
  `2 x (1 + 2/32 + 2/8) + 1 = 3.63` accesses/cell against 
  **4.14 measured**, 14 % apart. 

--------------------------------------------------------------------------------

# 33. Summary

<br>

-> Run: `mdp presentation.md`  |  figures in `fig/`, data in `data/` <-
