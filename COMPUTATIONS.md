# Performance Metrics, Hardware Architecture & Computational Optimizations
### 3D Heterogeneous Heat Diffusion on the NVIDIA Jetson Nano

**Status:** reference sheet for every number quoted in `presentation_FINAL.md`.

**Data of record:** `data/benchmarks_reduced.csv` (N = 128, 100 steps, 3 repeats),
`data/benchmarks_large.csv` (N = 384), `data/mms_convergence_3d.csv`,
`data/avg_u_evo_3d.csv`. 

**If a number in a slide disagrees with this file, this
file is wrong until the CSV is re-read — the CSVs are the only source of truth.**

**Code constants of record:** `src/heat_3d.h` (`FLOPS_PER_CELL`,
`BYTES_PER_CELL_MIN`, `OMP_TILE_*`), `src/heat_cube_cuda_3d.cu`,
`src/heat_cube_omp_3d.c`, `src/Makefile`.

**Related documents:** `../utils/METRICS.md`, `../utils/COMPUTATIONS.md`,
`../utils/AUDIT.md`, `../utils/Guidelines_MCP2025.pdf`. Course slides
`../../01_ParallelProcessingRecap.pdf` … `../../15_matmul*.pdf`.

**Reading rule.** Every figure below is either (a) read directly from a CSV,
(b) derived by an equation stated in full on this page, or (c) read from a
profiler run whose exact invocation is quoted. Nothing is asserted without one
of those three provenances.

---

## 0. The three canonical operating points

Everything else is derived from these three rows of `benchmarks_reduced.csv`.

| Role | Configuration | Time (s) | GLUPS |
|---|---|---:|---:|
| Serial baseline | Sequential C, untiled, 1 thread | 9.9223 ± 0.2741 | 0.02113 |
| Best CPU | OpenMP, 8 threads, tiled 64×16×16 | 2.8356 ± 0.0638 | 0.07397 |
| Best GPU | CUDA, block 256 (tile 32×8) | 0.9501 ± 0.0004 | 0.22070 |

Benchmark point: $N = 128^3 = 2{,}097{,}152$ cells, $N_{\text{steps}} = 100$,
total lattice updates $= 209{,}715{,}200$, 3 repeats, mean ± σ.

---

## 1. Metrics: definitions, formulations and explicit calculations

### 1.1 The invariant figure of merit: lattice updates per second

Different backends do different amounts of arithmetic on different array sets,
so GLUPS — spatial updates per unit time — is the **only** invariant
cross-backend figure of merit. It is what wall-clock time and speedup are
computed from.

$$\text{LUPS} = \frac{N_x N_y N_z \, N_{\text{steps}}}{t_{\text{solve}}},
\qquad \text{GLUPS} = \frac{\text{LUPS}}{10^9}$$

Explicit, with $N_x N_y N_z N_{\text{steps}} = 2.097152\times10^{6}\times100 = 2.097152\times10^{8}$:

$$\text{GLUPS}_{\text{seq}} = \frac{2.097152\times10^{8}}{9.9223 \times 10^{9}} = 0.02113$$

$$\text{GLUPS}_{\text{omp8}} = \frac{2.097152\times10^{8}}{2.8356 \times 10^{9}} = 0.07396$$

$$\text{GLUPS}_{\text{cuda256}} = \frac{2.097152\times10^{8}}{0.9501 \times 10^{9}} = 0.22073$$

(The CSV stores GLUPS to 5 significant digits; recomputing from `Time_s`
reproduces it to the last digit shown. This is the internal-consistency check
for the whole file.)

---

### 1.2 FLOP counting and compute performance

$$\text{GFLOP/s} = \text{GLUPS} \times \text{FLOP per cell}$$

**CPU backends — precomputed face diffusivities** (`compute_alpha()` builds
`alpha_x, alpha_y, alpha_z` once at init):

* 6 face fluxes, $F = \alpha_{\text{face}}\,(u_{\text{nbr}} - u_{\text{curr}})\,\Delta t/\Delta x^2$:
  $6 \times (1\ \text{sub} + 2\ \text{mul}) = 18$ FLOP
* combine and update: 5 add/sub + 1 add $= 6$ FLOP
* **24 FLOP/cell** (`FLOPS_PER_CELL 24.0`, `heat_3d.h`)

**CUDA backend — on-the-fly harmonic means:**

* $\alpha_{i+1/2} = \dfrac{2\alpha_i\alpha_{i+1}}{\alpha_i + \alpha_{i+1}}$:
  2 mul + 1 add + 1 div $= 4$ FLOP, six times $= 24$ FLOP
* plus the same 24 FLOP stencil
* **48 FLOP/cell** (`FLOPS_PER_CELL 48.0` under `__CUDACC__`)

Explicit at $N = 128$:

* $\text{GFLOPS}_{\text{seq}} = 0.02113 \times 24 = 0.5075$
* $\text{GFLOPS}_{\text{omp8}} = 0.07397 \times 24 = 1.7756$
* $\text{GFLOPS}_{\text{cuda256}} = 0.22070 \times 48 = 10.5946$

> **Warning.** FLOP/cell differs (24 vs 48), so GFLOPS is **not** a speedup
> metric across backends. Quoting the GFLOPS ratio 10.595 / 0.508 = 20.9×
> would double-count the algorithm change. The speedup is 10.09×, from time.
> A division is counted as 1 FLOP here; on Maxwell a `div.rn.f32` expands to
> 10–20 instructions, so the GPU's 48 FLOP/cell is an *undercount of work* and
> the quoted GPU GFLOPS is a lower bound. This is a known, unpriced item.

---

### 1.3 Memory traffic models and effective bandwidth

Course convention: **effective bandwidth counts compulsory traffic** — each
distinct location once per timestep, re-reads served by cache or shared memory
are by construction not traffic.

$$B_{\text{eff}} = \text{GLUPS} \times B_{\text{compulsory}}$$

| Backend | No reuse (as written) | Compulsory (perfect reuse) |
|---|---|---|
| CPU (seq, OpenMP) | 7 $u$ + 6 $\alpha$ + 1 write = 14 acc = **56 B** | 1 $u$ + 3 $\alpha$ + 1 write = 5 acc = **20 B** |
| CUDA | 7 $u$ + 7 $\alpha$ + 1 write = 15 acc = **60 B** | 1 $u$ + 1 $\alpha$ + 1 write = 3 acc = **12 B** |

Explicit at $N = 128$, against the 25.6 GB/s DRAM spec:

* $B_{\text{eff,seq}} = 0.02113 \times 20 = 0.4230$ GB/s $\Rightarrow$ **1.652 %**
* $B_{\text{eff,omp4}} = 0.07147 \times 20 = 1.4289$ GB/s $\Rightarrow$ **5.582 %**
* $B_{\text{eff,omp8}} = 0.07397 \times 20 = 1.4793$ GB/s $\Rightarrow$ **5.779 %**
* $B_{\text{eff,cuda256}} = 0.22070 \times 12 = 2.6484$ GB/s $\Rightarrow$ **10.345 %**

Equivalently, in the invariant unit:

$$\text{GLUPS}_{\max} = \frac{25.6}{B_{\text{compulsory}}}
= 1.280\ (\text{CPU}), \quad 2.1333\ (\text{GPU})$$

$$\frac{0.07397}{1.280} = 5.78\,\%, \qquad \frac{0.22070}{2.1333} = 10.35\,\%$$

Same numbers, one unit. Note this identity — it is the consistency check that
ties §1.3 to §1.5.

> **Caveat that must be stated whenever the CPU percentages are quoted.**
> 25.6 GB/s is the **CPU + GPU aggregate** LPDDR4 figure. What the four
> Cortex-A57 cores alone can sustain was never measured (a STREAM triad would
> give it). The CPU "% of peak" column therefore bounds utilisation from
> below and cannot support the claim that the CPU saturates DRAM.

#### Measured global traffic — `nvprof`, block 256

```
nvprof --metrics gld_throughput,gst_throughput,gld_efficiency,\
                 achieved_occupancy,sysmem_read_throughput,sysmem_write_throughput \
       ./heat_cube_cuda_3d --n 128 --steps 100 --block_size 256
```

`solve_stencil_cuda_kernel`, averaged over 100 invocations:

$$
\texttt{gld\_throughput} = 2.838\ \text{GB/s}, \qquad \texttt{gst\_throughput} = 0.903\ \text{GB/s}
$$

$$\text{load/store ratio} = \frac{2.838}{0.903} = 3.143$$

The kernel performs exactly **one** 4-byte store per cell, so

$$B_{\text{loads}} = 4 \times 3.143 = 12.57\ \text{B/cell}$$

$$B_{\text{measured}} = 4 + 12.57 = \mathbf{16.57\ \text{B/cell}}
= 4.14\ \text{accesses/cell}$$

$$\text{reuse efficiency} = \frac{B_{\text{compulsory}}}{B_{\text{measured}}}
= \frac{12.0}{16.57} = \mathbf{72.4\, \%}$$

Also measured: `gld_efficiency = 79.61 %`, `achieved_occupancy = 0.9852`,
`sysmem_read/write_throughput = 0` (zero PCIe traffic, as expected on a
unified-memory SoC).

---

### 1.4 CGMA and arithmetic intensity

$$\text{CGMA} = \frac{\text{FLOP}}{\text{global memory accesses}},
\qquad I = \frac{\text{FLOP}}{\text{bytes}} = \frac{\text{CGMA}}{4\ \text{B/access}}$$

| Variant | FLOP/cell | acc/cell | B/cell | CGMA | $I$ (FLOP/B) |
|---|:---:|:---:|:---:|:---:|:---:|
| CPU, as written (no reuse) | 24 | 14 | 56 | 1.714 | 0.429 |
| CPU, compulsory (tiled) | 24 | 5 | 20 | **4.800** | **1.200** |
| CUDA, as written (no reuse) | 48 | 15 | 60 | 3.200 | 0.800 |
| CUDA, compulsory (tiled) | 48 | 3 | 12 | **16.000** | **4.000** |
| CUDA, measured (nvprof, blk 256) | 48 | 4.14 | 16.57 | **11.594** | **2.897** |

The project's whole design argument is the movement down this table: 2.5D
tiling buys reuse, on-the-fly harmonic means buy bytes with arithmetic, and
together they take the realised CUDA point from CGMA 3.20 to 11.59.

---

### 1.5 Roofline and the Jetson Nano ridge point

$$P(I) = \min\!\left(P_{\text{peak}},\ I \times B_{\text{peak}}\right)$$

with $P_{\text{peak}} = 235.776$ GFLOP/s and $B_{\text{peak}} = 25.6$ GB/s.

$$I_{\text{ridge}} = \frac{235.776}{25.6} = \mathbf{9.210\ \text{FLOP/B}}
\qquad (\text{CGMA}_{\text{ridge}} = 36.84\ \text{FLOP/access})$$

Every operating point in this project has $I < I_{\text{ridge}}$, so all roofs
are the sloped memory roof $P_{\text{roof}} = I \times 25.6$.

| Point | $I$ (FLOP/B) | roof (GFLOP/s) | measured | % of roof |
|---|:---:|:---:|:---:|:---:|
| Sequential (compulsory) | 1.200 | 30.72 | 0.5075 | **1.652 %** |
| OpenMP 4 th (compulsory) | 1.200 | 30.72 | 1.7147 | **5.582 %** |
| OpenMP 8 th (compulsory) | 1.200 | 30.72 | 1.6999 | **5.533 %** |
| CUDA blk 256 (compulsory) | 4.000 | 102.40 | 10.5946 | **10.346 %** |
| CUDA blk 256 (measured traffic) | 2.897 | 74.16 | 10.5946 | **14.29 %** |

> **Consistency identity.** Under compulsory traffic,
> $$\frac{\text{GFLOPS}}{P_{\text{roof}}}
> = \frac{\text{GLUPS}\times F}{\left(F/B_c\right)\times 25.6}
> = \frac{\text{GLUPS}\times B_c}{25.6} = \frac{B_{\text{eff}}}{25.6}$$
> so the "% of roof" column above is numerically identical to the "% of peak
> bandwidth" column in §1.3. If they ever differ, one of the two tables has
> been edited without the other.

All points sit 10× to 58× below their own memory roof (58.5× sequential, 18.1× OpenMP 8, 9.7× CUDA). 
The roofline is useful here **in the negative**: it excludes the compute ceiling *and* the bandwidth
ceiling. The actual performance bottlenecks are: memory access latency, CPU scalar pipeline limitations,
and GPU barrier and launch overheads. 

---

### 1.6 Speedup, efficiency and Amdahl's law

$$S_{\text{seq}} = \frac{T_{\text{seq}}}{T(n)}, \quad
S_{\text{omp1}}(n) = \frac{T_{\text{omp}}(1)}{T_{\text{omp}}(n)}, \quad
E(n) = \frac{S_{\text{omp1}}(n)}{n}$$

$$S(n) = \frac{1}{(1-p) + p/n} \;\Longrightarrow\;
p = \frac{1 - 1/S(n)}{1 - 1/n}$$

Measured, $N = 128$: $T_{\text{seq}} = 9.9223$,
$T_{\text{omp}}(1,2,4,8,16) = 8.9536,\ 4.8226,\ 2.9353,\ 2.8356,\ 2.9119$ s.

| $n$ | $T$ (s) | $S_{\text{omp1}}$ | $S_{\text{seq}}$ | $E$ | implied $p$ |
|:---:|---:|---:|---:|---:|---:|
| 1 | 8.9536 | 1.0000 | 1.1082 | 1.000 | — |
| 2 | 4.8226 | 1.8566 | 2.0575 | 0.928 | 0.9227 |
| 4 | 2.9353 | 3.0503 | 3.3803 | 0.763 | 0.8962 |
| 8 | 2.8356 | 3.1576 | 3.4992 | 0.395 | 0.7809 |
| 16 | 2.9119 | 3.0748 | 3.4075 | 0.192 | 0.7198 |

**Separation of the two effects at $n = 4$:**

* tiling alone (cache reuse at one thread): $9.9223/8.9536 = \mathbf{1.1082\times}$
* worksharing alone: $8.9536/2.9353 = \mathbf{3.0503\times}$
* product: $1.1082 \times 3.0503 = \mathbf{3.3803\times}$ = $S_{\text{seq}}(4)$ ✓

**Amdahl fails as a model.** Fitting $p$ at $n = 2$
gives $p = 0.9227$, which predicts $S(4) = 3.247$, $S(8) = 5.192$,
$S(16) = 7.412$. Measured: 3.050, 3.158, 3.075. The fitted $p$ drifts
monotonically downward (0.923 → 0.896 → 0.781 → 0.720) — the signature of a
model that does not apply, not of a large serial section.

Note the $n = 4$ / $n = 8$ gap: 2.9353 ± 0.0168 against 2.8356 ± 0.0638 is
1.5 σ, so "OpenMP saturates at four threads" is the defensible statement and
"8 threads is fastest" is not.

Two candidate explanations, and the data separates them:

1. **$n \ge 8$ is saturation, not damage.** The Cortex-A57 cluster has 4
   physical cores with one hardware thread each. $n = 8$ and $n = 16$ are
   oversubscription; performance is flat, within scatter, from $n = 4$ on.
2. **$n = 4$ is where the real ceiling sits, and it is not L2.** The tile
   working set (§3.2) at four threads is 1.36 MiB against a 2 MiB shared L2.
   The falsifiable test was run: shrinking the tile from $64\times32\times16$
   (2.67 MiB at four threads) to $64\times16\times16$ (1.36 MiB) did **not**
   lift $S(4)$. **The L2-capacity hypothesis failed** (allowing all memory necessary
   tocfit overlapping tiling operations into CPU L2 cache will favour $n \le 4$).
   Nor is it DRAM saturation: four threads move 1.41 GB/s, 5.5 % of the (aggregate) spec.

The honest residual: the limiting factor at $n = 4$ is not identified.

---

### 1.7 Verification metrics

#### A. Method of Manufactured Solutions — spatial convergence

$$u_{\text{exact}}(x,y,z,t) = \sin\frac{\pi x}{L}\sin\frac{\pi y}{L}
\sin\frac{\pi z}{L} \text{ } \exp\left(-\frac{3\alpha\pi^2 t}{L^2}\right)$$

with $L = 1$, uniform $\alpha = 0.143$, Dirichlet $u = 0$ on all faces
(`src/heat_mms_seq_3d.c: exact_field()`). It satisfies
$\partial_t u = \alpha\nabla^2 u$ exactly, so every deviation is discretisation
error plus round-off.

$$L_2 = \sqrt{\frac{1}{N_xN_yN_z}\sum_{i,j,k}\big(u_{ijk} - u_{\text{exact}}(x_i,y_j,z_k)\big)^2}$$

$$\text{order} = \frac{\ln(L_2^{\text{coarse}}/L_2^{\text{fine}})}{\ln(h^{\text{coarse}}/h^{\text{fine}})},
\qquad h = \frac{L}{N+1}$$

From `data/mms_convergence_3d.csv`:

| $N$ | $h$ | $L_2$ (seq) | $L_2$ (omp) | $L_2$ (cuda) | order |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.058824 | 4.198606e-04 | 4.198606e-04 | 4.198567e-04 | — |
| 32 | 0.030303 | 1.059722e-04 | 1.059722e-04 | 1.059696e-04 | 2.0756 |
| 64 | 0.015385 | 2.669650e-05 | 2.669650e-05 | 2.669486e-05 | 2.0338 |
| 128 | 0.007752 | 6.784441e-06 | 6.784441e-06 | 6.783708e-06 | 1.9986 |

Worked, $16 \to 32$:

$$\text{order} = \frac{\ln(4.198606/1.059722)}{\ln(0.058824/0.030303)}
= \frac{1.37680}{0.66330} = 2.0756$$

$32 \to 64$: $1.37870/0.67788 = 2.0338$. $64 \to 128$:
$1.36995/0.68544 = 1.9986$. Second order confirmed as $h \to 0$.

Sequential and OpenMP agree to all 7 digits — no race condition. CUDA differs
in the 5th digit only, consistent with FP32 reassociation.

**What MMS does not test:** it runs at uniform $\alpha$, where a harmonic mean
of two equal values is trivially correct. The interface treatment is validated
only by B.

#### B. Discrete energy conservation — insulated heterogeneous domain

With zero-flux Neumann walls, $\langle u\rangle$ is a conserved quantity, and it should
remain such. Run: CUDA backend, $N = 64$, 500 s of physics,
170,400 steps, $\alpha$ ratio 776 (`data/avg_u_evo_3d.csv`).

$$\langle u\rangle(0) = 20.395508\ ^\circ\text{C}, \qquad
\langle u\rangle(499.718\ \text{s}) = 20.390408\ ^\circ\text{C}$$

$$\text{drift} = 5.100\times10^{-3}\ ^\circ\text{C}
= \frac{5.100\times10^{-3}}{20.395508} = 2.5006\times10^{-4}\ \text{relative}$$

$$\text{per step} = \frac{2.5006\times10^{-4}}{170{,}400} = 1.467\times10^{-9}$$

That is FP32 round-off, not scheme error. Cube average over the same run:
80.000000 → 26.239162 °C, still well above the 20.4 °C equilibrium, consistent
with the 1134 s water relaxation time (§1.8).

Because this run is the CUDA backend, it validates the **on-the-fly** harmonic
means specifically: the GPU rebuilds the face conductances from raw `alpha`
each step and still conserves to round-off.

---

### 1.8 The $\mathcal{O}(N^5)$ cost law

Von Neumann stability for explicit FTCS in 3D, with the **global** maximum
diffusivity:

$$\Delta t \le \frac{h^2}{6\alpha_{\max}}, \qquad
\Delta t = r\frac{h^2}{6\alpha_{\max}}, \quad r = 0.80, \quad
\alpha_{\max} = \alpha_{\text{Cu}} = 111.0\ \text{mm}^2/\text{s}$$

The solver uses $h = L/N$ (not $L/(N+1)$; that spacing is the MMS convention
only). Since $\Delta t \propto N^{-2}$, $N_{\text{steps}} \propto N^2$ and

$$\text{work} = N^3 \times N_{\text{steps}} \propto N^5$$

Inverted for a wall-clock budget $t_{\text{wall}}$:

$$T_{\text{phys}} = t_{\text{wall}}
\frac{r L^2\text{LUPS}}{6\alpha_{\max}} N^{-5}$$

With $t_{\text{wall}} = 3600$ s, $L = 100$ mm, $\text{LUPS} = 0.22070\times10^9$:

$$K = 3600 \times \frac{0.80 \times 100^2 \times 0.22070\times10^{9}}{6 \times 111.0}
= \mathbf{9.5438\times10^{12}}$$

| $N$ | $\Delta t$ (s) | physical seconds per **hour** of GPU time |
|---:|---:|---:|
| 64 | 2.9326e-03 | **8889** s (2.47 h of physics) |
| 128 | 7.3316e-04 | **277.8** s |
| 384 | 8.1462e-05 | **1.133** s |

($N = 384$ uses its own measured 0.21880 GLUPS.) The law is directly visible:
$64\to128$ is a factor 2 in resolution and a factor **32.0** ($\sim 2^5$) in reachable
physical time; $128\to384$ is a factor 3 and a factor **245.1** ($\sim 3^5$) — agreement to 0.1 %.

**Cost of the runs that matter:**

| target | $N$ | steps | GPU time |
|---|---:|---:|---:|
| 500 s of physics | 64 | 170,496 | **202.5 s** (3.4 min) |
| 500 s of physics | 128 | 681,984 | 6,480 s (1.80 h) |
| 500 s of physics | 384 | 6,137,856 | 1.588e6 s (18.4 days) |
| water relaxation $\tau_w = 1134$ s | 64 | 386,685 | **459.3 s** (7.7 min) |
| water relaxation $\tau_w = 1134$ s | 128 | 1,546,740 | 14,697 s (4.08 h) |
| water relaxation $\tau_w = 1134$ s | 384 | 13,920,657 | 3.602e6 s (41.7 days) |

This is what fixed the evolution run at $N = 64$, 500 s. The choice was
computed, not tried. It is also the clearest limiting factor in the project:
**the scheme, not the hardware**, is what puts a converged $384^3$ run out of
reach.

Relaxation times used above: $\tau = L^2/(\pi^2\alpha)$ gives
$\tau_{\text{Cu}} = 0.37$ s ($L = 20$ mm) and $\tau_w = 1134$ s ($L = 40$ mm),
a separation of 3105 — the stiffness that drives the whole cost.

---

## 2. Hardware architecture and derived constraints

### 2.1 Tegra X1 SoC (Jetson Nano)

```
   +---------------------------------------------------------------+
   |            NVIDIA Tegra X1 / T210  (Jetson Nano)              |
   +---------------------------------------------------------------+
   |     4 GB unified LPDDR4, 64-bit @ 1600 MHz  ->  25.6 GB/s     |
   |             (one pool, shared by CPU and GPU)                 |
   +-------------------------------+-------------------------------+
                                   |
                +------------------+------------------+
                |                                     |
    +-----------------------------+     +-----------------------------+
    |  4x ARM Cortex-A57          |     |  1 Maxwell SM               |
    |  4 cores, 1 thread/core     |     |  128 CUDA cores, warp 32    |
    |  32 KB L1D/core, 64 B line  |     |  4 warp schedulers          |
    |  2 MB shared L2             |     |  2048 max resident threads  |
    +-----------------------------+     |  64K x 32-bit registers     |
                                        |  64 KB shared mem / SM      |
                                        |  48 KB shared mem / block   |
                                        |  256 KB L2, 32 B line       |
                                        +-----------------------------+
```

**Peak FP32 (GPU):**

$$P_{\text{peak}} = 128\ \text{cores} \times 2\ \frac{\text{FLOP}}{\text{cycle}}
\ (\text{FMA}) \times 0.921\ \text{GHz} = \mathbf{235.776\ \text{GFLOP/s}}$$

`deviceQuery` reports a 900 MHz base clock, which would give 230.4 GFLOP/s.
The 2 % difference changes no conclusion; 235.776 (rounded to 236 in figures)
is used throughout for consistency with `fig/`.

**Peak bandwidth:**

$$B_{\text{peak}} = \frac{64\ \text{bit}}{8} \times 1600\ \text{MHz} \times 2\ (\text{DDR})
= 8\ \text{B} \times 3.2\times10^{9} = \mathbf{25.6\ \text{GB/s}}$$

**FP64:** Maxwell sm_53 issues FP64 at 1/32 of FP32:

$$P_{\text{peak,FP64}} = 235.776/32 = 7.368\ \text{GFLOP/s}$$

*Design rule:* FP32 everywhere in the hot loop. FP64 appears only in the
single-thread aggregation kernel (§3.7), where the 1/32 rate costs nothing and
the precision is needed to sum 2.1 M values without losing significance.

### 2.2 Latency hiding on a single SM

With **one** SM there is no spatial multiprocessor scaling: a larger grid
simply queues more blocks through the same hardware.

* global memory latency: 400–800 cycles; an FP operation: ~1 cycle
* 4 warp schedulers, zero-overhead switching between eligible warps
* residency: 2048 threads = 64 warps; occupancy = active warps / 64

Hiding a 400-cycle stall at CGMA ≈ 3–12 requires the scheduler to always find
an eligible warp. That is why occupancy (§3.6), not shared memory, was the
binding GPU constraint.

---

## 3. Memory hierarchy, tiling and implementation strategy

### 3.1 3D flattening and layout

$$\text{IDX3}(i,j,k) = (k \cdot N_y + j)\cdot N_x + i$$

* $i$ (X): contiguous, stride 1 float = 4 B
* $j$ (Y): stride $N_x$ floats
* $k$ (Z): stride $N_x N_y$ floats

One layout satisfies both architectures: a 64 B CPU cache line fetches 16
contiguous floats along $i$; a 32-thread GPU warp fetches 128 contiguous bytes
along $i$.

### 3.2 CPU cache blocking (OpenMP)

```c
#define OMP_TILE_X 64
#define OMP_TILE_Y 16
#define OMP_TILE_Z 16
```

Tile = $64\times16\times16 = 16{,}384$ cells. Working set per tile, counting the
exact index ranges the loop touches (`alpha_x` needs $i$ and $i-1$, so 65
values in X but only 16 in Y and Z — and symmetrically for `alpha_y`,
`alpha_z`):

| array | extent | bytes | KiB |
|---|---|---:|---:|
| $u$ read (1-cell halo) | $66\times18\times18$ | 85,536 | 83.53 |
| $\alpha_x$ read | $65\times16\times16$ | 66,560 | 65.00 |
| $\alpha_y$ read | $64\times17\times16$ | 69,632 | 68.00 |
| $\alpha_z$ read | $64\times16\times17$ | 69,632 | 68.00 |
| $u_{\text{next}}$ write | $64\times16\times16$ | 65,536 | 64.00 |
| **total** | | **356,896** | **348.53** |

| threads | aggregate | vs 2 MiB L2 |
|---:|---:|---|
| 1 | 348.5 KiB | 17.0 % |
| 4 | 1.361 MiB | 68.1 % — fits |
| 8 | 2.723 MiB | 136 % — does not fit, but $n = 8$ is oversubscription anyway |

The larger tile $64\times32\times16$ weighs 683.8 KiB, so four of them
(**2.671 MiB**) do not fit. Shrinking to $64\times16\times16$ was run as a
falsifiable test of the L2 hypothesis and **did not** improve $S(4)$ — see
§1.6.

**Directives.**

```c
#pragma omp parallel for collapse(3) schedule(static)
```

* `collapse(3)` merges the three tile loops: the outer loop alone has only
  $N_z/16 = 8$ iterations at $N = 128$, too coarse for 4 threads.
* `schedule(static)`: every cell costs the same, so the load is balanced by
  construction and dynamic scheduling would add overhead for nothing.
* No false sharing: `OMP_TILE_X = 64` floats = 256 B, an exact multiple of the
  64 B cache line, so tile boundaries in X fall on line boundaries.

### 3.3 CUDA coalescing

`threadIdx.x` is bound to the contiguous axis $i$:

$$i = \text{blockIdx.x}\cdot\text{blockDim.x} + \text{threadIdx.x}
\text{ }\Rightarrow\text{ } \text{addr} = \text{base} + 4\text{threadIdx.x}$$

so 32 lanes collapse into **one 128 B transaction**. Mapping `threadIdx.x` to
$k$ instead would issue ~32 separate transactions per warp.

Halo loads are the exception: when only `tx == 0` or `tx == tile_x - 1` is
active, one useful 4 B word rides a 32 B sector (12.5 %) or a 128 B
transaction (3.125 %). Minimising halo loads per interior cell is what block
geometry optimisation is for, and it is what pulls the measured
`gld_efficiency` to 79.61 %.

### 3.4 2.5D shared-memory tiling with a register sliding window

**Why not full 3D tiling.** A $32\times8\times8$ tile staging both fields would
need $2\times(34\times10\times10)\times4 = 27{,}200$ B = 26.56 KiB per block.
Against 64 KB of SM shared memory that admits $\lfloor 65536/27200 \rfloor = 2$
blocks = 512 threads = **25 % occupancy** — enough to be the binding
constraint, which is why the scheme is 2.5D instead.

**The scheme.** One 2D XY slice of **both** `u` and `alpha` is staged in shared
memory with halos; the Z direction is streamed through registers.

```
    pitch = tile_x + 2*RADIUS = 32 + 2 = 34 floats
    slice = pitch * (tile_y + 2*RADIUS)
    s_alpha = &s_mem[slice]              (one dynamic allocation, split)

    [halo Y]  +---------------------------------------+
              | h |        s_u  and  s_alpha      | h |
    [slice k] | a |       (threadIdx.x,           | a |
              | l |        threadIdx.y)           | l |
              | o |        32 x tile_y threads    | o |
    [halo Y]  +---------------------------------------+

      u_top  <- the only global read   (k+1)
      u_curr <- shifted from u_top     (k)
      u_bot  <- shifted from u_curr    (k-1)
```

Per $k$ step: shift the 6 registers ($u,\alpha \times$ top/curr/bot); read only
`u[i,j,k+1]` and `alpha[i,j,k+1]` from global; stage the slice and its X/Y
halos into shared; `__syncthreads()`; compute the 7-point stencil with X/Y
neighbours from shared and Z neighbours from registers; store
`u_next[i,j,k]`; `__syncthreads()`.

Z needs no halo and no barrier, because that reuse is register-local to one
thread rather than shared between threads.

$$\text{shared bytes/block} = 2 \times (\text{tile}_x + 2)(\text{tile}_y + 2)\times 4$$

$$
\text{model accesses/cell} = 2\left(1 + \frac{2}{\text{tile}_x} + \frac{2}{\text{tile}_y}\right) + 1
$$

the leading 2 because **both** fields are staged, the trailing 1 for the store.

### 3.5 Block geometry: the sweep and its model

`get_2d_tile_dims` fixes $\text{tile}_x = 32$ (warp width) and sets
$\text{tile}_y = B/32$.

| $B$ | tile | smem/block (B) | KiB | resident blocks @2048 th | total smem (KiB) | model acc/cell | **measured time (s)** |
|---:|---|---:|---:|---:|---:|---:|---:|
| 64 | 32×2 | 1,088 | 1.0625 | 32 | 34.00 | 5.125 | 1.0812 ± 0.0006 |
| 128 | 32×4 | 1,632 | 1.5938 | 16 | 25.50 | 4.125 | 0.9904 ± 0.0006 |
| **256** | **32×8** | **2,720** | **2.6562** | **8** | **21.25** | **3.625** | **0.9501 ± 0.0004** |
| 512 | 32×16 | 4,896 | 4.7812 | 4 | 19.13 | 3.375 | 0.9629 ± 0.0004 |
| 1024 | 32×32 | 9,248 | 9.0312 | 2 | 18.06 | 3.250 | 1.0506 ± 0.0003 |

**Optimum at $B = 256$**, spread across the sweep **13.8 %**. Two competing
effects, and the sweep shows both:

* **below 256** the Y-halo dominates: block 64 pays $2\times(2/2) = 2.00$ extra
  reads per cell against $2\times(2/8) = 0.50$ at block 256. The model
  accesses/cell column tracks the measured times monotonically here.
* **above 256** the halo keeps shrinking but the model stops predicting: 512
  and 1024 are *slower* despite lower modelled traffic, because resident
  blocks fall from 8 to 4 to 2, so each `__syncthreads()` barrier spans 16 and
  then 32 warps instead of 8, and the scheduler has fewer independent blocks
  to interleave.

No configuration exhausts the 64 KB shared budget — shared memory was never
the binding constraint. Note that with the earlier 24-FLOP kernel (one staged
array) this sweep was flat to 3 %; staging a *second* array is what made block
size a real parameter.

### 3.6 Register pressure and occupancy

Maxwell SM: 65,536 32-bit registers (256 KB).

$$\text{regs/thread for 100 \% occupancy} = \frac{65{,}536}{2048} = 32$$

| build | regs/thread | resident threads | theoretical occ. | measured |
|---|---:|---:|---:|---:|
| default `nvcc -O3 -arch=sm_53` | 64 | 1024 | 50.0 % | 0.4997 |
| `--maxrregcount=32` | 32 | 2048 | 100.0 % | 0.9852 (blk 256) |

`ptxas -v`, same source, only the flag differs:

```
  default          : Used 64 registers, 0 bytes spill stores, 0 bytes spill loads
  -maxrregcount=32 : Used 32 registers, 0 bytes spill stores, 0 bytes spill loads
```

**Zero spills in both builds.** The compiler used 64 registers because it
could, not because it needed to; the cap doubles resident warps at no memory
penalty. Both `Makefile` and `Makefile_jetson` carry the flag, so every
benchmark binary in the CSVs is a 32-register build.

Caveat: higher occupancy is not automatically faster — it helps only if the
kernel is latency-limited. `ptxas` isolates the cause and `achieved_occupancy`
the effect; the wall-clock benefit is not separately measured.

### 3.7 On-device hierarchical reduction

Domain averages $\langle u\rangle_{\text{total}}$ and $\langle u\rangle_{\text{cube}}$
every `stride` steps, without a synchronous host transfer inside the time loop.

1. `reduction_averages_cuda_kernel`: each thread writes its cell into dynamic
   shared memory (`s_cube = &s_mem[threads_per_block]`, one allocation split in
   two so any `blockDim` works), then a stride-halving tree reduction:

```c
for (int stride = threads_per_block / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        s_tot[tid]  += s_tot[tid + stride];
        s_cube[tid] += s_cube[tid + stride];
    }
    __syncthreads();
}
```

   Halving from $B/2$ downward keeps active threads contiguous in the low
   thread IDs, so whole warps retire rather than every warp running half-idle:
   no intra-warp divergence. Thread 0 writes one partial per block to
   `d_block_tot[]` / `d_block_cube[]`.

2. `aggregate_averages_cuda_kernel<<<1,256>>>`: sums the partials in **FP64**
   into the device evolution array. Summing 2.1 M FP32 values loses
   significance; with a single accumulating thread the 1/32 FP64 rate is free.

Nothing returns to the host until the run ends.

---

## 4. Master benchmark table

### 4.1 $N = 128^3$, 100 steps, 3 repeats (`data/benchmarks_reduced.csv`)

| Configuration | Time (s) | σ (s) | GLUPS | GFLOP/s | FLOP/cell | $B_{\text{eff}}$ (GB/s) | % of 25.6 | $S_{\text{seq}}$ | $S_{\text{omp1}}$ | $E$ |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| Sequential (untiled) | 9.9223 | 0.2741 | 0.02113 | 0.5075 | 24 | 0.4230 | 1.652 % | 1.000× | — | — |
| OpenMP 1 thread | 8.9536 | 0.0596 | 0.02343 | 0.5622 | 24 | 0.4685 | 1.830 % | 1.108× | 1.000× | 1.000 |
| OpenMP 2 threads | 4.8226 | 0.0868 | 0.04350 | 1.0439 | 24 | 0.8699 | 3.398 % | 2.057× | 1.857× | 0.928 |
| OpenMP 4 threads | 2.9353 | 0.0168 | 0.07147 | 1.7147 | 24 | 1.4289 | 5.582 % | 3.380× | 3.050× | 0.763 |
| **OpenMP 8 threads** | **2.8356** | **0.0638** | **0.07397** | **1.7756** | 24 | **1.4797** | **5.780 %** | **3.499×** | **3.158×** | 0.395 |
| OpenMP 16 threads | 2.9119 | 0.0118 | 0.07200 | 1.7285 | 24 | 1.4404 | 5.627 % | 3.408× | 3.075× | 0.192 |
| CUDA block 64 (32×2) | 1.0812 | 0.0006 | 0.19393 | 9.3104 | 48 | 2.3276 | 9.092 % | 9.177× | — | — |
| CUDA block 128 (32×4) | 0.9904 | 0.0006 | 0.21173 | 10.1638 | 48 | 2.5409 | 9.926 % | 10.018× | — | — |
| **CUDA block 256 (32×8)** | **0.9501** | **0.0004** | **0.22070** | **10.5946** | 48 | **2.6487** | **10.346 %** | **10.443×** | — | — |
| CUDA block 512 (32×16) | 0.9629 | 0.0004 | 0.21777 | 10.4540 | 48 | 2.6135 | 10.209 % | 10.304× | — | — |
| CUDA block 1024 (32×32) | 1.0506 | 0.0003 | 0.19963 | 9.5817 | 48 | 2.3954 | 9.357 % | 9.444× | — | — |

Measured DRAM traffic for the block-256 kernel (nvprof, §1.3): 16.57 B/cell
= 3.658 GB/s = 14.29 % of spec.

### 4.2 $N = 384^3$, 100 steps (`data/benchmarks_large.csv`)

| Configuration | Time (s) | σ (s) | GLUPS | GFLOP/s | $B_{\text{eff}}$ (GB/s) | % of 25.6 | $S_{\text{seq}}$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sequential (untiled) | 290.022 | 0.405 | 0.01950 | 0.4686 | 0.3900 | 1.525 % | 1.000× |
| OpenMP 8 threads | 77.751 | 0.108 | 0.07280 | 1.7478 | 1.4565 | 5.690 % | 3.730× |
| CUDA block 256 | 25.878 | 0.004 | 0.21880 | 10.5027 | 2.6257 | 10.257 % | 11.207× |

**Scaling from $128^3$ to $384^3$ (27× the cells):**

| backend | $S_{\text{seq}}$ @128 | $S_{\text{seq}}$ @384 | change |
|---|---:|---:|---:|
| OpenMP 8 th | 3.499× | 3.730× | **+6.6 %** |
| CUDA blk 256 | 10.443× | 11.207× | **+7.3 %** |

**CUDA throughput is flat to 0.86 %** (0.22070 → 0.21880 GLUPS) across a 27×
increase in cell count. That is the strong result: the 2.5D scheme reuses data
*inside* a block, in shared memory and registers, and that reuse is
independent of the total working-set size. Only the residual cross-block reuse
through the 256 KB L2 degrades, and it was never carrying much.

Both speedup ratios improve, but the **denominator moves**.
Every backend loses throughput at 56.6 M cells, and the serial baseline loses
by far the most:

| backend | GLUPS @128 | GLUPS @384 | change |
|---|---:|---:|---:|
| Sequential | 0.02113 | 0.01950 | **−7.73 %** |
| OpenMP 8 th | 0.07397 | 0.07280 | −1.58 % |
| CUDA blk 256 | 0.22070 | 0.21880 | −0.86 % |

So the +6.6 % and -7.3 % are statements about the serial baseline degrading
9× faster than CUDA. The untiled triple loop is the only implementation 
whose reuse depends on the total working-set size. The GLUPS column is flat (mildly negative).

---

## 5. Summary of architectural findings

1. **GLUPS is the only invariant benchmark.** The GPU restructuring (48
   FLOP/cell against 24) makes GFLOPS ratios meaningless as speedup. Time and
   GLUPS are the comparison; GFLOPS and bandwidth are each read against their
   own backend's roof.

2. **Arithmetic bought bandwidth, in the right direction only.** Rebuilding
   the six harmonic means in-kernel costs 24 extra FLOP/cell and saves two
   $\alpha$ arrays: compulsory traffic 20 → 12 B/cell, intensity
   $1.20 \to 4.00$ FLOP/B. On a machine with $I_{\text{ridge}} = 9.21$ and
   idle arithmetic units during memory stalls, that moves the operating point
   *toward* the ridge (a factor 7.7 below it, now a factor 2.3). The same
   trade on the Cortex-A57 — limited scalar ALUs, expensive division — would
   lose, which is why the CPU keeps the precomputed arrays. Net result:
   **10.443× over the sequential baseline at $N = 128$, 11.207× at $N = 384$.**

3. **The register file was the binding GPU constraint, not shared memory.**
   Shared use peaks at 2.66 KiB of 48 KiB per block. The default build spent
   64 registers/thread for 50 % occupancy; `--maxrregcount=32` reaches 98.5 %
   achieved occupancy with **zero spills in both builds**.

4. **Neither engine is bandwidth-bound.** Best CPU 5.5 %, best GPU 10.3 % of
   the 25.6 GB/s spec on compulsory traffic (14.3 % on measured traffic). Both
   sit far below their own memory roof, so the limiter is latency and warp
   residency on the GPU, and — on the CPU — something not yet identified.

5. **Three hypotheses held before measuring were wrong.**
   * The CPU was expected to saturate the memory controller at $n = 4$. It
     reaches 5.5 % of spec.
   * L2 oversubscription was blamed for the OpenMP plateau. Shrinking the tile
     from 2.67 MiB to 1.36 MiB at four threads changed nothing — refuted.
   * Shared memory was expected to cap GPU residency. It was the register file.

6. **The binding limit overall is the scheme, not the chip.** $\mathcal{O}(N^5)$
   puts a converged $384^3$ run at 41.7 days of GPU time for one water
   relaxation time. No amount of tuning on this hardware closes that.

**Open items, stated rather than hidden:**

* STREAM triad for the CPU-only bandwidth ceiling — the largest remaining gap,
  since 25.6 GB/s is a CPU+GPU aggregate.
* Price the six FP32 divisions (`__fdividef`, `-use_fast_math`); `div.rn.f32`
  is 10–20 instructions, not the 1 FLOP counted, so 48 FLOP/cell undercounts.
* Re-profile block 512 and 1024 to confirm the barrier-width explanation in
  §3.5, which is currently inference from the model/measurement divergence.
* $N = 512^3$ on the GPU alone (1.61 GB with three arrays instead of six).

---
