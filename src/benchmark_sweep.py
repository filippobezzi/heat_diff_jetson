#!/usr/bin/env python3
"""
Performance benchmark sweep for the 3D heat solver.

Reports TWO baselines:

  Speedup_vs_seq   -- against the untiled sequential C solver. Measures the
                      whole optimisation given: cache blocking AND threading.
  Speedup_vs_omp1  -- against the OpenMP solver run on one thread w/out tiling.
                      Measures threading alone.

Timing statistics: mean, standard deviation and minimum over the repeats.
"""

import os
import re
import subprocess

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

JETSON_PEAK_BW = 25.6      # GB/s, LPDDR4 spec

OMP_THREADS = [1, 2, 4, 8, 16]
CUDA_BLOCKS = [64, 128, 256, 512, 1024]


def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, cwd=SRC_DIR, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    return res.stdout


def build_all():
    print("[Build] Compiling simulation targets...")
    code_cuda = subprocess.run("which nvcc || test -x /usr/local/cuda/bin/nvcc", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    targets = ["heat_cube_seq_3d", "heat_cube_omp_3d"]
    if code_cuda == 0:
        targets.append("heat_cube_cuda_3d")
    run_cmd(f"make {' '.join(targets)}")


def parse_metrics(out):
    """Read the metrics that print_performance() emits."""
    t = re.search(r"Solve Wall-Clock Time\s*:\s*([0-9.]+)", out)
    glups = re.search(r"Throughput \(GLUPS\)\s*:\s*([0-9.]+)", out)
    gflops = re.search(r"Compute Performance\s*:\s*([0-9.]+)", out)
    bw = re.search(r"Effective (?:Memory )?Bandwidth\s*:\s*([0-9.]+)\s*GB/s", out)
    return (float(t.group(1)) if t else None,
            float(glups.group(1)) if glups else None,
            float(gflops.group(1)) if gflops else None,
            float(bw.group(1)) if bw else None)


def benchmark(binary, n, steps, extra_args="", repeats=3):
    """Run one configuration `repeats` times; return timing statistics."""
    times, glups_l, gflops_l, bw_l = [], [], [], []
    for _ in range(repeats):
        out = run_cmd(f"./{binary} --n {n} --steps {steps} {extra_args} --output /dev/null")
        t, g, f, b = parse_metrics(out)
        if t is None or g is None or f is None or b is None:
            raise RuntimeError(f"Could not parse output of ./{binary} {extra_args}:\n{out}")
        times.append(t); glups_l.append(g); gflops_l.append(f); bw_l.append(b)

    return {
        "Time_s": float(np.mean(times)),
        "Time_std_s": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
        "Time_min_s": float(np.min(times)),
        "GLUPS": float(np.mean(glups_l)),
        "GFLOPS": float(np.mean(gflops_l)),
        "Bandwidth_GBs": float(np.mean(bw_l)),
    }


def record(backend, config, param, n, stats, t_seq, t_omp1):
    r = dict(stats)
    r.update({
        "Backend": backend,
        "Config": config,
        "Threads_or_BlockSize": param,
        "N": n,
        "Speedup": t_seq / r["Time_s"],
        "Speedup_vs_seq": t_seq / r["Time_s"],
        "Speedup_vs_omp1": (t_omp1 / r["Time_s"]) if t_omp1 else np.nan,
        "Pct_Peak_BW": 100.0 * r["Bandwidth_GBs"] / JETSON_PEAK_BW,
    })
    return r


def show(label, r):
    print(f"{label:<24} {r['Time_s']:>8.4f} +/- {r['Time_std_s']:<7.4f} "
          f"(min {r['Time_min_s']:>7.4f})  "
          f"S_seq={r['Speedup']:>6.2f}x  S_omp1={r['Speedup_vs_omp1']:>6.2f}x  "
          f"{r['GFLOPS']:>6.2f} GF  BW {r['Bandwidth_GBs']:>5.2f} GB/s")


def run_sweep(n=128, steps=100, repeats=3):
    print(f"\n--- Parameter Sweep (N={n}^3, {steps} steps, {repeats} repeats) ---")
    print("Bandwidth is reported as a [perfect-reuse, no-reuse] bound, identically")
    print("for every backend. The true DRAM traffic needs a profiler to settle.\n")

    has_cuda = os.path.exists(os.path.join(SRC_DIR, "heat_cube_cuda_3d"))
    records = []

    seq = benchmark("heat_cube_seq_3d", n, steps, repeats=repeats)
    t_seq = seq["Time_s"]

    # 1-thread OpenMP: the tiled baseline for parallel scaling.
    omp1 = benchmark("heat_cube_omp_3d", n, steps, "--threads 1", repeats)
    t_omp1 = omp1["Time_s"]

    r = record("Sequential", "1 thread (untiled)", 1, n, seq, t_seq, t_omp1)
    records.append(r); show("Sequential (untiled)", r)

    for th in OMP_THREADS:
        stats = omp1 if th == 1 else benchmark("heat_cube_omp_3d", n, steps,
                                               f"--threads {th}", repeats)
        r = record("OpenMP CPU", f"{th} threads", th, n, stats, t_seq, t_omp1)
        records.append(r); show(f"OpenMP ({th} threads)", r)

    if has_cuda:
        for b in CUDA_BLOCKS:
            stats = benchmark("heat_cube_cuda_3d", n, steps, f"--block_size {b}", repeats)
            r = record("CUDA GPU", f"block={b}", b, n, stats, t_seq, t_omp1)
            records.append(r); show(f"CUDA (block={b})", r)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(DATA_DIR, "benchmarks_reduced.csv"), index=False)

    omp = df[df["Backend"] == "OpenMP CPU"]
    cuda = df[df["Backend"] == "CUDA GPU"]
    best_omp = int(omp.loc[omp["Time_s"].idxmin(), "Threads_or_BlockSize"]) if not omp.empty else 4
    best_cuda = int(cuda.loc[cuda["Time_s"].idxmin(), "Threads_or_BlockSize"]) if not cuda.empty else 256

    # Warn when the "optimum" is inside the run-to-run noise --
    for name, sub in (("OpenMP", omp), ("CUDA", cuda)):
        if not sub.empty:
            span = sub["Time_s"].max() - sub["Time_s"].min()
            if span < 2.0 * sub["Time_std_s"].max():
                print(f"[warn] {name} sweep spans {span:.4f} s, within 2 sigma of the "
                      f"run-to-run spread ({sub['Time_std_s'].max():.4f} s). "
                      "The selected optimum is not statistically distinguishable.")

    print(f"\n>> Selected: OpenMP threads = {best_omp} | CUDA block = {best_cuda}")
    print("")
    return df, best_omp, best_cuda


def run_large(n=384, steps=100, opt_omp=4, opt_cuda=512, repeats=3):
    total = n ** 3
    print(f"\n--- Large Grid (N={n}^3 = {total:.3e} cells, "
          f"{6*total*4/1e9:.3g} GB, {steps} steps, {repeats} repeats) ---")
    has_cuda = os.path.exists(os.path.join(SRC_DIR, "heat_cube_cuda_3d"))
    records = []

    seq = benchmark("heat_cube_seq_3d", n, steps, repeats=repeats)
    t_seq = seq["Time_s"]
    omp1 = benchmark("heat_cube_omp_3d", n, steps, "--threads 1", repeats)
    t_omp1 = omp1["Time_s"]

    r = record("Sequential", "1 thread (untiled)", 1, n, seq, t_seq, t_omp1)
    records.append(r); show("Sequential (untiled)", r)

    stats = benchmark("heat_cube_omp_3d", n, steps, f"--threads {opt_omp}", repeats)
    r = record("OpenMP CPU", f"{opt_omp} threads", opt_omp, n, stats, t_seq, t_omp1)
    records.append(r); show(f"OpenMP ({opt_omp} threads)", r)

    if has_cuda:
        stats = benchmark("heat_cube_cuda_3d", n, steps, f"--block_size {opt_cuda}", repeats)
        r = record("CUDA GPU", f"block={opt_cuda}", opt_cuda, n, stats, t_seq, t_omp1)
        records.append(r); show(f"CUDA (block={opt_cuda})", r)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(DATA_DIR, "benchmarks_large.csv"), index=False)
    print("")
    return df


def main():
    build_all()
    _, best_omp, best_cuda = run_sweep(n=128, steps=100, repeats=3)
    run_large(n=384, steps=100, opt_omp=best_omp, opt_cuda=best_cuda, repeats=3)


if __name__ == "__main__":
    main()
