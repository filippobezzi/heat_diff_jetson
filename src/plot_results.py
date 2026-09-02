#!/usr/bin/env python3
"""
Visualization for 3D Heat Diffusion Benchmarks & Verification.
Outputs presentation-ready, title-free figures into fig/:
  - temperature_evolution.png : Cube vs Total domain average temperature field evolution
  - parameter_sweep_times.png : 2 panels (OpenMP execution time vs threads, CUDA execution time vs block size)
  - speedup_ideal.png         : 2 panels (OpenMP scaling vs ideal linear, CUDA scaling (block size) vs BW limit)
  - speedup_grid_sizes.png    : Speedup comparison across grid sizes N=128^3 and N=384^3)
  - mms_test.png              : Grid convergence (L2 error vs h) for MMS verification
  - effective_bandwidth.png   : Effective memory bandwidth
  - roofline_model.png        : Jetson Nano Roofline model
"""

import os
import sys
import re
import math
import argparse
import subprocess
import shlex
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Directory paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIG_DIR = os.path.join(PROJECT_DIR, "fig")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Presentation plot styling
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'lines.linewidth': 2.0
})

def run_cmd(cmd, cwd=SRC_DIR):
    res = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout, res.stderr

# Ensure temperature evolution dataset exists
def evolution_dataset(csv_path, n=64, time_scale=500.0, stride=200):
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "time" in df.columns and df["time"].max() >= (time_scale * 0.8):
                return
        except Exception:
            pass

    # Remove stale/short dataset if it doesn't reach the required time
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except OSError:
            pass

    out_arg = shlex.quote(csv_path)
    # Attempt CUDA -> OpenMP -> Sequential
    run_cmd("make heat_cube_cuda_3d")
    if os.path.exists(os.path.join(SRC_DIR, "heat_cube_cuda_3d")):
        code, _, _ = run_cmd(f"./heat_cube_cuda_3d --n {n} --time {time_scale} --stride {stride} --block_size 256 --output {out_arg}")
        if code == 0 and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if df["time"].max() >= (time_scale * 0.8):
                    return
            except Exception:
                pass

    run_cmd("make heat_cube_omp_3d")
    if os.path.exists(os.path.join(SRC_DIR, "heat_cube_omp_3d")):
        code, _, _ = run_cmd(f"./heat_cube_omp_3d --n {n} --time {time_scale} --stride {stride} --threads 4 --output {out_arg}")
        if code == 0 and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if df["time"].max() >= (time_scale * 0.8):
                    return
            except Exception:
                pass

    run_cmd("make heat_cube_seq_3d")
    run_cmd(f"./heat_cube_seq_3d --n {n} --time {time_scale} --stride {stride} --output {out_arg}")

# 1. Temperature Evolution (Single Panel)
def plot_time_evolution(evo_csv_path=None, output_fig_path=None, n=64, time_scale=500.0, stride=200):
    if evo_csv_path is None:
        evo_csv_path = os.path.join(DATA_DIR, "avg_u_evo_3d.csv")
    if output_fig_path is None:
        output_fig_path = os.path.join(FIG_DIR, "temperature_evolution.png")

    evolution_dataset(evo_csv_path, n=n, time_scale=time_scale, stride=stride)
    if not os.path.exists(evo_csv_path):
        print(f"[Warning] File not found: {evo_csv_path}")
        return

    df = pd.read_csv(evo_csv_path)
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)

    ax.plot(df["time"], df["u_avg_cube"], label=r"Copper Cube $\langle u \rangle_{\mathrm{cube}}(t)$",
            color="#D9381E", linewidth=2.5)
    ax.plot(df["time"], df["u_avg_total"], label=r"Total Domain $\langle u \rangle_{\mathrm{total}}(t)$",
            color="#1F77B4", linewidth=2.2, linestyle="--")

    ax.set_xlabel("Physical Time (s)", fontweight="bold")
    ax.set_ylabel(r"Temperature ($^\circ$C)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.92)
    ax.set_ylim(min(df["u_avg_total"].min() - 2, 18), 85)

    plt.tight_layout()
    plt.savefig(output_fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_fig_path}")


# 2. MMS Grid Convergence Plot
def plot_mms_test(output_fig_path=None):
    """Plot L2 error vs grid spacing. Runs solvers if data is missing."""
    import math as _math

    mms_conv_csv = os.path.join(DATA_DIR, "mms_convergence_3d.csv")
    if output_fig_path is None:
        output_fig_path = os.path.join(FIG_DIR, "mms_test.png")

    # Generate data if CSV is missing
    if not os.path.exists(mms_conv_csv):
        print("[MMS] Convergence CSV not found -- running solvers...")
        L, ALPHA, R = 1.0, 0.143, 0.80 / 6.0
        TAU = L**2 / (3.0 * _math.pi**2 * ALPHA)
        T_END = 2 * TAU
        GRIDS = [16, 32, 64, 128]

        run_cmd("make heat_mms_seq_3d heat_mms_omp_3d")
        has_cuda = os.path.exists(os.path.join(SRC_DIR, "heat_mms_cuda_3d")) or \
                   run_cmd("make heat_mms_cuda_3d")[0] == 0

        rows = []
        for n in GRIDS:
            h = L / (n + 1)
            dt = R * h**2 / ALPHA
            steps = max(1, round(T_END / dt))

            def get_l2(stdout):
                m = re.search(r"L2-Norm Error\s*:\s*([0-9.eE+-]+)", stdout)
                return float(m.group(1)) if m else None

            _, out_s, _ = run_cmd(f"./heat_mms_seq_3d --n {n} --steps {steps}")
            _, out_o, _ = run_cmd(f"./heat_mms_omp_3d --n {n} --steps {steps} --threads 4")
            l2_cuda = None
            if has_cuda:
                _, out_c, _ = run_cmd(f"./heat_mms_cuda_3d --n {n} --steps {steps} --block_size 256")
                l2_cuda = get_l2(out_c)

            rows.append({"N": n, "h": h, "L2_seq": get_l2(out_s),
                         "L2_omp": get_l2(out_o), "L2_cuda": l2_cuda})

        df_conv = pd.DataFrame(rows)
        df_conv.to_csv(mms_conv_csv, index=False)
        print(f"[MMS] Saved: {mms_conv_csv}")
    else:
        df_conv = pd.read_csv(mms_conv_csv)

    h = df_conv["h"].values
    l2_seq = df_conv["L2_seq"].values

    log_h = np.log(h)
    log_l2 = np.log(l2_seq)
    slope_seq, intercept = np.polyfit(log_h, log_l2, 1)

    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)

    ax.loglog(h, l2_seq, 'o-', color='#D9381E', markersize=8, linewidth=2.2,
              label=f"Sequential CPU (Slope = {slope_seq:.2f})")

    if "L2_omp" in df_conv.columns and df_conv["L2_omp"].notnull().all():
        ax.loglog(h, df_conv["L2_omp"].values, 's--', color='#1F77B4', markersize=7, linewidth=1.8,
                  label="OpenMP CPU (4 threads)")

    if "L2_cuda" in df_conv.columns and df_conv["L2_cuda"].notnull().all():
        ax.loglog(h, df_conv["L2_cuda"].values, 'D-.', color='#2CA02C', markersize=7, linewidth=1.8,
                  label="CUDA GPU (256 th/blk)")

    ref_line = l2_seq[0] * (h / h[0]) ** 2
    ax.loglog(h, ref_line, ':', color='gray', linewidth=2.0, alpha=0.85,
              label=r"Theoretical $\mathcal{O}(h^2)$ Slope")

    for idx, (x, y, n_val) in enumerate(zip(h, l2_seq, df_conv["N"].values)):
        xy_off = (-10, 10) if idx != len(h) - 1 else (-10, -18)
        ax.annotate(f"$N={n_val}$\n{y:.1e}", (x, y), textcoords="offset points",
                    xytext=xy_off, ha='right', fontsize=9, fontweight='bold', color='#D9381E')

    ax.set_xlabel(r"Grid Spacing $h = L/(N+1)$", fontweight="bold")
    ax.set_ylabel(r"$L_2$ Error Norm $\|u - u_{\mathrm{exact}}\|_2$", fontweight="bold")
    ax.set_xlim(min(h) * 0.70, max(h) * 1.35)
    ax.set_ylim(min(l2_seq) * 0.35, max(l2_seq) * 3.5)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_fig_path}")

# 3. Execution Times Parameter Sweep
def plot_execution_times(reduced_csv=None, output_fig_path=None):
    if reduced_csv is None:
        reduced_csv = os.path.join(DATA_DIR, "benchmarks_reduced.csv")
    if output_fig_path is None:
        output_fig_path = os.path.join(FIG_DIR, "parameter_sweep_times.png")

    if not os.path.exists(reduced_csv):
        print(f"[Warning] File not found: {reduced_csv}")
        return

    df_red = pd.read_csv(reduced_csv)
    seq_row = df_red[df_red["Backend"] == "Sequential"].iloc[0]
    t_seq_red = seq_row["Time_s"]

    omp_df = df_red[df_red["Backend"] == "OpenMP CPU"].sort_values("Threads_or_BlockSize")
    cuda_df = df_red[df_red["Backend"] == "CUDA GPU"].sort_values("Threads_or_BlockSize")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)

    if cuda_df.empty:
        print("[Warning] No CUDA rows in the benchmark CSV; skipping "
              f"{os.path.basename(output_fig_path)}. Run benchmark_sweep.py on a "
              "CUDA-capable machine to produce it.")
        plt.close(fig)
        return

    # Left: OpenMP Execution Time vs Threads
    threads = omp_df["Threads_or_BlockSize"].values
    omp_times = omp_df["Time_s"].values

    ax1.plot(threads, omp_times, 'o-', color='#1F77B4', markersize=8, linewidth=2.2, label="OpenMP CPU")
    ax1.axhline(y=t_seq_red, color='#D9381E', linestyle='--', linewidth=1.8, label=f"Sequential ({t_seq_red:.3f} s)")

    for idx, (x, y) in enumerate(zip(threads, omp_times)):
        if idx == 0:  # P=1 (to the left)
            ha, x_off, y_off = 'right', -8, 8
        elif idx == len(threads) - 1:  # P=8
            ha, x_off, y_off = 'center', 0, 10
        else:  # P=2, P=4
            ha, x_off, y_off = 'center', 0, 10
        ax1.annotate(f"{y:.3f} s", (x, y), textcoords="offset points", xytext=(x_off, y_off),
                     ha=ha, fontweight='bold', color='#1F77B4')

    ax1.set_xlim(0.0, 8.8)
    ax1.set_ylim(0, max(t_seq_red, max(omp_times)) * 1.35)
    ax1.set_xlabel("OpenMP Threads ($P$)", fontweight="bold")
    ax1.set_ylabel("Execution Time (s)", fontweight="bold")
    ax1.set_xticks(threads)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.92)

    # Right: CUDA Execution Time vs Block Size
    block_sizes = cuda_df["Threads_or_BlockSize"].values
    cuda_times = cuda_df["Time_s"].values
    x_indices = np.arange(len(block_sizes))

    ax2.plot(x_indices, cuda_times, 's-', color='#2CA02C', markersize=8, linewidth=2.2, label="CUDA GPU")
    ax2.axhline(y=t_seq_red, color='#D9381E', linestyle='--', linewidth=1.8) #, label=f"Sequential ({t_seq_red:.3f} s")

    for idx, (x, y) in enumerate(zip(x_indices, cuda_times)):
        ha = 'left' if idx == 0 else ('right' if idx == len(x_indices) - 1 else 'center')
        x_off = 6 if idx == 0 else (-6 if idx == len(x_indices) - 1 else 0)
        ax2.annotate(f"{y:.3f} s", (x, y), textcoords="offset points", xytext=(x_off, 10),
                     ha=ha, fontweight='bold', color='#2CA02C')

    ax2.set_xlim(-0.4, len(block_sizes) - 0.6)
    ax2.set_ylim(0, max(t_seq_red, max(cuda_times)) * 1.35)
    ax2.set_xlabel("Threads per Block", fontweight="bold")
    # ax2.set_ylabel("Execution Time (s)", fontweight="bold")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([str(b) for b in block_sizes])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.92)

    plt.tight_layout()
    plt.savefig(output_fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_fig_path}")

# 4A. Parallel Speedup Parameter Sweeps
def plot_speedup_ideal(reduced_csv=None, output_fig_path=None):
    if reduced_csv is None:
        reduced_csv = os.path.join(DATA_DIR, "benchmarks_reduced.csv")
    if output_fig_path is None:
        output_fig_path = os.path.join(FIG_DIR, "speedup_ideal.png")

    if not os.path.exists(reduced_csv):
        print(f"[Warning] File not found: {reduced_csv}")
        return

    df_red = pd.read_csv(reduced_csv)
    omp_df = df_red[df_red["Backend"] == "OpenMP CPU"].sort_values("Threads_or_BlockSize")
    cuda_df = df_red[df_red["Backend"] == "CUDA GPU"].sort_values("Threads_or_BlockSize")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)

    if cuda_df.empty:
        print("[Warning] No CUDA rows in the benchmark CSV; skipping "
              f"{os.path.basename(output_fig_path)}. Run benchmark_sweep.py on a "
              "CUDA-capable machine to produce it.")
        plt.close(fig)
        return

    # Left: OpenMP Speedup vs Threads
    threads = omp_df["Threads_or_BlockSize"].values
    speedup_omp = omp_df["Speedup_vs_seq"].values
    speedup_omp1 = omp_df["Speedup_vs_omp1"].values

    # Two baselines: vs the untiled sequential solver (tiling + threading) and
    # vs 1-thread OpenMP (threading alone).  Only the second is parallel scaling.
    ax1.plot(threads, speedup_omp1, 'o-', color='#1F77B4', markersize=8, linewidth=2.5,
             label=r"Threading only ($T_{\mathrm{omp}(1)} / T_{\mathrm{omp}(P)}$)")
    ax1.plot(threads, speedup_omp, 's--', color='#8C564B', markersize=7, linewidth=2.0,
             alpha=0.9, label=r"Tiling + threading ($T_{\mathrm{seq}} / T_{\mathrm{omp}(P)}$)")
    ax1.plot(threads, threads, '--', color='gray', alpha=0.7, label=r"Ideal Linear Scaling ($S=P$)")
    ax1.axhline(y=1.0, color='#D9381E', linestyle=':', label="Baseline ($S=1.0$)")

    for idx, (x, y) in enumerate(zip(threads, speedup_omp1)):
        x_off, y_off, va = (0, -14, 'top') if idx > 0 else (-6, -12, 'top')
        ha = 'center' if idx > 0 else 'right'
        ax1.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(x_off, y_off),
                     ha=ha, va=va, fontweight='bold', color='#1F77B4')

    ax1.set_xlim(0.0, 8.8)
    ax1.set_xlabel("OpenMP Threads ($P$)", fontweight="bold")
    ax1.set_ylabel("Speedup Factor", fontweight="bold")
    ax1.set_xticks(threads)
    ax1.set_ylim(0, max(max(threads), np.nanmax(speedup_omp)) * 1.18)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.92)

    # Right: CUDA Speedup vs Block Size
    block_sizes = cuda_df["Threads_or_BlockSize"].values
    speedup_cuda = cuda_df["Speedup_vs_seq"].values
    x_indices = np.arange(len(block_sizes))

    ax2.plot(x_indices, speedup_cuda, 's-', color='#2CA02C', markersize=8, linewidth=2.5,
             label=r"Measured Speedup $S_{\mathrm{cuda}}$")
    ax2.axhline(y=1.0, color='#D9381E', linestyle=':', label="Sequential Baseline ($S=1.0$)")
    # Ceiling = how much faster the sequential run could get if it saturated peak
    # DRAM bandwidth.
    seq_df = df_red[df_red["Backend"] == "Sequential"]
    seq_bw = seq_df["Bandwidth_GBs"].values[0] if (not seq_df.empty and "Bandwidth_GBs" in seq_df.columns) else (seq_df["BW_max_GBs"].values[0] if not seq_df.empty and "BW_max_GBs" in seq_df.columns else 1.0)
    bw_limit = 25.6 / seq_bw if seq_bw > 0 else 25.0
    ax2.axhline(y=bw_limit, color='gray', linestyle='--', alpha=0.8,
                label=rf"BW Peak Ceiling ($S \approx {bw_limit:.1f}\times$)")

    for idx, (x, y) in enumerate(zip(x_indices, speedup_cuda)):
        x_off = 10 if idx == 0 else (-6 if idx == len(x_indices) - 1 else 0)
        ha = 'left' if idx == 0 else ('right' if idx == len(x_indices) - 1 else 'center')
        ax2.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(x_off, 10),
                     ha=ha, fontweight='bold', color='#2CA02C')

    ax2.set_xlabel("Threads per Block", fontweight="bold")
    ax2.set_ylabel(r"Speedup vs Sequential ($T_{\mathrm{seq}} / T_{\mathrm{cuda}}$)", fontweight="bold")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([str(b) for b in block_sizes])
    ax2.set_ylim(0, max(bw_limit, max(speedup_cuda)) * 1.30)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.92)

    plt.tight_layout()
    plt.savefig(output_fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_fig_path}")

# 4B. Speedup Comparison Across Grid Sizes
def plot_speedup_grid_sizes(reduced_csv=None, large_csv=None, output_fig_path=None):
    if reduced_csv is None:
        reduced_csv = os.path.join(DATA_DIR, "benchmarks_reduced.csv")
    if large_csv is None:
        large_csv = os.path.join(DATA_DIR, "benchmarks_large.csv")
    if output_fig_path is None:
        output_fig_path = os.path.join(FIG_DIR, "speedup_grid_sizes.png")

    if not os.path.exists(reduced_csv):
        print(f"[Warning] File not found: {reduced_csv}")
        return

    df_red = pd.read_csv(reduced_csv)
    omp_df = df_red[df_red["Backend"] == "OpenMP CPU"]
    cuda_df = df_red[df_red["Backend"] == "CUDA GPU"]

    # Build the bar groups from the backends actually present, so the figure
    # degrades cleanly when a machine has no CUDA rather than crashing.
    backends, red_speedups = ["Seq Baseline"], [1.0]
    if not omp_df.empty:
        opt_omp_th = int(omp_df.loc[omp_df["Speedup_vs_seq"].idxmax(), "Threads_or_BlockSize"])
        backends.append(f"OpenMP ({opt_omp_th} th)")
        red_speedups.append(float(omp_df["Speedup_vs_seq"].max()))
    if not cuda_df.empty:
        opt_cuda_b = int(cuda_df.loc[cuda_df["Speedup_vs_seq"].idxmax(), "Threads_or_BlockSize"])
        backends.append(f"CUDA (blk {opt_cuda_b})")
        red_speedups.append(float(cuda_df["Speedup_vs_seq"].max()))

    large_speedups, large_n = None, None
    if os.path.exists(large_csv):
        df_large = pd.read_csv(large_csv)
        large_speedups = list(df_large["Speedup_vs_seq"].values)
        large_n = int(df_large["N"].iloc[0])
        # Pad so both groups have one bar per backend label.
        while len(large_speedups) < len(backends):
            large_speedups.append(np.nan)
        large_speedups = large_speedups[:len(backends)]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    x = np.arange(len(backends))
    width = 0.35 if large_speedups is not None else 0.55
    red_n = int(df_red["N"].iloc[0])

    if large_speedups is not None:
        rects1 = ax.bar(x - width/2, red_speedups, width,
                        label=rf'Reduced Grid ($N={red_n}^3$)', color='#4C72B0')
        rects2 = ax.bar(x + width/2, large_speedups, width,
                        label=rf'Large Grid ($N={large_n}^3$)', color='#55A868')
    else:
        rects1 = ax.bar(x, red_speedups, width,
                        label=rf'$N={red_n}^3$', color='#4C72B0')
        rects2 = []

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}x', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        if np.isnan(h):
            continue
        ax.annotate(f'{h:.2f}x', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(backends, fontweight='bold')
    ax.set_ylabel(r"Speedup Factor ($S = T_{\mathrm{seq}} / T_{\mathrm{par}}$)", fontweight="bold")
    ax.axhline(y=1.0, color='gray', linestyle=':')
    _all = list(red_speedups) + (list(large_speedups) if large_speedups else [])
    ax.set_ylim(0, float(np.nanmax(_all)) * 1.25)
    ax.grid(True, linestyle="--", alpha=0.5, axis='y')
    ax.legend(loc="upper left", framealpha=0.92)

    plt.tight_layout()
    plt.savefig(output_fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_fig_path}")

# 5. Effective Memory Bandwidth
def plot_effective_bandwidth(reduced_csv=None, output_fig_path=None):
    if reduced_csv is None:
        reduced_csv = os.path.join(DATA_DIR, "benchmarks_reduced.csv")
    if output_fig_path is None:
        output_fig_path = os.path.join(FIG_DIR, "effective_bandwidth.png")

    if not os.path.exists(reduced_csv):
        print(f"[Warning] File not found: {reduced_csv}")
        return

    df_red = pd.read_csv(reduced_csv)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)

    labels, bws, colors = [], [], []
    for _, row in df_red.iterrows():
        b_name = row["Backend"]
        param = int(row["Threads_or_BlockSize"])
        if b_name == "Sequential":
            labels.append("Seq (1 th, untiled)"); colors.append("#D9381E")
        elif b_name == "OpenMP CPU":
            labels.append(f"OMP ({param} th)"); colors.append("#1F77B4")
        else:
            labels.append(f"CUDA (blk {param})"); colors.append("#2CA02C")
        bws.append(row["Bandwidth_GBs"] if "Bandwidth_GBs" in row else (row["BW_min_GBs"] if "BW_min_GBs" in row else 0.0))

    bws = np.array(bws)
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, bws, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.axvline(x=25.6, color="red", linestyle="--", linewidth=2.0,
               label="Jetson Peak BW (25.6 GB/s, spec)")

    for bar, bw in zip(bars, bws):
        ax.text(bw + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{bw:.2f} GB/s", va='center', fontweight='bold', fontsize=8.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontweight='bold', fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(bws.max(), 25.6) * 1.25)
    ax.set_xlabel("Effective Memory Bandwidth (GB/s)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis='x')
    ax.legend(loc="lower right", framealpha=0.92, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_fig_path}")

# 6. Roofline Model (Single Panel, No Titles)
def plot_roofline_model(reduced_csv=None, output_fig_path=None):
    if reduced_csv is None:
        reduced_csv = os.path.join(DATA_DIR, "benchmarks_reduced.csv")
    if output_fig_path is None:
        output_fig_path = os.path.join(FIG_DIR, "roofline_model.png")

    if not os.path.exists(reduced_csv):
        print(f"[Warning] File not found: {reduced_csv}")
        return

    df_red = pd.read_csv(reduced_csv)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)

    ai_range = np.logspace(-2, 2, 200)
    bw_bound = ai_range * 25.6
    fp32_peak = np.full_like(ai_range, 236.0)
    roofline = np.minimum(bw_bound, fp32_peak)

    ax.loglog(ai_range, roofline, color='#333333', linewidth=2.5, label="Jetson Ceiling (236 GFLOPS)")
    ax.axvline(x=9.22, color='gray', linestyle=':', label=r"Ridge Point ($I_r = 9.22$ FLOP/B)")

    # Arithmetic Intensity (compulsory / minimal traffic):
    # CPU: 24 FLOP / 20 B = 1.20 FLOP/B
    # GPU: 48 FLOP / 12 B = 4.00 FLOP/B
    ai_cpu = 24.0 / 20.0
    ai_gpu = 48.0 / 12.0

    seq_gflops = df_red[df_red["Backend"] == "Sequential"]["GFLOPS"].iloc[0]
    omp_gflops = df_red[df_red["Backend"] == "OpenMP CPU"]["GFLOPS"].max()
    cuda_rows = df_red[df_red["Backend"] == "CUDA GPU"]
    cuda_gflops = cuda_rows["GFLOPS"].max() if not cuda_rows.empty else np.nan

    backends_info = [
        (seq_gflops,  ai_cpu, '#D9381E', 'o', 'Sequential'),
        (omp_gflops,  ai_cpu, '#1F77B4', '^', 'OpenMP CPU'),
        (cuda_gflops, ai_gpu, '#2CA02C', 's', 'CUDA GPU')
    ]

    for gf, ai, colour, marker, name in backends_info:
        if gf is None or (isinstance(gf, float) and np.isnan(gf)):
            continue
        ax.plot(ai, gf, marker, color=colour, markersize=9,
                zorder=6, label=f"{name} ({gf:.2f} GFLOPS)")

    ax.set_xlabel("Arithmetic Intensity [FLOP/Byte]", fontweight="bold")
    ax.set_ylabel("Performance [GFLOPS]", fontweight="bold")
    ax.set_xlim(0.05, 50)
    min_perf = float(np.nanmin([seq_gflops, omp_gflops, cuda_gflops]))
    ax.set_ylim(min(0.01, min_perf * 0.5), 400)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(output_fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_fig_path}")


# Dispatcher
def main():
    parser = argparse.ArgumentParser(description="Plots for 3D heat solver benchmarks.")
    parser.add_argument("--plot", choices=["evolution", "mms", "times", "speedup_ideal", "speedup_grids", "bandwidth", "roofline", "all"],
                        default="all", help="Target plot (default: all)")
    parser.add_argument("--time", type=float, default=500.0, help="Physical duration in seconds (default: 500.0)")
    parser.add_argument("--grid", type=int, default=64, help="Grid resolution N (default: 64)")
    args = parser.parse_args()

    if args.plot in ["evolution", "all"]:
        plot_time_evolution(time_scale=args.time, n=args.grid)
    if args.plot in ["mms", "all"]:
        plot_mms_test()
    if args.plot in ["times", "all"]:
        plot_execution_times()
    if args.plot in ["speedup_ideal", "all"]:
        plot_speedup_ideal()
    if args.plot in ["speedup_grids", "all"]:
        plot_speedup_grid_sizes()
    if args.plot in ["bandwidth", "all"]:
        plot_effective_bandwidth()
    if args.plot in ["roofline", "all"]:
        plot_roofline_model()

if __name__ == "__main__":
    main()

