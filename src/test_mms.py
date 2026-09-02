#!/usr/bin/env python3
"""
Grid convergence and verification test for the 3D MMS heat solver.

Verifies all three backends (Sequential, OpenMP, CUDA) against the exact
manufactured solution:
    u(x, y, z, t) = sin(pi x/L) sin(pi y/L) sin(pi z/L) exp(-3 alpha (pi/L)^2 t)

With r = alpha dt / h^2 held constant, dt ~ h^2, so the total truncation
error is O(h^2). Halving h reduces the L2 error by ~4x (order ~2).

Saves datasets to data/ for consumption by plot_results.py:
  - mms_convergence_3d.csv : L2 error vs grid spacing h across all backends
  - mms_u_evo_3d.csv        : Temporal temperature decay validation
  - mms_<backend>_n<N>.csv : Step-by-step evolution data per grid
"""

import os
import sys
import math
import re
import subprocess
import shlex
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# MMS Simulation Parameters
L = 1.0
ALPHA = 0.143
R = 0.80 / 6.0  # Safety factor 0.80 of the 3D stability limit dt <= dx^2/(6*alpha)
TAU = L**2 / (3.0 * math.pi**2 * ALPHA)
T_END = 2.0 * TAU
GRIDS = [16, 32, 64, 128]


def run_cmd(cmd):
    p = subprocess.run(cmd, shell=True, cwd=SRC_DIR, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def build():
    print("[Build] Compiling MMS simulation targets...")
    targets = ["heat_mms_seq_3d", "heat_mms_omp_3d"]
    code_cuda, _, _ = run_cmd("which nvcc || test -x /usr/local/cuda/bin/nvcc")
    has_cuda = (code_cuda == 0)

    if has_cuda:
        targets.append("heat_mms_cuda_3d")

    ret, _, err = run_cmd(f"make {' '.join(targets)}")
    if ret != 0:
        print(f"Build failed:\n{err}")
        sys.exit(1)
    return has_cuda


def parse_l2(stdout_text):
    m = re.search(r"L2-Norm Error\s*:\s*([0-9.eE+-]+)", stdout_text)
    return float(m.group(1)) if m else None


def test_convergence(has_cuda=False):
    print("\n--- 3D MMS Grid Convergence Test ---")
    rows = []
    backend_errors = {"seq": {}, "omp": {}, "cuda": {}}

    for n in GRIDS:
        h = L / (n + 1)
        dt = R * h**2 / ALPHA
        steps = max(1, round(T_END / dt))

        # 1. Sequential C baseline
        csv_seq = os.path.join(DATA_DIR, f"mms_seq_n{n}.csv")
        _, out_s, _ = run_cmd(f"./heat_mms_seq_3d --n {n} --steps {steps} --output {shlex.quote(csv_seq)}")
        l2_s = parse_l2(out_s)
        backend_errors["seq"][n] = (h, l2_s)

        # 2. OpenMP CPU (4 threads)
        csv_omp = os.path.join(DATA_DIR, f"mms_omp_n{n}.csv")
        _, out_o, _ = run_cmd(f"./heat_mms_omp_3d --n {n} --steps {steps} --threads 4 --output {shlex.quote(csv_omp)}")
        l2_o = parse_l2(out_o)
        backend_errors["omp"][n] = (h, l2_o)

        # 3. CUDA GPU (256 th/blk)
        l2_c = None
        if has_cuda and os.path.exists(os.path.join(SRC_DIR, "heat_mms_cuda_3d")):
            csv_cuda = os.path.join(DATA_DIR, f"mms_cuda_n{n}.csv")
            _, out_c, _ = run_cmd(f"./heat_mms_cuda_3d --n {n} --steps {steps} --block_size 256 --output {shlex.quote(csv_cuda)}")
            l2_c = parse_l2(out_c)
            backend_errors["cuda"][n] = (h, l2_c)

        rows.append({
            "N": n,
            "h": h,
            "L2_seq": l2_s,
            "L2_omp": l2_o,
            "L2_cuda": l2_c
        })

    df_conv = pd.DataFrame(rows)

    # Compute observed order of convergence
    orders = [np.nan]
    for i in range(1, len(df_conv)):
        h0, h1 = df_conv.loc[i-1, "h"], df_conv.loc[i, "h"]
        e0, e1 = df_conv.loc[i-1, "L2_seq"], df_conv.loc[i, "L2_seq"]
        order = math.log(e0 / e1) / math.log(h0 / h1)
        orders.append(order)
    df_conv["Observed_Order"] = orders

    # Print summary table
    print("-" * 88)
    if has_cuda and df_conv["L2_cuda"].notnull().all():
        print(f"{'N':<6} {'h':<12} {'L2 Error (Seq)':<18} {'L2 Error (OMP)':<18} {'L2 Error (CUDA)':<18} {'Order':<8}")
        print("-" * 88)
        for _, row in df_conv.iterrows():
            ord_str = f"{row['Observed_Order']:.2f}" if not np.isnan(row['Observed_Order']) else "N/A"
            print(f"{int(row['N']):<6} {row['h']:<12.6f} {row['L2_seq']:<18.6E} {row['L2_omp']:<18.6E} {row['L2_cuda']:<18.6E} {ord_str:<8}")
    else:
        print(f"{'N':<6} {'h':<12} {'L2 Error (Seq)':<18} {'L2 Error (OMP)':<18} {'Order':<8}")
        print("-" * 70)
        for _, row in df_conv.iterrows():
            ord_str = f"{row['Observed_Order']:.2f}" if not np.isnan(row['Observed_Order']) else "N/A"
            print(f"{int(row['N']):<6} {row['h']:<12.6f} {row['L2_seq']:<18.6E} {row['L2_omp']:<18.6E} {ord_str:<8}")
    print("-" * 88)

    # Save convergence CSV for plot_results.py
    conv_csv_path = os.path.join(DATA_DIR, "mms_convergence_3d.csv")
    df_conv.to_csv(conv_csv_path, index=False)
    print(f"Convergence dataset saved to: {conv_csv_path}")

    # Verify second-order convergence assertions
    for name, err_dict in backend_errors.items():
        if not err_dict or any(v[1] is None for v in err_dict.values()):
            continue
        prev = GRIDS[0]
        for n in GRIDS[1:]:
            h0, e0 = err_dict[prev]
            h1, e1 = err_dict[n]
            ord_val = math.log(e0 / e1) / math.log(h0 / h1)
            ok = 1.7 < ord_val < 2.3
            assert ok, f"Backend {name}: order {ord_val:.2f} outside [1.7, 2.3]"
            prev = n

    print("All backends confirmed second-order spatial convergence O(h^2).")
    return df_conv


def test_decay(has_cuda=False):
    print("\n--- 3D MMS Temperature Decay Test ---")
    n = 64
    fixed_steps = 200
    out_csv_decay = os.path.join(DATA_DIR, "mms_u_evo_3d.csv")

    if has_cuda and os.path.exists(os.path.join(SRC_DIR, "heat_mms_cuda_3d")):
        cmd = f"./heat_mms_cuda_3d --n {n} --steps {fixed_steps} --block_size 256 --output {shlex.quote(out_csv_decay)}"
    else:
        cmd = f"./heat_mms_omp_3d --n {n} --steps {fixed_steps} --threads 4 --output {shlex.quote(out_csv_decay)}"

    ret, _, err = run_cmd(cmd)
    if ret != 0:
        print(f"Error in decay test:\n{err}")
        return

    if os.path.exists(out_csv_decay):
        df_evo = pd.read_csv(out_csv_decay)
        diff = np.abs(df_evo["u_avg"] - df_evo["u_exact_avg"])
        print(f"Decay dataset saved to: {out_csv_decay}")
        print(f"Steps: {len(df_evo)} | Max Error: {diff.max():.4E} | Mean Error: {diff.mean():.4E}")


def main():
    has_cuda = build()
    test_convergence(has_cuda)
    test_decay(has_cuda)
    print("\nMMS tests completed successfully.\n")


if __name__ == "__main__":
    main()
