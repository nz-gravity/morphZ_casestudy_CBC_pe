from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHODS = [
    "dynesty",
    "mcmc",
    "dynesty_morphz",
    "mcmc_morphz",
]
CACHE_FILENAME = "fast_pp_lnz_cache.csv"
PLOT_FILENAME = "fast_pp_lnz.png"


def parse_seed_id(label: str) -> int | None:
    if label.startswith("seed_"):
        try:
            return int(label.split("_", 1)[1])
        except ValueError:
            pass
    match = re.search(r"(\d+)", label)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def collect_lnz_comparisons(search_root: str) -> pd.DataFrame:
    rows = []
    csv_files = []
    for root, _, files in os.walk(search_root):
        for filename in files:
            if filename.endswith("_lnz_comparison.csv"):
                csv_files.append(os.path.join(root, filename))
    for csv_path in sorted(csv_files):
        try:
            df = pd.read_csv(csv_path)
        except OSError:
            continue
        required_cols = {"method", "lnz", "lnz_err"}
        if not required_cols.issubset(df.columns):
            continue

        stem = os.path.splitext(os.path.basename(csv_path))[0]
        label = stem.replace("_lnz_comparison", "")
        row = {
            "label": label,
            "seed_id": parse_seed_id(label),
            "source_file": csv_path,
        }
        for method in METHODS:
            row[f"lnz_{method}"] = float("nan")
            row[f"lnz_err_{method}"] = float("nan")

        for _, method_row in df.iterrows():
            method_name = str(method_row["method"]).strip().lower()
            if method_name not in METHODS:
                continue
            row[f"lnz_{method_name}"] = method_row["lnz"]
            row[f"lnz_err_{method_name}"] = method_row["lnz_err"]

        rows.append(row)

    return pd.DataFrame(rows)


def load_or_create_cache(search_root: str, cache_path: str) -> pd.DataFrame:
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    cache_dir = os.path.dirname(cache_path) or "."
    os.makedirs(cache_dir, exist_ok=True)
    df = collect_lnz_comparisons(search_root)
    if not df.empty:
        df.to_csv(cache_path, index=False)
    return df


def plot_lnz(df: pd.DataFrame, output_path: str) -> None:
    if df.empty:
        print("No LnZ comparison files found. Nothing to plot.")
        return

    df = df.sort_values(by=["seed_id", "label"], na_position="last").reset_index(drop=True)
    seed_ids = df["seed_id"].copy()
    if seed_ids.isna().all():
        seed_ids = pd.Series(range(len(df)), index=df.index)
    else:
        fallback = pd.Series(range(len(df)), index=df.index)
        seed_ids = seed_ids.fillna(fallback)
    y = seed_ids.astype(float).values

    dynesty_col = "lnz_dynesty"
    dynesty_err_col = "lnz_err_dynesty"
    if dynesty_col not in df.columns:
        print("Dynesty data not available; cannot compute differences.")
        return
    dynesty_vals = df[dynesty_col].astype(float)
    dynesty_err = df[dynesty_err_col].astype(float) if dynesty_err_col in df.columns else pd.Series(np.nan, index=df.index)

    plt.figure(figsize=(10, 6))
    for method in METHODS:
        if method == "dynesty":
            continue
        x_col = f"lnz_{method}"
        xerr_col = f"lnz_err_{method}"
        if x_col not in df.columns:
            continue
        method_vals = df[x_col].astype(float)
        if method_vals.isna().all():
            continue
        diff = method_vals - dynesty_vals
        if xerr_col in df.columns:
            method_err = df[xerr_col].astype(float)
        else:
            method_err = pd.Series(np.nan, index=df.index)
        diff_err = np.sqrt(method_err**2 + dynesty_err**2)
        plt.errorbar(
            diff,
            y,
            xerr=diff_err,
            marker="o",
            linestyle="none",
            capsize=3,
            label=method.replace("_", " "),
        )

    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Seed ID")
    plt.xlabel("Δ log evidence vs dynesty")
    plt.title("Fast-PP log evidence differences")
    plt.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    cache_path = os.path.join(script_dir, CACHE_FILENAME)
    plot_path = os.path.join(script_dir, PLOT_FILENAME)

    df = load_or_create_cache(project_root, cache_path)
    plot_lnz(df, plot_path)


if __name__ == "__main__":
    main()
