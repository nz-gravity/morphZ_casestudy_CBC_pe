from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    "dynesty",
    "mcmc",
    "dynesty_morphz",
    "mcmc_morphz",
]
DEFAULT_ROOT_DIR = Path("/fred/oz303/ezahraoui/github/fast_pp/outdir")
CACHE_FILENAME = "evidences.csv"
PLOT_FILENAME = "pp_lnz.png"


def load_seed_comparisons(root_dir: Path) -> dict[str, pd.DataFrame]:
    """Return DataFrames keyed by seed folder names (e.g., seed_1)."""
    data: dict[str, pd.DataFrame] = {}
    for seed_dir in sorted(root_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        csv_path = seed_dir / f"{seed_dir.name}_lnz_comparison.csv"
        if not csv_path.is_file():
            continue
        data[seed_dir.name] = pd.read_csv(csv_path)
    return data


def comparisons_to_frame(comparisons: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flatten the per-seed comparisons into a single table."""
    rows = []
    for seed, df in comparisons.items():
        df = df.copy()
        df["seed"] = seed
        rows.append(df[["seed", "method", "lnz", "lnz_err"]])
    if not rows:
        return pd.DataFrame(columns=["seed", "method", "lnz", "lnz_err"])
    return pd.concat(rows, ignore_index=True)


def load_cached_results(
    root_dir: Path, cache_path: Path, force_refresh: bool = False
) -> dict[str, pd.DataFrame]:
    """Load cached evidences or refresh from the source directory."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_refresh:
        return frame_to_comparisons(pd.read_csv(cache_path))

    comparisons = load_seed_comparisons(root_dir)
    combined = comparisons_to_frame(comparisons)
    if combined.empty:
        raise FileNotFoundError(
            f"No lnz comparison CSV files found under {root_dir}"
        )
    combined.to_csv(cache_path, index=False)
    return frame_to_comparisons(combined)


def frame_to_comparisons(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert the cached table back into a seed -> DataFrame mapping."""
    results: dict[str, pd.DataFrame] = {}
    grouped = frame.groupby("seed", sort=True)
    for seed, df in grouped:
        results[str(seed)] = df[["method", "lnz", "lnz_err"]].reset_index(drop=True)
    return results


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 18,
            "axes.labelsize": 24,
            "axes.titlesize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
        }
    )


def plot_delta_lnz_spaghetti(
    results_by_seed,
    baseline="dynesty",
    methods=("dynesty", "mcmc", "dynesty_morphz", "mcmc_morphz"),
    sample=None,
    propagate_baseline_error=False,
    colors=None,
    random_state=None,
):
    """
    results_by_seed: dict[str -> pd.DataFrame] with columns ['method','lnz','lnz_err']
    baseline: method name to anchor at x=0
    methods: order to plot
    sample: optional int to subsample seeds
    propagate_baseline_error: if True, xerr for non-baseline methods is
      sqrt(err_method^2 + err_baseline^2). If False (default), uses each method's own err.
    colors: optional dict to override colors by method
    """

    seeds = list(results_by_seed.keys())
    if sample is not None and sample < len(seeds):
        rng = np.random.default_rng(random_state)
        seeds = list(rng.choice(seeds, size=sample, replace=False))

    lnz_list, err_list, kept = [], [], []
    needed = set(methods)
    for s in seeds:
        df = results_by_seed[s]
        if not needed.issubset(set(df["method"])):
            continue
        row = df.set_index("method").loc[list(methods)]
        lnz_list.append(row["lnz"].to_numpy())
        err_list.append(row["lnz_err"].to_numpy())
        kept.append(s)

    if not kept:
        raise ValueError("No seeds contain all requested methods.")

    lnz = np.asarray(lnz_list)
    err = np.asarray(err_list)
    seeds = kept

    base_idx = methods.index(baseline)
    delta = lnz - lnz[:, [base_idx]]

    if propagate_baseline_error:
        rel_err = np.sqrt(err**2 + err[:, [base_idx]] ** 2)
        rel_err[:, base_idx] = err[:, base_idx]
    else:
        rel_err = err

    order = np.argsort(lnz[:, base_idx])
    seeds = [seeds[i] for i in order]
    delta = delta[order]
    rel_err = rel_err[order]

    default_colors = {
        "dynesty": "#8ab2b7",
        "mcmc": "#f8474a",
        "dynesty_morphz": "#19bac9",
        "mcmc_morphz": "#c31e23",
    }

    if colors:
        default_colors.update(colors)

    fig, ax = plt.subplots(figsize=(11, 24))
    y = np.arange(len(seeds))
    labels = ["Dynesty", "SS(MCMC)", "Morphz(Dynesty)", "Morphz(MCMC)"]
    for j, method in enumerate(methods):
        if j == 1:
            continue
        ax.errorbar(
            delta[:, j],
            y,
            xerr=rel_err[:, j],
            fmt="o",
            ms=9,
            capsize=1,
            elinewidth=7,
            color=default_colors.get(method),
            alpha=0.9,
            label=f"{labels[j]} (baseline)" if j == base_idx else labels[j],
        )
    ax.errorbar(
        delta[:, 0],
        y,
        xerr=rel_err[:, 0],
        fmt="o",
        ms=8,
        capsize=1,
        elinewidth=7,
        color=default_colors.get("dynesty"),
        alpha=0.9,
    )
    ax.vlines(x=1, ymin=0, ymax=len(seeds) - 1, linestyles="dashed", alpha=0.5, colors="grey")
    ax.vlines(x=-1, ymin=0, ymax=len(seeds) - 1, linestyles="dashed", alpha=0.6, colors="grey")

    ax.legend(loc="lower right", frameon=True)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.6)
    ax.set_yticks(np.arange(0, len(seeds)))
    ax.set_yticklabels(np.arange(1, len(seeds) + 1))
    ax.set_xlabel("Δ log(z) relative to Dynesty")
    ax.set_ylim([-1, len(seeds)])

    ax.grid(axis="y", alpha=0.1)
    fig.tight_layout()
    return fig, ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot log-evidence comparisons across seeds.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help="Directory containing seed_* subfolders with lnz comparison CSVs.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).resolve().parent / CACHE_FILENAME,
        help="Path to the cached flattened evidences CSV.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path(__file__).resolve().parent / PLOT_FILENAME,
        help="Output path for the spaghetti plot PNG.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Rebuild the cache from the root directory even if it exists.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optionally subsample this many seeds before plotting.",
    )
    parser.add_argument(
        "--baseline",
        choices=METHODS,
        default="dynesty",
        help="Method to anchor Δlog(z).",
    )
    parser.add_argument(
        "--propagate-baseline-error",
        action="store_true",
        help="Propagate baseline uncertainty into Δlog(z) errors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    results_by_seed = load_cached_results(
        root_dir=args.root,
        cache_path=args.cache,
        force_refresh=args.force_refresh,
    )

    fig, _ = plot_delta_lnz_spaghetti(
        results_by_seed,
        baseline=args.baseline,
        sample=args.sample,
        propagate_baseline_error=args.propagate_baseline_error,
    )
    fig.savefig(args.plot)
    plt.close(fig)


if __name__ == "__main__":
    main()
