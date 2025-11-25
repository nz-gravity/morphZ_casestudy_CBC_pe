#!/usr/bin/env python
"""Generate LaTeX tables summarising GW150914 and injection-ensemble evidences."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GW150914_METHOD_ORDER = [
    ("dynesty", "Dynesty"),
    ("mcmc", "MCMC"),
    ("dynesty_morphz", "MorphZ(Dynesty)"),
    ("mcmc_morphz", "MorphZ(MCMC)"),
]


def format_unc(value: float, err: float) -> str:
    return f"{value:.3f} \\pm {err:.3f}"


def generate_gw150914_table(csv_path: Path) -> str:
    df = pd.read_csv(csv_path).set_index("method")
    if "dynesty" not in df.index:
        raise ValueError("GW150914 CSV must contain a 'dynesty' row")

    header_cols = ["\\textbf{Dataset}"] + [
        f"$\\ln Z_{{\\mathrm{{{label}}}}} \\pm \\sigma$" for _, label in GW150914_METHOD_ORDER
    ]
    header_line = " & ".join(header_cols) + r" \\\\" 

    gw_row = ["GW150914"]
    for method, _ in GW150914_METHOD_ORDER:
        row = df.loc[method]
        gw_row.append(format_unc(row["lnz"], row["lnz_err"]))
    gw_line = " & ".join(gw_row) + r" \\\\" 

    delta_row = ["\\Delta \\text{vs Dynesty}"]
    base = df.loc["dynesty", "lnz"]
    for method, _ in GW150914_METHOD_ORDER:
        delta_row.append(f"{df.loc[method, 'lnz'] - base:.3f}")
    delta_line = " & ".join(delta_row) + r" \\\\" 

    lines = [
        "\\begin{tabular}{l" + "c" * len(GW150914_METHOD_ORDER) + "}",
        "\\toprule",
        header_line,
        gw_line,
        delta_line,
        "\\bottomrule",
        "\\end{tabular}",
    ]

    return "\n".join(lines)


def summarise_deltas(values: pd.Series) -> tuple[str, str, float, float]:
    median = values.median()
    q25 = values.quantile(0.25)
    q75 = values.quantile(0.75)
    mean = values.mean()
    std = values.std(ddof=1)
    frac_half = (values.abs() < 0.5).mean() * 100
    frac_one = (values.abs() < 1.0).mean() * 100
    return (
        f"{median:.3f}\\,[{q25:.3f},\\,{q75:.3f}]",
        f"{mean:.3f}\\,\\pm\\,{std:.3f}",
        frac_half,
        frac_one,
    )


def generate_pp_table(csv_path: Path) -> str:
    df = pd.read_csv(csv_path)
    required = [
        "lnz_dynesty",
        "lnz_mcmc",
        "lnz_morph_dynesty",
        "lnz_morph_mcmc",
    ]
    if not set(required).issubset(df.columns):
        raise ValueError("PP evidences CSV is missing required columns")

    deltas = {
        "MCMC": df["lnz_mcmc"] - df["lnz_dynesty"],
        "MorphZ(Dynesty)": df["lnz_morph_dynesty"] - df["lnz_dynesty"],
        "MorphZ(MCMC)": df["lnz_morph_mcmc"] - df["lnz_dynesty"],
    }

    header = (
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Method & Median [IQR] & Mean $\\pm$ Std & $|\\Delta|<0.5$ & $|\\Delta|<1.0$ \\\\ \n"
        "\\midrule"
    )

    rows = []
    for label in ["MCMC", "MorphZ(Dynesty)", "MorphZ(MCMC)"]:
        med_iqr, mean_std, frac_half, frac_one = summarise_deltas(deltas[label].dropna())
        rows.append(
            f"{label} & {med_iqr} & {mean_std} & {frac_half:.1f}\\% & {frac_one:.1f}\\% \\\\"  # noqa: E501
        )

    footer = "\\bottomrule\n\\end{tabular}"
    return "\n".join([header, *rows, footer])


def save_table(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from evidence CSV files.")
    parser.add_argument(
        "--gw150914-csv",
        type=Path,
        default=Path("GW150914_bayeswave/evidences.csv"),
        help="Input CSV for GW150914 evidences.",
    )
    parser.add_argument(
        "--pp-csv",
        type=Path,
        default=Path("fast_pp/evidences.csv"),
        help="Input CSV for the injection ensemble evidences.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("tables"),
        help="Directory for generated LaTeX tables.",
    )
    parser.add_argument(
        "--skip-gw",
        action="store_true",
        help="Skip generating the GW150914 table.",
    )
    parser.add_argument(
        "--skip-pp",
        action="store_true",
        help="Skip generating the injection ensemble table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_gw:
        table = generate_gw150914_table(args.gw150914_csv)
        save_table(table, args.outdir / "gw150914_table.tex")

    if not args.skip_pp:
        table = generate_pp_table(args.pp_csv)
        save_table(table, args.outdir / "pp_ensemble_table.tex")


if __name__ == "__main__":
    main()
