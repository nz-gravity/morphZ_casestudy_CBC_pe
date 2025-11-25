#!/usr/bin/env python
"""Regenerate LaTeX tables for GW150914 and the injection ensemble evidences."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


GW_CSV = Path("GW150914_bayeswave/evidences.csv")
PP_CSV = Path("fast_pp/evidences.csv")
OUTDIR = Path("tables")

GW_METHODS = [
    ("dynesty", "Dynesty"),
    ("mcmc", "MCMC"),
    ("dynesty_morphz", "MorphZ(Dynesty)"),
    ("mcmc_morphz", "MorphZ(MCMC)"),
]


def format_unc(value: float, err: float) -> str:
    return f"{value:.3f} \\pm {err:.3f}"


def generate_gw_table() -> str:
    df = pd.read_csv(GW_CSV).set_index("method")
    if "dynesty" not in df.index:
        raise ValueError("GW150914 evidences.csv must contain a 'dynesty' entry.")

    header = (
        ["\\textbf{Dataset}"]
        + [f"$\\ln Z_{{\\mathrm{{{label}}}}} \\pm \\sigma$" for _, label in GW_METHODS]
    )
    body_row = ["GW150914"]
    for method, _ in GW_METHODS:
        body_row.append(format_unc(df.loc[method, "lnz"], df.loc[method, "lnz_err"]))

    base = df.loc["dynesty", "lnz"]
    delta_row = ["\\Delta \\text{vs Dynesty}"]
    for method, _ in GW_METHODS:
        delta_row.append(f"{df.loc[method, 'lnz'] - base:.3f}")

    lines = [
        "\\begin{tabular}{l" + "c" * len(GW_METHODS) + "}",
        "\\toprule",
        " & ".join(header) + r" \\",
        " & ".join(body_row) + r" \\",
        " & ".join(delta_row) + r" \\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def summarise_deltas(values: pd.Series) -> tuple[str, str, str, str]:
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
        f"{frac_half:.1f}\\%",
        f"{frac_one:.1f}\\%",
    )


def generate_pp_table() -> str:
    df = pd.read_csv(PP_CSV)
    required = {
        "lnz_dynesty",
        "lnz_mcmc",
        "lnz_morph_dynesty",
        "lnz_morph_mcmc",
    }
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"PP evidences missing columns: {missing}")

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
        rows.append(f"{label} & {med_iqr} & {mean_std} & {frac_half} & {frac_one} \\\\")

    footer = "\\bottomrule\n\\end{tabular}"
    return "\n".join([header, *rows, footer])


def save_table(content: str, filename: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / filename
    path.write_text(content)
    print(f"Wrote {path}")


def main() -> None:
    save_table(generate_gw_table(), "gw150914_table.tex")
    save_table(generate_pp_table(), "pp_ensemble_table.tex")


if __name__ == "__main__":
    main()
