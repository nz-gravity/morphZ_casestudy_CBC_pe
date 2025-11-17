#!/usr/bin/env python
"""Shared helpers for computing morphZ evidence values from bilby results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import bilby
import numpy as np
from morphZ import evidence as morphz_evidence


def load_result(result_path: str | Path) -> bilby.result.Result:
    """Load a bilby result JSON file."""
    path = Path(result_path)
    if not path.exists():
        raise FileNotFoundError(f"Result file {path} not found")
    return bilby.result.read_in_result(str(path))


def compute_morphz_evidence(
    *,
    result: bilby.result.Result,
    likelihood: bilby.likelihood.Likelihood,
    priors: bilby.prior.PriorDict | None = None,
    n_resamples: int = 1000,
    morph_type: str = "pair",
    kde_bw: str | float = "silverman",
    n_estimations: int = 1,
    verbose: bool = True,
) -> dict:
    """Run morphZ evidence estimation given a bilby result."""
    posterior = result.posterior
    if posterior is None or posterior.empty:
        raise RuntimeError("Result contains no posterior samples for morphZ.")

    param_names = list(result.search_parameter_keys)
    if not param_names:
        raise RuntimeError("Result does not list search parameter keys.")

    missing_columns = {"log_likelihood", "log_prior"} - set(posterior.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RuntimeError(f"Posterior missing required columns: {missing}")

    morph_prior = priors or getattr(result, "priors", None)
    if morph_prior is None:
        raise RuntimeError("Priors are required for morphZ evidence calculation.")

    def log_posterior(theta: np.ndarray) -> float:
        params = dict(zip(param_names, theta))
        log_prior = morph_prior.ln_prob(params)
        if not np.isfinite(log_prior):
            return log_prior
        log_likelihood = likelihood.log_likelihood(params)
        return log_likelihood + log_prior

    morphz_samples = posterior[param_names].to_numpy()
    log_posterior_vals = (
        posterior["log_likelihood"].to_numpy()
        + posterior["log_prior"].to_numpy()
    )

    morphz_output_dir = Path(getattr(result, "outdir", ".")).joinpath("morphZ")
    morphz_output_dir.mkdir(parents=True, exist_ok=True)

    morphz_runs = morphz_evidence(
        post_samples=morphz_samples,
        log_posterior_values=log_posterior_vals,
        log_posterior_function=log_posterior,
        n_resamples=n_resamples,
        morph_type=morph_type,
        kde_bw=kde_bw,
        param_names=param_names,
        output_path=str(morphz_output_dir),
        n_estimations=n_estimations,
        verbose=verbose,
    )

    results_array = np.atleast_2d(np.asarray(morphz_runs, dtype=float))
    logz_runs = results_array[:, 0]
    err_runs = (
        results_array[:, 1]
        if results_array.shape[1] > 1
        else np.full_like(logz_runs, np.nan)
    )

    logz_estimate = float(np.mean(logz_runs))
    if results_array.shape[1] > 1 and np.isfinite(err_runs).all():
        error_estimate = float(np.sqrt(np.mean(err_runs**2)))
    elif logz_runs.size > 1:
        error_estimate = float(np.std(logz_runs, ddof=1) / np.sqrt(logz_runs.size))
    else:
        error_estimate = float(np.nan)

    return {
        "logz_runs": logz_runs.tolist(),
        "error_runs": err_runs.tolist(),
        "logz_estimate": logz_estimate,
        "error_estimate": error_estimate,
        "param_names": param_names,
        "n_resamples": n_resamples,
        "morph_type": morph_type,
        "kde_bw": kde_bw,
        "n_estimations": n_estimations,
        "n_runs": int(logz_runs.size),
    }


def write_evidence_summary(
    outdir: str | Path,
    rows: Iterable[dict],
    filename: str = "evidence_summary.csv",
) -> Path:
    """Persist evidence summaries to CSV."""
    output_path = Path(outdir) / filename
    fieldnames = [
        "entry",
        "method",
        "kind",
        "log_evidence",
        "log_evidence_err",
        "n_runs",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {name: "" for name in fieldnames}
            record.update(row)
            writer.writerow(record)
    return output_path
