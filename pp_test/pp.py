#!/usr/bin/env python
"""
Run Bilby PE on one injection chosen from a CSV file.

Usage:
    python run_analysis.py <index>

This script:
  - loads the <index>th injection from injections.csv
  - creates outdir/seed_<index>/
  - reads priors from pp.prior
  - runs both Dynesty and Bilby-MCMC samplers
"""

import sys
from pathlib import Path

import bilby
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from compute_morphz_evidence import (
    compute_morphz_evidence,
    write_evidence_summary,
)

# -----------------------------------------------------------------------------
# Parse CLI argument
# -----------------------------------------------------------------------------
if len(sys.argv) != 2:
    sys.exit("Usage: python run_analysis.py <index>")

idx = int(sys.argv[1])
label = f"seed_{idx}"
checkpoint_delta_t = 1800  # seconds between checkpoints

# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
injection_csv = Path("injections.csv")
prior_file = Path("pp.prior")
outdir = Path("outdir") / label
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)

# -----------------------------------------------------------------------------
# Load injection parameters
# -----------------------------------------------------------------------------
injections = pd.read_csv(injection_csv, index_col=0)
if idx < 0 or idx >= len(injections):
    sys.exit(f"Index {idx} out of range (0–{len(injections)-1})")
injection_parameters = injections.iloc[idx].to_dict()
print(f"Loaded injection {idx}:")
for k, v in injection_parameters.items():
    print(f"  {k} = {v}")

# -----------------------------------------------------------------------------
# Load priors
# -----------------------------------------------------------------------------
priors = bilby.gw.prior.BBHPriorDict(filename=str(prior_file))
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 0.1,
    maximum=injection_parameters['geocent_time'] + 0.1,
    name='geocent_time',
)

# -----------------------------------------------------------------------------
# Analysis parameters
# -----------------------------------------------------------------------------
detectors = ["H1", "L1"]
duration = 4.0
sampling_frequency = 4096
post_trigger_duration = 2.0
minimum_frequency = 20.0
maximum_frequency = 1024.0
tukey_roll_off = 0.4

# -----------------------------------------------------------------------------
# Waveform generator
# -----------------------------------------------------------------------------
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=20,
    ),
)

# -----------------------------------------------------------------------------
# Interferometers + injection
# -----------------------------------------------------------------------------
ifos = bilby.gw.detector.InterferometerList(detectors)
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)
ifos.plot_data(outdir=outdir, label=label)

# -----------------------------------------------------------------------------
# Likelihood
# -----------------------------------------------------------------------------
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifos,
    waveform_generator,
    priors=priors,
    time_marginalization=True,
    phase_marginalization=False,
    distance_marginalization=True,
)

# -----------------------------------------------------------------------------
dynesty_label = label + "_dynesty"
dynesty_result_file = outdir / f"{dynesty_label}_result.json"
if dynesty_result_file.exists():
    print(f"Loading existing Dynesty result from {dynesty_result_file}")
    result_dynesty = bilby.result.read_in_result(str(dynesty_result_file))
else:
    result_dynesty = bilby.run_sampler(
        likelihood,
        priors,
        sampler="dynesty",
        outdir=outdir,
        label=dynesty_label,
        nlive=1000,
        nact=50,
        check_point_delta_t=checkpoint_delta_t,
        npool=1,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
    )
    result_dynesty.plot_corner()

print(
    f"Dynesty LnZ: {result_dynesty.log_evidence:.3f} +/- "
    f"{result_dynesty.log_evidence_err:.3f}"
)
dynesty_morph = compute_morphz_evidence(
    result=result_dynesty,
    likelihood=likelihood,
    priors=priors,
)
print(
    f"morphZ LnZ (Dynesty posterior): {dynesty_morph['logz_estimate']:.3f} +/- "
    f"{dynesty_morph['error_estimate']:.3f}"
)

# -----------------------------------------------------------------------------
mcmc_label = label + "_mcmc"
mcmc_result_file = outdir / f"{mcmc_label}_result.json"
if mcmc_result_file.exists():
    print(f"Loading existing Bilby-MCMC result from {mcmc_result_file}")
    result_mcmc = bilby.result.read_in_result(str(mcmc_result_file))
else:
    result_mcmc = bilby.run_sampler(
        likelihood,
        priors,
        sampler="bilby_mcmc",
        outdir=outdir,
        label=mcmc_label,
        sampler_kwargs=dict(
            nsamples=2000,
            thin_by_nact=0.2,
            ntemps=1,
            npool=1,
            Tmax_from_SNR=20,
            adapt=True,
            proposal_cycle="gwA",
            L1steps=100,
            L2steps=5,
        ),
        check_point_delta_t=checkpoint_delta_t,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    )
    result_mcmc.plot_corner()

mcmc_morph = compute_morphz_evidence(
    result=result_mcmc,
    likelihood=likelihood,
    priors=priors,
)
print(
    f"morphZ LnZ (Bilby-MCMC posterior): {mcmc_morph['logz_estimate']:.3f} +/- "
    f"{mcmc_morph['error_estimate']:.3f}"
)

summary_rows = [
    {
        "entry": "dynesty_original",
        "method": "dynesty",
        "kind": "sampler",
        "log_evidence": result_dynesty.log_evidence,
        "log_evidence_err": result_dynesty.log_evidence_err,
        "n_runs": 1,
    },
    {
        "entry": "dynesty_morphz",
        "method": "dynesty",
        "kind": "morphZ",
        "log_evidence": dynesty_morph["logz_estimate"],
        "log_evidence_err": dynesty_morph["error_estimate"],
        "n_runs": dynesty_morph["n_runs"],
    },
    {
        "entry": "mcmc_morphz",
        "method": "bilby_mcmc",
        "kind": "morphZ",
        "log_evidence": mcmc_morph["logz_estimate"],
        "log_evidence_err": mcmc_morph["error_estimate"],
        "n_runs": mcmc_morph["n_runs"],
    },
]
summary_path = write_evidence_summary(outdir, summary_rows)
print(f"Evidence summary saved to {summary_path}")

print(f"\nAnalysis for injection {idx} complete.")
print(f"Results saved in {outdir.resolve()}")
