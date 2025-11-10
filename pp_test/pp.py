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
import bilby
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------------
# Parse CLI argument
# -----------------------------------------------------------------------------
if len(sys.argv) != 2:
    sys.exit("Usage: python run_analysis.py <index>")

idx = int(sys.argv[1])
label = f"seed_{idx}"

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
# Dynesty run
# -----------------------------------------------------------------------------
result_dynesty = bilby.run_sampler(
    likelihood,
    priors,
    sampler="dynesty",
    outdir=outdir,
    label=label + "_dynesty",
    nlive=1000,
    nact=50,
    check_point_delta_t=600,
    npool=1,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    result_class=bilby.gw.result.CBCResult,
)
result_dynesty.plot_corner()

# -----------------------------------------------------------------------------
# Bilby-MCMC run
# -----------------------------------------------------------------------------
result_mcmc = bilby.run_sampler(
    likelihood,
    priors,
    sampler="bilby_mcmc",
    outdir=outdir,
    label=label + "_mcmc",
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
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
)
result_mcmc.plot_corner()

print(f"\nAnalysis for injection {idx} complete.")
print(f"Results saved in {outdir.resolve()}")
