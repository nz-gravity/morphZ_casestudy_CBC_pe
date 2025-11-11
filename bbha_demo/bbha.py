#!/usr/bin/env python
"""
Two-stage Bilby analysis:
  (1) Run Dynesty nested sampling
  (2) Run Bilby-MCMC on the same injected signal

Injection parameters and sampler configurations are adapted from config files used in bilby-mcmc paper
https://git.ligo.org/gregory.ashton/bilby_mcmc_validation/-/tree/master/BBH_A
"""

import sys
from pathlib import Path

import bilby

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from compute_morphz_evidence import (
    compute_morphz_evidence,
    write_evidence_summary,
)

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
label = "bbh_A"
outdir = "outdir"
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
checkpoint_delta_t = 1800  # seconds between checkpoints

# -----------------------------------------------------------------------------
# Injection parameters
# -----------------------------------------------------------------------------
injection_parameters = dict(
    chirp_mass=17.051544979894693,
    mass_ratio=0.6183945489993522,
    chi_1=0.29526500202350264,
    chi_2=0.13262056301313416,
    chi_1_in_plane=0.0264673717225983,
    chi_2_in_plane=0.3701305583885513,
    phi_12=1.0962562029664955,
    phi_jl=0.518241237045709,
    luminosity_distance=497.2983560174788,
    dec=0.2205292600865073,
    ra=3.952677097361719,
    theta_jn=1.8795187965094322,
    psi=2.6973435044499543,
    phase=3.686990398567503,
    geocent_time=0.040833669551002205,
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
# Construct waveform generator
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
# Generate Gaussian-noise interferometers and inject signal
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
# Priors
# -----------------------------------------------------------------------------
priors = bilby.gw.prior.BBHPriorDict(filename="bbha.prior")
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 0.1,
    maximum=injection_parameters['geocent_time'] + 0.1,
    name='geocent_time',
)

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
dynesty_result_file = Path(outdir) / f"{dynesty_label}_result.json"
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
# Run Bilby-MCMC sampler (load cached result if present)
# -----------------------------------------------------------------------------
mcmc_label = label + "_mcmc"
mcmc_result_file = Path(outdir) / f"{mcmc_label}_result.json"
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

print("\nAnalyses complete!")
print(f"Results saved in: {Path(outdir).resolve()}")
print(f"Evidence summary: {summary_path}")
