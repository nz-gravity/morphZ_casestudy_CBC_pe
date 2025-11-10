#!/usr/bin/env python
"""
Two-stage Bilby analysis:
  (1) Run Dynesty nested sampling
  (2) Run Bilby-MCMC on the same injected signal

Injection parameters and sampler configurations are adapted from config files used in bilby-mcmc paper
https://git.ligo.org/gregory.ashton/bilby_mcmc_validation/-/tree/master/BBH_A
"""

import bilby
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
label = "bbh_A"
outdir = "outdir"
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)

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
priors = bilby.gw.prior.BBHPriorDict()
priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(
    name="chirp_mass", minimum=16, maximum=18
)
priors["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(
    name="mass_ratio", minimum=0.125, maximum=1
)
priors["a_1"] = bilby.core.prior.Uniform(name="a_1", minimum=0, maximum=0.99)
priors["a_2"] = bilby.core.prior.Uniform(name="a_2", minimum=0, maximum=0.99)
priors["tilt_1"] = bilby.core.prior.Sine(name="tilt_1")
priors["tilt_2"] = bilby.core.prior.Sine(name="tilt_2")
priors["phi_12"] = bilby.core.prior.Uniform(
    name="phi_12", minimum=0, maximum=2 * np.pi, boundary="periodic"
)
priors["phi_jl"] = bilby.core.prior.Uniform(
    name="phi_jl", minimum=0, maximum=2 * np.pi, boundary="periodic"
)
priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
    name="luminosity_distance", minimum=100, maximum=5000, unit="Mpc"
)
priors["dec"] = bilby.core.prior.Cosine(name="dec")
priors["ra"] = bilby.core.prior.Uniform(
    name="ra", minimum=0, maximum=2 * np.pi, boundary="periodic"
)
priors["cos_theta_jn"] = bilby.core.prior.Uniform(
    name="cos_theta_jn", minimum=-1, maximum=1
)
priors["psi"] = bilby.core.prior.Uniform(
    name="psi", minimum=0, maximum=np.pi, boundary="periodic"
)
priors["delta_phase"] = bilby.core.prior.Uniform(
    name="delta_phase", minimum=-np.pi, maximum=np.pi, boundary="periodic"
)
priors["geocent_time"] = bilby.core.prior.Uniform(
    minimum=injection_parameters["geocent_time"] - 0.1,
    maximum=injection_parameters["geocent_time"] + 0.1,
    name="geocent_time",
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
# Run Dynesty Sampler
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
# Run Bilby-MCMC Sampler
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

print("\nAnalyses complete!")
print(f"Results saved in: {Path(outdir).resolve()}")
