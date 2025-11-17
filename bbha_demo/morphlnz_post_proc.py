#!/usr/bin/env python
"""Post-process a finished bilby run to estimate morphZ evidence.

This script rebuilds the likelihood/priors used in ``bbha.py`` and runs
``compute_morphz_evidence`` on the cached sampler result at ``RESULT_PATH``.
"""

import sys
from pathlib import Path

import bilby

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from compute_morphz_evidence import compute_morphz_evidence, load_result

# Path to the completed bilby run (Bilby-MCMC result JSON)
RESULT_PATH = '/fred/oz303/avajpeyi/studies/morphZ_casestudy_CBC_pe/bbha_demo/outdir/bbh_A_mcmc_result.json'


def build_likelihood_and_priors():
    """Recreate the priors and likelihood used in bbha.py."""
    # Injection parameters match those used when generating the cached result
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

    detectors = ["H1", "L1"]
    duration = 4.0
    sampling_frequency = 4096

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

    ifos = bilby.gw.detector.InterferometerList(detectors)
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - 2,
    )
    ifos.inject_signal(
        waveform_generator=waveform_generator, parameters=injection_parameters
    )

    prior_file = Path(__file__).resolve().parent / "bbha.prior"
    priors = bilby.gw.prior.BBHPriorDict(filename=str(prior_file))
    priors['geocent_time'] = bilby.core.prior.Uniform(
        minimum=injection_parameters['geocent_time'] - 0.1,
        maximum=injection_parameters['geocent_time'] + 0.1,
        name='geocent_time',
    )

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifos,
        waveform_generator,
        priors=priors,
        time_marginalization=True,
        phase_marginalization=False,
        distance_marginalization=True,
    )

    return likelihood, priors


def main():
    result = load_result(RESULT_PATH)
    likelihood, priors = build_likelihood_and_priors()

    morph = compute_morphz_evidence(
        result=result,
        likelihood=likelihood,
        priors=priors,
    )
    print(
        f"morphZ LnZ: {morph['logz_estimate']:.3f} +/- "
        f"{morph['error_estimate']:.3f}"
    )
    print(f"Finished. morphZ outputs written to {Path(result.outdir) / 'morphZ'}")


if __name__ == "__main__":
    main()
