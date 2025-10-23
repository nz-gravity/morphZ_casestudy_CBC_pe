import json
import os
import sys

import bilby
import numpy as np
from morphZ import evidence as morphz_evidence

# CONSTANTS
duration = 4.0
sampling_frequency = 2048.0
minimum_frequency = 20
injection_prior = bilby.gw.prior.BBHPriorDict()
waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2",
    reference_frequency=50.0,
    minimum_frequency=minimum_frequency,
)
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)



def compute_morphz_evidence(
    *,
    result: bilby.result.Result,
    likelihood: bilby.gw.GravitationalWaveTransient,
    analysis_priors: bilby.gw.prior.BBHPriorDict,
    fixed_parameters: dict,
    n_resamples: int = 2000,
    morph_type: str = "tree",
    kde_bw: str | float = "isj",
    n_estimations: int = 2,
    verbose: bool = True,
) -> dict:
    """Run morphZ evidence estimation and return summary stats."""
    posterior = result.posterior
    if posterior is None or posterior.empty:
        raise RuntimeError("Result contains no posterior samples for morphZ.")
    morphz_param_names = list(result.search_parameter_keys)
    required_columns = {"log_likelihood", "log_prior"}
    if not required_columns.issubset(posterior.columns):
        missing = sorted(required_columns.difference(posterior.columns))
        raise RuntimeError(f"Posterior missing columns needed for morphZ: {missing}")

    missing_params = [name for name in morphz_param_names if name not in posterior.columns]
    if missing_params:
        raise RuntimeError(f"Posterior missing search parameters needed for morphZ: {missing_params}")

    morphz_samples = posterior[morphz_param_names].to_numpy()
    log_posterior_vals = posterior["log_likelihood"].to_numpy() + posterior["log_prior"].to_numpy()

    base_parameters = fixed_parameters.copy()

    def log_posterior(theta: np.ndarray) -> float:
        params = dict(zip(morphz_param_names, theta))
        full_params = base_parameters.copy()
        full_params.update(params)
        log_prior = analysis_priors.ln_prob(full_params)
        if not np.isfinite(log_prior):
            return log_prior
        log_likelihood = likelihood.log_likelihood(full_params)
        return log_likelihood + log_prior

    morphz_output_dir = os.path.join(getattr(result, "outdir", "."), "morphZ")
    os.makedirs(morphz_output_dir, exist_ok=True)
    morphz_results = morphz_evidence(
        post_samples=morphz_samples,
        log_posterior_values=log_posterior_vals,
        log_posterior_function=log_posterior,
        n_resamples=n_resamples,
        morph_type=morph_type,
        kde_bw=kde_bw,
        param_names=morphz_param_names,
        output_path=morphz_output_dir,
        n_estimations=n_estimations,
        verbose=verbose,
    )

    results_array = np.atleast_2d(np.asarray(morphz_results, dtype=float))
    logz_runs = results_array[:, 0]
    err_runs = results_array[:, 1] if results_array.shape[1] > 1 else np.full_like(logz_runs, np.nan)
    logz_estimate = float(np.mean(logz_runs))
    if results_array.shape[1] > 1 and np.isfinite(err_runs).all():
        error_estimate = float(np.sqrt(np.mean(err_runs**2)))
    elif logz_runs.size > 1:
        error_estimate = float(np.std(logz_runs, ddof=1) / np.sqrt(logz_runs.size))
    else:
        error_estimate = float(np.nan)

    summary = {
        "logz_runs": logz_runs.tolist(),
        "error_runs": err_runs.tolist(),
        "logz_estimate": logz_estimate,
        "error_estimate": error_estimate,
        "param_names": morphz_param_names,
        "n_resamples": n_resamples,
        "morph_type": morph_type,
        "kde_bw": kde_bw,
        "n_estimations": n_estimations,
    }

    return summary


def main(seed=88170235):
    outdir = f"outdir/seed_{seed}"
    label = f"seed_{seed}"
    bilby.core.utils.random.seed(seed)
    os.makedirs(outdir, exist_ok=True)

    injection_parameters = injection_prior.sample()
    injection_parameters["geocent_time"] = 2.0


    ifos = bilby.gw.detector.InterferometerList(["H1"]) # Ignore L1 for faster runs during testing
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - 2,
    )
    ifos.inject_signal(
        waveform_generator=waveform_generator, parameters=injection_parameters
    )

    # set delta functions for parameters we want to fix during analysis
    analysis_priors = bilby.gw.prior.BBHPriorDict()
    for key in [
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "psi",
        "ra",
        "dec",
        "geocent_time",
        "phase",
        # some more to make analysis even simpler
        "luminosity_distance",
        "theta_jn",
    ]:
        analysis_priors[key] = injection_parameters[key]
    # Perform a check that the prior does not extend to a parameter space longer than the data
    analysis_priors.validate_prior(duration, minimum_frequency)

    # Initialise the likelihood by passing in the interferometer data (ifos) and
    # the waveform generator
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator
    )

    # Run sampler.  In this case we're going to use the `dynesty` sampler
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=analysis_priors,
        sampler="dynesty",
        npoints=1000,
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=label,
    )

    # Make a corner plot + save to outdir
    result.plot_corner()
    print(f"Nested sampling LnZ: {result.log_evidence} +\- {result.log_evidence_err}")
    morphz_summary = compute_morphz_evidence(
        result=result,
        likelihood=likelihood,
        analysis_priors=analysis_priors,
        fixed_parameters=injection_parameters,
    )
    runs = len(morphz_summary["logz_runs"])
    logz_est = morphz_summary["logz_estimate"]
    err_est = morphz_summary["error_estimate"]
    print(
        f"morphZ LnZ (averaged over {runs} run{'s' if runs != 1 else ''}): "
        f"{logz_est:.3f} +/- {err_est:.3f}"
    )
    evidence_summary = {
        "seed": seed,
        "nested_sampling": {
            "log_evidence": float(result.log_evidence),
            "log_evidence_err": float(result.log_evidence_err),
        },
        "morphz": {
            "log_evidence": morphz_summary["logz_estimate"],
            "log_evidence_err": morphz_summary["error_estimate"],
            "n_runs": len(morphz_summary["logz_runs"]),
        },
    }

    summary_path = os.path.join(outdir, "evidence_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(evidence_summary, fh, indent=2)



if __name__ == "__main__":
    args = sys.argv[1:]
    seed = 0
    if len(args) > 0:
        seed = int(args[0])
    main(seed=seed)
    
