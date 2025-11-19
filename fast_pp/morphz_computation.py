import bilby 
from pp_setup import load_simulation
import pandas as pd
import numpy as np
import os
from morphZ import evidence as morphz_evidence



def get_morphz_evidence(result: bilby.result.Result,
                        priors:bilby.prior.PriorDict,
                        likelihood:bilby.likelihood.Likelihood,
                        label: str = ""
                        ) -> dict:
    posterior = result.posterior
    param_names = list(priors.keys())
    fixed_params = priors.fixed_keys
    search_params = [p for p in param_names if p not in fixed_params] 
    morph_priors = {p: priors[p] for p in search_params}  
    morph_priors = bilby.prior.PriorDict(morph_priors)
    # remove 'mass_1' and 'mass_2' if 'chirp_mass' and 'mass_ratio' are present
    if 'chirp_mass' in search_params and 'mass_ratio' in search_params:
        search_params = [p for p in search_params if p not in ['mass_1', 'mass_2']]
    fixed_param_vals = {p: priors[p].peak for p in fixed_params}
    # remove the priors with fixed params
    
    print(f"Computing morphZ evidence for parameters: {search_params}")
    print("morph-prior:")
    for p in morph_priors:
        print(f"  {p}: {morph_priors[p]}")

    samples = posterior[search_params].to_numpy()
    log_likelihoods = posterior["log_likelihood"].to_numpy()
    log_priors = posterior["log_prior"].to_numpy()
    log_posterior_values = log_likelihoods + log_priors

    print(f"Number of posterior samples: {samples.shape[0]}")
    print("Number of params:", samples.shape[1])

    def log_posterior(theta: np.ndarray) -> float:
        params = dict(zip(search_params, theta))
        log_prior = morph_priors.ln_prob(params)
        params.update(fixed_param_vals)
        if not np.isfinite(log_prior):
            return log_prior
        likelihood.parameters.update(params)
        log_likelihood = likelihood.log_likelihood(params)
        return log_likelihood + log_prior


    # sanity check - compare computed log posterior values with stored ones for ~100 random samples
    random_indices = np.random.choice(len(samples), size=min(100, len(samples)), replace=False)
    test_samples = samples[random_indices]
    test_log_posteriors = log_posterior_values[random_indices]
    computed_log_posteriors = np.array([log_posterior(sample) for sample in test_samples])  
    if not np.allclose(test_log_posteriors, computed_log_posteriors, atol=1e-6):
        print("WARNING: Log posterior function does not match stored values.")
        print("Max difference:", np.max(np.abs(test_log_posteriors - computed_log_posteriors))) 

    
    morphz_runs = morphz_evidence(
        post_samples=samples,
        log_posterior_values=log_posterior_values,
        log_posterior_function=log_posterior,
        n_resamples=1000,
        morph_type='pair',
        kde_bw='silverman',
        param_names=search_params,
        output_path=f"{result.outdir}/morphZ_{label}",
        n_estimations=10,
        verbose=True,
    )
    return morphz_runs


def collect_lnz(idx):
    # Load simulation data
    (
        likelihood, priors, outdir, label, _
    ) = load_simulation(idx)

    # load the two sets of results
    result_dynesty = bilby.result.read_in_result(outdir / f"dynesty_result.json")
    result_mcmc = bilby.result.read_in_result(outdir / f"mcmc_result.json")

    # compute morphz evidence for both
    morphz_dynesty = get_morphz_evidence(result_dynesty, priors, likelihood)
    morphz_mcmc = get_morphz_evidence(result_mcmc, priors, likelihood)

    # write LnZs to a csv
    lnz_data = {
        'method': ['dynesty', 'mcmc', 'dynesty_morphz', 'mcmc_morphz'],
        'lnz': [
            result_dynesty.log_evidence,
            result_mcmc.log_evidence,
            morphz_dynesty['lnz_mean'],
            morphz_mcmc['lnz_mean']
        ]
    }
    lnz_df = pd.DataFrame(lnz_data)
    lnz_df.to_csv(outdir / f"{label}_lnz_comparison.csv", index=False)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python compute_morphz_evidence.py <injection_index>")
        sys.exit(1)

    injection_index = int(sys.argv[1])
    collect_lnz(injection_index)

