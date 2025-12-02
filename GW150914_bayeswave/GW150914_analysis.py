import argparse
from GW150914_setup import (
    likelihood,
    priors,
    outdir,
    checkpoint_delta_t,
    NPOOL,
    logger,
    bilby,
)

DEFAULT_DYNESTY = dict(nlive=2000, nact=20, sample="rwalk")


def run_dynesty(nlive=DEFAULT_DYNESTY["nlive"], label=None):
    label = label or ("dynesty" if nlive == DEFAULT_DYNESTY["nlive"] else f"dynesty_nlive{nlive}")
    logger.info(
        "Running Dynesty with nlive=%s, nact=%s, sample=%s (label=%s)",
        nlive,
        DEFAULT_DYNESTY["nact"],
        DEFAULT_DYNESTY["sample"],
        label,
    )
    result = bilby.run_sampler(
        likelihood,
        priors,
        sampler="dynesty",
        outdir=outdir,
        label=label,
        nlive=nlive,
        nact=DEFAULT_DYNESTY["nact"],
        sample=DEFAULT_DYNESTY["sample"],
        check_point_delta_t=checkpoint_delta_t,
        check_point_plot=True,
        npool=NPOOL,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
    )
    result.plot_corner()
    logger.info(f"Dynesty LnZ = {result.log_evidence:.3f} ± {result.log_evidence_err:.3f}")


def run_mcmc():
    from GW150914_setup import likelihood, priors, outdir, checkpoint_delta_t, NPOOL, logger, bilby
    label = 'mcmc'
    result_path = f"{outdir}/{label}_result.json"
    logger.info("Running MCMC...")
    result = bilby.run_sampler(
        likelihood,
        priors,
        sampler="bilby_mcmc",
        outdir=outdir,
        label=label,
        nsamples=2000,
        thin_by_nact=0.2,
        ntemps=NPOOL,
        npool=NPOOL,
        Tmax_from_SNR=20,
        adapt=True,
        proposal_cycle="gwA",
        L1steps=100,
        L2steps=5,
        check_point_delta_t=checkpoint_delta_t,
        check_point_plot=True,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
    )
    result.plot_corner()
    logger.info(f"MCMC LnZ = {result.log_evidence:.3f} ± {result.log_evidence_err:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Run GW150914 analysis with specified sampler.")
    parser.add_argument('--sampler', type=str, choices=['dynesty', 'mcmc'], required=True,
                        help='Sampler to use for the analysis: dynesty or mcmc')
    parser.add_argument('--nlive', type=int, default=None,
                        help=f'Dynesty live points (default {DEFAULT_DYNESTY["nlive"]})')
    parser.add_argument('--label', type=str, default=None,
                        help='Optional label override for dynesty runs')
    args = parser.parse_args()
    if args.sampler == 'dynesty':
        run_dynesty(
            nlive=args.nlive or DEFAULT_DYNESTY["nlive"],
            label=args.label,
        )
    elif args.sampler == 'mcmc':
        run_mcmc()
    else:
        logger.error(f"Unknown sampler: {args.sampler}")



if __name__ == "__main__":
    main()
