#!/usr/bin/env python
"""
ADAPTED FROM https://github.com/bilby-dev/bilby/blob/main/examples/gw_examples/data_examples/GW150914.py
"""
import sys
from pathlib import Path

import bilby
from gwpy.timeseries import TimeSeries

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from compute_morphz_evidence import (
    compute_morphz_evidence,
    write_evidence_summary,
)

logger = bilby.core.utils.logger
outdir = "outdir"
label = "GW150914"
checkpoint_delta_t = 1800  # seconds between checkpoints

# Note you can get trigger times using the gwosc package, e.g.:
# > from gwosc import datasets
# > datasets.event_gps("GW150914")
trigger_time = 1126259462.4
detectors = ["H1", "L1"]
maximum_frequency = 512
minimum_frequency = 20
roll_off = 0.4  # Roll off duration of tukey window in seconds, default is 0.4s
duration = 4  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

cache_dir = Path(outdir) / "cached_data"
cache_dir.mkdir(parents=True, exist_ok=True)


def fetch_cached_timeseries(detector, start, end, kind):
    """Load cached GW data if present; otherwise download and cache it."""
    start_tag = str(start).replace(".", "p")
    end_tag = str(end).replace(".", "p")
    cache_file = cache_dir / f"{detector}_{kind}_{start_tag}_{end_tag}.hdf5"
    if cache_file.exists():
        logger.info(
            "Loading %s data for ifo %s from cache %s",
            kind,
            detector,
            cache_file.name,
        )
        return TimeSeries.read(cache_file)
    logger.info("Downloading %s data for ifo %s", kind, detector)
    data = TimeSeries.fetch_open_data(detector, start, end)
    data.write(cache_file, overwrite=True)
    return data

# We now use gwpy to obtain analysis and psd data and create the ifo_list
ifo_list = bilby.gw.detector.InterferometerList([])
for det in detectors:
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    data = fetch_cached_timeseries(det, start_time, end_time, "analysis")
    ifo.strain_data.set_from_gwpy_timeseries(data)

    psd_data = fetch_cached_timeseries(det, psd_start_time, psd_end_time, "psd")
    psd_alpha = 2 * roll_off / duration
    psd = psd_data.psd(
        fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median"
    )
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value
    )
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency
    ifo_list.append(ifo)

logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)

# We now define the prior.
# We have defined our prior distribution in a local file, GW150914.prior
# The prior is printed to the terminal at run-time.
# You can overwrite this using the syntax below in the file,
# or choose a fixed value by just providing a float value as the prior.
priors = bilby.gw.prior.BBHPriorDict(filename="GW150914.prior")

# Add the geocent time prior
priors["geocent_time"] = bilby.core.prior.Uniform(
    trigger_time - 0.1, trigger_time + 0.1, name="geocent_time"
)

# In this step we define a `waveform_generator`. This is the object which
# creates the frequency-domain strain. In this instance, we are using the
# `lal_binary_black_hole model` source model. We also pass other parameters:
# the waveform approximant and reference frequency and a parameter conversion
# which allows us to sample in chirp mass and ratio rather than component mass
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments={
        "waveform_approximant": "IMRPhenomPv2",
        "reference_frequency": 50,
    },
)

# In this step, we define the likelihood. Here we use the standard likelihood
# function, passing it the data and the waveform generator.
# Note, phase_marginalization is formally invalid with a precessing waveform such as IMRPhenomPv2
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list,
    waveform_generator,
    priors=priors,
    time_marginalization=True,
    phase_marginalization=False,
    distance_marginalization=True,
)

# Finally, we run the sampler. This function takes the likelihood and prior
# along with some options for how to do the sampling and how to save the data
dynesty_result_file = Path(outdir) / f"{label}_result.json"
if dynesty_result_file.exists():
    logger.info("Loading existing Dynesty result from %s", dynesty_result_file)
    result_dynesty = bilby.result.read_in_result(str(dynesty_result_file))
else:
    result_dynesty = bilby.run_sampler(
        likelihood,
        priors,
        sampler="dynesty",
        outdir=outdir,
        label=label,
        nlive=1000,
        nact=50, ## <-- NEW SETTING TO MATCH BILBY-MCMC PAPER BBH-A
        check_point_delta_t=checkpoint_delta_t,
        check_point_plot=True,
        npool=1,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
    )
    result_dynesty.plot_corner()
logger.info(
    "Dynesty LnZ: %.3f +/- %.3f",
    result_dynesty.log_evidence,
    result_dynesty.log_evidence_err,
)
dynesty_morph = compute_morphz_evidence(
    result=result_dynesty,
    likelihood=likelihood,
    priors=priors,
)
logger.info(
    "morphZ LnZ (Dynesty posterior): %.3f +/- %.3f",
    dynesty_morph["logz_estimate"],
    dynesty_morph["error_estimate"],
)


# RUN A SECOND ANALYSIS WITH BILBY-MCMC
# SETTINGS: sampler-kwargs={nsamples=2000, thin_by_nact=0.2, ntemps=1, npool=1, Tmax_from_SNR=20, adapt=True, proposal_cycle='gwA', L1steps=100, L2steps=5}
mcmc_label = label + "_mcmc"
mcmc_result_file = Path(outdir) / f"{mcmc_label}_result.json"
if mcmc_result_file.exists():
    logger.info("Loading existing Bilby-MCMC result from %s", mcmc_result_file)
    result_mcmc = bilby.result.read_in_result(str(mcmc_result_file))
else:
    result_mcmc = bilby.run_sampler(
        likelihood,
        priors,
        sampler="bilby-mcmc",
        outdir=outdir,
        label=mcmc_label,
        sampler_kwargs={
            "nsamples": 2000,
            "thin_by_nact": 0.2,
            "ntemps": 1,
            "npool": 1,
            "Tmax_from_SNR": 20,
            "adapt": True,
            "proposal_cycle": "gwA",
            "L1steps": 100,
            "L2steps": 5,
        },
        check_point_delta_t=checkpoint_delta_t,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    )
    result_mcmc.plot_corner()
mcmc_morph = compute_morphz_evidence(
    result=result_mcmc,
    likelihood=likelihood,
    priors=priors,
)
logger.info(
    "morphZ LnZ (Bilby-MCMC posterior): %.3f +/- %.3f",
    mcmc_morph["logz_estimate"],
    mcmc_morph["error_estimate"],
)

evidence_rows = [
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
        "method": "bilby-mcmc",
        "kind": "morphZ",
        "log_evidence": mcmc_morph["logz_estimate"],
        "log_evidence_err": mcmc_morph["error_estimate"],
        "n_runs": mcmc_morph["n_runs"],
    },
]
summary_path = write_evidence_summary(outdir, evidence_rows)
logger.info("Evidence summary saved to %s", summary_path)
