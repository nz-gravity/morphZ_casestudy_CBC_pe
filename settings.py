"""
Centralized sampler settings for Dynesty and Bilby-MCMC runs.

Import `get_sampler_settings()` and pass the returned dicts directly to
`bilby.run_sampler` to keep configuration consistent across scripts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SamplerMode:
    """Container for the per-sampler keyword dictionaries."""

    dynesty: Dict[str, Any]
    bilby_mcmc: Dict[str, Any]

    def as_dynesty_kwargs(self) -> Dict[str, Any]:
        return deepcopy(self.dynesty)

    def as_bilby_mcmc_kwargs(self) -> Dict[str, Any]:
        return deepcopy(self.bilby_mcmc)


PROD_SETTINGS = SamplerMode(
    dynesty=dict(
        nlive=1000,
        nact=50,
    ),
    bilby_mcmc=dict(
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
)

# QUICK mode sacrifices resolution for turn-around time; the aim is to triage runs
# or debug plumbing with lower computational cost while remaining representative.
QUICK_SETTINGS = SamplerMode(
    dynesty=dict(
        nlive=500,
        nact=10,
    ),
    bilby_mcmc=dict(
        nsamples=1000,
        thin_by_nact=0.2,
        ntemps=1,
        npool=1,
        Tmax_from_SNR=15,
        adapt=True,
        proposal_cycle="gwA",
        L1steps=50,
        L2steps=3,
    ),
)


def get_sampler_settings(quick: bool = False) -> SamplerMode:
    """
    Return the SamplerMode corresponding to the desired fidelity.

    Parameters
    ----------
    quick:
        When True, prefer faster but lower-resolution settings.
    """

    return QUICK_SETTINGS if quick else PROD_SETTINGS
