"""diebold-yilmaz — Diebold-Yilmaz connectedness via generalized FEVD.

v0.1: Static (DY 2012) connectedness — `connectedness(psi, sigma, ...)` over
generalized (Pesaran-Shin 1998) FEVD. Optional statsmodels VAR fitter.

v0.2: Time-varying + bootstrap + group-aggregation extensions:
  - `rolling_connectedness(returns, window, step, ...)` — DY **2014**
    sliding-window form, returns time series of total/from/to/net.
  - `bootstrap_connectedness_ci(returns, n_boot, block_len, ...)` —
    moving-block bootstrap (Künsch 1989) confidence bands.
  - `aggregate_by_group(result, groups)` — collapse N×N table to K×K
    sector-/community-level connectedness.

Public API:

    from diebold_yilmaz import (
        # v0.1 static
        ma_coefficients, generalized_fevd, connectedness,
        SpilloverResult, fit_var,
        # v0.2 extensions
        rolling_connectedness, RollingConnectednessResult,
        bootstrap_connectedness_ci, BootstrapConnectednessCI,
        aggregate_by_group, GroupedConnectednessResult,
    )
"""

from diebold_yilmaz.core import (
    SpilloverResult,
    connectedness,
    generalized_fevd,
    ma_coefficients,
)
from diebold_yilmaz.rolling import (
    BootstrapConnectednessCI,
    GroupedConnectednessResult,
    RollingConnectednessResult,
    aggregate_by_group,
    bootstrap_connectedness_ci,
    rolling_connectedness,
)

try:
    from diebold_yilmaz.fit import fit_var  # requires statsmodels (optional)
except ImportError:  # pragma: no cover
    fit_var = None  # type: ignore

__all__ = [
    # v0.1 static
    "ma_coefficients",
    "generalized_fevd",
    "connectedness",
    "SpilloverResult",
    "fit_var",
    # v0.2 extensions
    "rolling_connectedness",
    "RollingConnectednessResult",
    "bootstrap_connectedness_ci",
    "BootstrapConnectednessCI",
    "aggregate_by_group",
    "GroupedConnectednessResult",
]
__version__ = "0.2.0"
