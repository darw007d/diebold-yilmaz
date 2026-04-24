"""diebold-yilmaz — Diebold-Yilmaz (2012) connectedness via generalized FEVD.

Public API:

    from diebold_yilmaz import (
        ma_coefficients,       # VAR coefficients → MA representation
        generalized_fevd,      # (psi, sigma) → row-normalised (N, N) table
        connectedness,         # one-call summary: total/from/to/net/pairwise
        SpilloverResult,
        fit_var,               # optional: statsmodels wrapper
    )

    # Already have VAR? Skip fit_var.
    psi = ma_coefficients([A1, A2], horizon=10)   # or use statsmodels' ma_rep
    result = connectedness(psi, sigma_residuals, horizon=10, variable_names=["AAPL", "MSFT", ...])
    result.total_index   # Diebold-Yilmaz Total Spillover Index (%)
    result.net           # per-variable NET spillover (to - from) (%)
"""

from diebold_yilmaz.core import (
    SpilloverResult,
    connectedness,
    generalized_fevd,
    ma_coefficients,
)

try:
    from diebold_yilmaz.fit import fit_var  # requires statsmodels (optional)
except ImportError:  # pragma: no cover
    fit_var = None  # type: ignore

__all__ = [
    "ma_coefficients",
    "generalized_fevd",
    "connectedness",
    "SpilloverResult",
    "fit_var",
]
__version__ = "0.1.0"
