"""Optional convenience: fit a VAR with statsmodels and extract (Psi, Sigma).

statsmodels is a soft dependency — only required if you use `fit_var`. Users
who already have VAR outputs (from custom code, R, etc.) should call the
core functions in `diebold_yilmaz.core` directly.
"""

from __future__ import annotations

import numpy as np


def fit_var(
    returns: np.ndarray,
    p: int | None = None,
    ic: str = "aic",
    horizon: int = 10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit a VAR(p) and extract MA coefficients + residual covariance.

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Time series of observations.
    p : int or None
        Lag order. If None, selected automatically by `ic`.
    ic : {"aic", "bic", "hqic"}
        Information criterion for automatic lag selection.
    horizon : int
        Number of MA coefficients to produce.

    Returns
    -------
    psi : (horizon, N, N) MA coefficients
    sigma : (N, N) residual covariance
    p_used : int, the actual lag order selected.
    """
    try:
        from statsmodels.tsa.vector_ar.var_model import VAR
    except ImportError as e:
        raise ImportError(
            "fit_var requires statsmodels. Install with "
            "`pip install diebold-yilmaz[statsmodels]`."
        ) from e

    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2-D (T, N), got shape {returns.shape}")
    model = VAR(returns)
    if p is None:
        res = model.fit(maxlags=None, ic=ic, trend="c")
    else:
        res = model.fit(p, trend="c")
    p_used = int(res.k_ar)
    psi = np.asarray(res.ma_rep(maxn=horizon - 1), dtype=np.float64)
    # statsmodels' ma_rep returns psi_0..psi_{maxn}, inclusive ⇒ length maxn+1
    # but some versions return maxn. Normalise to horizon rows.
    if psi.shape[0] < horizon:
        # pad with zeros (shouldn't happen for valid VAR but guard)
        pad = np.zeros((horizon - psi.shape[0], psi.shape[1], psi.shape[2]))
        psi = np.concatenate([psi, pad], axis=0)
    psi = psi[:horizon]
    sigma = np.asarray(res.sigma_u, dtype=np.float64)
    return psi, sigma, p_used
