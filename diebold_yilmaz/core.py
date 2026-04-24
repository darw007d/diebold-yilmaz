"""Core Diebold-Yilmaz connectedness computations.

Implements the generalized-FEVD spillover framework of:
- Diebold, F.X. & Yilmaz, K. (2012). "Better to give than to receive:
  Predictive directional measurement of volatility spillovers."
  Int. J. Forecasting 28(1).
- Diebold, F.X. & Yilmaz, K. (2014). "On the network topology of variance
  decompositions: Measuring the connectedness of financial firms."
  J. Econometrics 182(1).

Built on top of the Pesaran-Shin (1998) generalized forecast-error variance
decomposition, which is invariant to the ordering of the variables in the
VAR (unlike the Cholesky-orthogonalized FEVD that most stat libraries
expose — e.g. statsmodels' `VAR.fit().fevd()` uses Cholesky).

Inputs
------
The library works on VAR-MA coefficients + residual covariance. This keeps
it backend-agnostic: fit your VAR with statsmodels / R / whatever, then pass
the outputs here for the spillover math.

MA representation
-----------------
For a VAR(p): y_t = sum_{k=1..p} A_k y_{t-k} + eps_t, eps ~ N(0, Sigma).

The moving-average form is y_t = sum_{k=0..inf} Psi_k eps_{t-k}, with the
recursion Psi_0 = I, Psi_k = sum_{j=1..min(k,p)} A_j Psi_{k-j}.

`ma_coefficients(A_list, H)` computes Psi_0 ... Psi_{H-1} from the VAR
coefficient matrices. Users who already have Psi (e.g. from statsmodels'
`VARResults.ma_rep()`) can skip this helper.

Generalized FEVD
----------------
theta_ij^g(H) = sigma_jj^{-1} * sum_{h=0..H-1} (e_i' Psi_h Sigma e_j)^2
              / sum_{h=0..H-1} (e_i' Psi_h Sigma Psi_h' e_i)

Rows may not sum to 1 under GFEVD; we row-normalize to form the connectedness
table C, per Diebold-Yilmaz 2012 sec 2.2.

Indices
-------
- Total Spillover Index S = (1/N) * sum_{i != j} C_{ij} * 100  [%]
- "From" spillovers: sum_{j != i} C_{ij}  (what others spill INTO i)
- "To"   spillovers: sum_{i != j} C_{ij}  (what i spills TO others)
- "Net"  spillovers: To_i - From_i
- Net pairwise: C_{ij} - C_{ji}
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ma_coefficients(a_list: list[np.ndarray], horizon: int) -> np.ndarray:
    """Compute the MA representation Psi_0..Psi_{H-1} of a VAR(p).

    Parameters
    ----------
    a_list : list of ndarray, length p
        Coefficient matrices A_1..A_p, each shape (N, N).
    horizon : int
        Number of MA steps to compute (H >= 1).

    Returns
    -------
    psi : ndarray of shape (horizon, N, N)
        psi[0] = I, psi[k] = sum_{j=1..min(k,p)} A_j @ psi[k-j].
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if not a_list:
        raise ValueError("a_list must contain at least one VAR lag matrix")
    a_arr = [np.asarray(a, dtype=np.float64) for a in a_list]
    n = a_arr[0].shape[0]
    for i, a in enumerate(a_arr):
        if a.shape != (n, n):
            raise ValueError(
                f"a_list[{i}] has shape {a.shape}, expected ({n}, {n})"
            )
    p = len(a_arr)
    psi = np.zeros((horizon, n, n), dtype=np.float64)
    psi[0] = np.eye(n)
    for k in range(1, horizon):
        for j in range(1, min(k, p) + 1):
            psi[k] += a_arr[j - 1] @ psi[k - j]
    return psi


def generalized_fevd(
    psi: np.ndarray,
    sigma: np.ndarray,
    horizon: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Generalized forecast-error variance decomposition (Pesaran-Shin 1998).

    Parameters
    ----------
    psi : ndarray of shape (H, N, N) OR (H_available, N, N)
        MA coefficient matrices. psi[0] must equal the identity.
    sigma : ndarray of shape (N, N)
        Residual covariance of the VAR.
    horizon : int or None
        Summation horizon. None means use all H rows of psi.
    normalize : bool
        If True, row-normalize so rows sum to 1 (required for the Diebold-
        Yilmaz connectedness table).

    Returns
    -------
    theta : ndarray of shape (N, N)
        theta[i, j] is the fraction of the H-step forecast-error variance of
        variable i attributable to shocks in variable j.
    """
    psi = np.asarray(psi, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if psi.ndim != 3:
        raise ValueError(f"psi must be 3-D (H, N, N), got shape {psi.shape}")
    H_avail, N, N2 = psi.shape
    if N != N2:
        raise ValueError(f"psi last two dims must match, got {psi.shape}")
    if sigma.shape != (N, N):
        raise ValueError(f"sigma shape {sigma.shape} incompatible with N={N}")

    H = H_avail if horizon is None else int(horizon)
    if H < 1 or H > H_avail:
        raise ValueError(f"horizon {H} out of range [1, {H_avail}]")

    sigma_diag = np.diag(sigma)  # shape (N,)
    if np.any(sigma_diag <= 0):
        raise ValueError(f"sigma must be SPD with positive diagonal; got {sigma_diag}")

    # Numerator: sum_{h=0..H-1} (psi[h] @ sigma @ e_j)_i^2 for each (i, j).
    # (psi[h] @ sigma) has shape (N, N); element (i, j) is e_i' psi[h] sigma e_j.
    num = np.zeros((N, N), dtype=np.float64)
    for h in range(H):
        m = psi[h] @ sigma  # (N, N)
        num += m * m
    # Divide columns by sigma_jj
    num = num / sigma_diag[np.newaxis, :]

    # Denominator: sum_{h=0..H-1} (e_i' psi[h] sigma psi[h]' e_i) for each i.
    # This is the diagonal of psi[h] @ sigma @ psi[h]'.
    denom = np.zeros(N, dtype=np.float64)
    for h in range(H):
        m = psi[h] @ sigma @ psi[h].T
        denom += np.diag(m)
    if np.any(denom <= 0):
        raise RuntimeError(f"FEVD denominator non-positive at some variable: {denom}")

    theta = num / denom[:, np.newaxis]  # broadcast rows

    if normalize:
        row_sums = theta.sum(axis=1, keepdims=True)
        theta = theta / row_sums
    return theta


@dataclass(frozen=True)
class SpilloverResult:
    """Bundle of Diebold-Yilmaz (2012) connectedness measures."""

    table: np.ndarray            # (N, N) row-normalised connectedness table
    total_index: float           # scalar in [0, 100] percent
    from_others: np.ndarray      # (N,) percent; sum over row off-diagonal
    to_others: np.ndarray        # (N,) percent; sum over column off-diagonal
    net: np.ndarray              # (N,) percent; to - from
    net_pairwise: np.ndarray     # (N, N) percent; table[i,j] - table[j,i]
    horizon: int
    variable_names: list[str] | None = None


def connectedness(
    psi: np.ndarray,
    sigma: np.ndarray,
    horizon: int | None = None,
    variable_names: list[str] | None = None,
) -> SpilloverResult:
    """Full Diebold-Yilmaz connectedness summary.

    Parameters
    ----------
    psi : (H, N, N) MA coefficients (psi[0] = I).
    sigma : (N, N) residual covariance.
    horizon : summation horizon for FEVD. None = use all psi rows.
    variable_names : optional labels.

    Returns
    -------
    SpilloverResult with percentages (total_index, from_others, to_others,
    net, net_pairwise all in units of %).
    """
    theta = generalized_fevd(psi, sigma, horizon=horizon, normalize=True)
    N = theta.shape[0]
    if variable_names is not None and len(variable_names) != N:
        raise ValueError(f"variable_names length {len(variable_names)} != N={N}")

    # percentages
    table_pct = theta * 100.0
    diag_mask = np.eye(N, dtype=bool)

    # from_i: fraction of i's variance explained by OTHER variables
    from_others = table_pct.sum(axis=1) - np.diag(table_pct)
    # to_j: fraction of OTHERS' variance explained by j
    to_others = table_pct.sum(axis=0) - np.diag(table_pct)
    # total index: average off-diagonal mass
    off_sum = np.where(diag_mask, 0.0, table_pct).sum()
    total_index = float(off_sum / N)

    net_pairwise = table_pct - table_pct.T

    return SpilloverResult(
        table=table_pct,
        total_index=total_index,
        from_others=from_others,
        to_others=to_others,
        net=to_others - from_others,
        net_pairwise=net_pairwise,
        horizon=psi.shape[0] if horizon is None else int(horizon),
        variable_names=list(variable_names) if variable_names else None,
    )
