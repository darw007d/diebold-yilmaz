"""Rolling-window + block-bootstrap + community-aggregation extensions (v0.2+).

Builds on the static `connectedness()` API in `diebold_yilmaz.core` to provide:

- `rolling_connectedness(returns, window, step, ...)`  — the canonical
  Diebold-Yilmaz **2014** time-varying form (slide a fixed-length window
  over the returns matrix, refit VAR + recompute connectedness per window,
  return a time series).
- `bootstrap_connectedness_ci(returns, n_boot, block_len, ...)` — moving-
  block bootstrap (Künsch 1989) preserving short-range time-series
  structure; returns confidence bands on `total_index` + per-variable `net`.
- `aggregate_by_group(result, groups)` — collapse the (N×N) connectedness
  table to a (K×K) group-level table given a list of K-valued labels.
  Useful when N=975 tickers and you want sector-level spillover.

All three sit on top of `core.connectedness` — no new math, just useful
canonical extensions of the static computation that v0.1 shipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from diebold_yilmaz.core import SpilloverResult, connectedness


# =====================================================================
# Rolling-window connectedness (DY 2014 time-varying)
# =====================================================================

@dataclass(frozen=True)
class RollingConnectednessResult:
    """Per-window time series of Diebold-Yilmaz spillover measures."""

    window_end_indices: np.ndarray   # (T_w,) ints — END index of each window in `returns`
    total_index: np.ndarray          # (T_w,) percent
    from_others: np.ndarray          # (T_w, N) percent
    to_others: np.ndarray            # (T_w, N) percent
    net: np.ndarray                  # (T_w, N) percent
    tables: Optional[np.ndarray] = None  # (T_w, N, N) percent if return_tables=True
    horizon: int = 0
    window: int = 0
    step: int = 0
    p: int = 0
    n_failed: int = 0                # windows skipped because VAR fit failed


def rolling_connectedness(
    returns: np.ndarray,
    window: int,
    step: int = 1,
    p: Optional[int] = None,
    horizon: int = 10,
    ic: str = "aic",
    return_tables: bool = False,
    skip_failures: bool = True,
) -> RollingConnectednessResult:
    """Sliding-window Diebold-Yilmaz connectedness over time (DY 2014 form).

    For each window of length `window` shifted by `step`, fits a VAR + computes
    the Diebold-Yilmaz spillover summary, returning the per-window time series.

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Time series of N variables over T observations.
    window : int
        Window length (must be > 2*N to leave residual DOF for VAR fit).
    step : int
        Stride between consecutive windows. step=1 ⇒ daily roll; step=window
        ⇒ non-overlapping windows.
    p : int or None
        Fixed VAR lag order. None ⇒ select per-window via `ic`.
    horizon : int
        FEVD horizon for `connectedness`.
    ic : {"aic", "bic", "hqic"}
        Information criterion for per-window lag selection (when p is None).
    return_tables : bool
        Whether to retain the full (N, N) connectedness table per window.
        Memory cost: T_w * N * N * 8 bytes. Default False to keep memory
        bounded for long series with large N.
    skip_failures : bool
        If True, windows where the VAR fit fails (singular sigma, non-stable
        VAR, etc.) are skipped silently and the per-window arrays carry only
        successful windows. The `n_failed` field reports the count. Set
        False to raise on the first failure.

    Returns
    -------
    RollingConnectednessResult
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2-D (T, N), got shape {returns.shape}")
    T, N = returns.shape
    if window < 3:
        raise ValueError(f"window must be >= 3, got {window}")
    if window > T:
        raise ValueError(f"window={window} exceeds T={T}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    try:
        from diebold_yilmaz.fit import fit_var
    except ImportError as e:
        raise ImportError(
            "rolling_connectedness requires the optional statsmodels extra. "
            "Install with `pip install diebold-yilmaz[statsmodels]`."
        ) from e

    starts = list(range(0, T - window + 1, step))
    end_idx, total, from_, to_, net = [], [], [], [], []
    tables = [] if return_tables else None
    n_failed = 0

    for s in starts:
        chunk = returns[s : s + window]
        try:
            psi, sigma, _ = fit_var(chunk, p=p, ic=ic, horizon=horizon)
            res = connectedness(psi, sigma, horizon=horizon)
        except Exception:
            n_failed += 1
            if skip_failures:
                continue
            raise
        end_idx.append(s + window - 1)
        total.append(res.total_index)
        from_.append(res.from_others)
        to_.append(res.to_others)
        net.append(res.net)
        if tables is not None:
            tables.append(res.table)

    return RollingConnectednessResult(
        window_end_indices=np.asarray(end_idx, dtype=np.int64),
        total_index=np.asarray(total, dtype=np.float64),
        from_others=np.asarray(from_, dtype=np.float64) if from_ else np.zeros((0, N)),
        to_others=np.asarray(to_, dtype=np.float64) if to_ else np.zeros((0, N)),
        net=np.asarray(net, dtype=np.float64) if net else np.zeros((0, N)),
        tables=(np.asarray(tables, dtype=np.float64) if tables else None),
        horizon=horizon,
        window=window,
        step=step,
        p=p if p is not None else 0,
        n_failed=n_failed,
    )


# =====================================================================
# Block-bootstrap CI
# =====================================================================

@dataclass(frozen=True)
class BootstrapConnectednessCI:
    """Bootstrap confidence bands for Diebold-Yilmaz spillover measures."""

    point_estimate: SpilloverResult
    total_index_ci: tuple[float, float]      # (lower, upper) at requested level
    total_index_samples: np.ndarray          # (n_boot,) raw bootstrap draws
    net_ci: np.ndarray                       # (N, 2) lower/upper per variable
    net_samples: np.ndarray                  # (n_boot, N) raw bootstrap draws
    confidence_level: float = 0.95
    n_boot: int = 0
    block_len: int = 0
    n_failed: int = 0


def bootstrap_connectedness_ci(
    returns: np.ndarray,
    n_boot: int = 500,
    block_len: Optional[int] = None,
    confidence_level: float = 0.95,
    p: Optional[int] = None,
    horizon: int = 10,
    ic: str = "aic",
    rng: Optional[np.random.Generator] = None,
) -> BootstrapConnectednessCI:
    """Moving-block bootstrap CIs on Diebold-Yilmaz connectedness measures.

    The moving-block bootstrap (Künsch 1989) resamples contiguous BLOCKS of
    observations to preserve short-range time-series dependence, then refits
    the VAR + computes connectedness per resample. Returns percentile bands
    on the total spillover index and per-variable net spillovers.

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Time series. Must be long enough that T // block_len >= 2.
    n_boot : int
        Number of bootstrap resamples. 500 is a reasonable default for
        90/95% CIs; 1000+ for tail (99%) bands.
    block_len : int or None
        Block length. If None, set to ⌊T^(1/3)⌋ (Hall-Horowitz-Jing 1995
        rule of thumb for the moving-block bootstrap on weakly-dependent
        series). For VAR-residual-style data with short memory, T^(1/3)
        is a reasonable default; consider longer for highly autocorrelated
        series.
    confidence_level : float in (0, 1)
        Two-sided percentile bands. 0.95 ⇒ (2.5%, 97.5%) percentiles.
    p, horizon, ic : passed through to `fit_var` / `connectedness`.
    rng : numpy Generator, optional
        For reproducibility.

    Returns
    -------
    BootstrapConnectednessCI
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2-D (T, N), got shape {returns.shape}")
    T, N = returns.shape
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2, got {n_boot}")
    if not (0 < confidence_level < 1):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    if block_len is None:
        block_len = max(1, int(np.floor(T ** (1.0 / 3.0))))
    if block_len < 1:
        raise ValueError(f"block_len must be >= 1, got {block_len}")
    if T // block_len < 2:
        raise ValueError(
            f"T={T} too short for block_len={block_len}; need at least 2 blocks"
        )

    rng = rng if rng is not None else np.random.default_rng()

    try:
        from diebold_yilmaz.fit import fit_var
    except ImportError as e:
        raise ImportError(
            "bootstrap_connectedness_ci requires the optional statsmodels extra. "
            "Install with `pip install diebold-yilmaz[statsmodels]`."
        ) from e

    # Point estimate on the original sample
    psi_pt, sigma_pt, _ = fit_var(returns, p=p, ic=ic, horizon=horizon)
    point = connectedness(psi_pt, sigma_pt, horizon=horizon)

    # Pre-build all valid block starts: 0..T-block_len inclusive
    max_start = T - block_len
    n_blocks_per_resample = int(np.ceil(T / block_len))

    total_samples = []
    net_samples = []
    n_failed = 0

    for _ in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks_per_resample)
        # Concatenate blocks
        rs = np.concatenate(
            [returns[s : s + block_len] for s in starts], axis=0
        )[:T]
        try:
            psi_b, sigma_b, _ = fit_var(rs, p=p, ic=ic, horizon=horizon)
            res_b = connectedness(psi_b, sigma_b, horizon=horizon)
        except Exception:
            n_failed += 1
            continue
        total_samples.append(res_b.total_index)
        net_samples.append(res_b.net)

    if not total_samples:
        raise RuntimeError(
            f"all {n_boot} bootstrap resamples failed to fit a VAR; "
            f"input may be near-singular or block_len={block_len} unsuitable"
        )

    total_arr = np.asarray(total_samples, dtype=np.float64)
    net_arr = np.asarray(net_samples, dtype=np.float64)

    alpha = 1.0 - confidence_level
    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    total_ci = (
        float(np.percentile(total_arr, lo_pct)),
        float(np.percentile(total_arr, hi_pct)),
    )
    net_ci = np.column_stack([
        np.percentile(net_arr, lo_pct, axis=0),
        np.percentile(net_arr, hi_pct, axis=0),
    ])

    return BootstrapConnectednessCI(
        point_estimate=point,
        total_index_ci=total_ci,
        total_index_samples=total_arr,
        net_ci=net_ci,
        net_samples=net_arr,
        confidence_level=confidence_level,
        n_boot=n_boot,
        block_len=block_len,
        n_failed=n_failed,
    )


# =====================================================================
# Community / group aggregation
# =====================================================================

@dataclass(frozen=True)
class GroupedConnectednessResult:
    """Connectedness collapsed onto a coarser group partition (sectors etc.)."""

    group_table: np.ndarray          # (K, K) percent — within-group on diagonal
    group_total_index: float         # percent — fraction of mass off the diagonal
    group_from: np.ndarray           # (K,) percent
    group_to: np.ndarray             # (K,) percent
    group_net: np.ndarray            # (K,) percent
    group_net_pairwise: np.ndarray   # (K, K) percent
    group_labels: list[str]
    n_per_group: np.ndarray          # (K,) ints — variables per group


def aggregate_by_group(
    result: SpilloverResult,
    groups: list[str] | np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> GroupedConnectednessResult:
    """Collapse an (N×N) connectedness table onto group labels.

    Useful when N is large (e.g. 975 tickers) and the raw table is too dense
    to interpret directly — aggregating to K sectors (or K Louvain
    communities) gives a tractable K×K group-level spillover view.

    Aggregation rule: cell (a, b) of the group table is the SUM of
    connectedness from variables in group b INTO variables in group a, NOT
    averaged. Diagonal entries are the sum of within-group connectedness.
    Total mass is preserved (sum of group_table == sum of result.table).

    `weights` lets callers pre-compute capitalization-weighted variants
    (multiply each cell `result.table[i, j]` by `weights[i]` before summing).
    Default None ⇒ unit weights (raw connectedness mass).

    Parameters
    ----------
    result : SpilloverResult
        Output of `connectedness(...)`.
    groups : list[str] of length N OR ndarray of length N
        Group label per variable. Order matches `result.table` row/column.
    weights : ndarray of shape (N,) or None
        Optional per-variable weight (e.g. market cap). None ⇒ unweighted.

    Returns
    -------
    GroupedConnectednessResult
    """
    table = np.asarray(result.table, dtype=np.float64)
    N = table.shape[0]
    groups = list(groups)
    if len(groups) != N:
        raise ValueError(f"groups length {len(groups)} != N={N}")

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != N:
            raise ValueError(f"weights length {weights.shape[0]} != N={N}")
        # Multiply each row by its variable's weight before grouping
        weighted_table = table * weights[:, np.newaxis]
    else:
        weighted_table = table

    # Determine ordered group labels (preserve first-occurrence order — gives
    # readable output for sector lists like ["Tech", "Tech", "Banks", "Tech"])
    group_order: list[str] = []
    seen: set[str] = set()
    for g in groups:
        if g not in seen:
            seen.add(g)
            group_order.append(g)
    K = len(group_order)
    label_to_idx = {g: i for i, g in enumerate(group_order)}
    group_idx = np.array([label_to_idx[g] for g in groups], dtype=np.int64)

    # Build the (K, K) aggregated table by summation
    group_table = np.zeros((K, K), dtype=np.float64)
    for i in range(N):
        gi = group_idx[i]
        for j in range(N):
            gj = group_idx[j]
            group_table[gi, gj] += weighted_table[i, j]

    # Variable counts per group
    n_per_group = np.bincount(group_idx, minlength=K)

    # Group-level summary (same shape as SpilloverResult fields)
    diag_mask = np.eye(K, dtype=bool)
    off_sum = np.where(diag_mask, 0.0, group_table).sum()
    group_total_index = float(off_sum / K) if K > 0 else 0.0
    group_from = group_table.sum(axis=1) - np.diag(group_table)
    group_to = group_table.sum(axis=0) - np.diag(group_table)
    group_net = group_to - group_from
    group_net_pairwise = group_table - group_table.T

    return GroupedConnectednessResult(
        group_table=group_table,
        group_total_index=group_total_index,
        group_from=group_from,
        group_to=group_to,
        group_net=group_net,
        group_net_pairwise=group_net_pairwise,
        group_labels=group_order,
        n_per_group=n_per_group,
    )
