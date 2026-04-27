"""Tests for v0.2 additions: rolling-window, bootstrap CI, group aggregation."""

import numpy as np
import pytest

from diebold_yilmaz import (
    aggregate_by_group,
    bootstrap_connectedness_ci,
    connectedness,
    fit_var,
    rolling_connectedness,
)


# Skip everything if statsmodels isn't available (rolling + bootstrap need it)
statsmodels = pytest.importorskip("statsmodels")


def _stable_var_returns(rng, n, T, p=1, scale=0.15, sigma_scale=0.5):
    """Generate stable VAR(p) returns for testing."""
    A_list = [rng.standard_normal((n, n)) * scale for _ in range(p)]
    # Shrink for stability
    for _ in range(20):
        companion_top = np.concatenate(A_list, axis=1)
        if p > 1:
            companion_bottom = np.eye(n * p)[: (p - 1) * n]
            companion = np.vstack([companion_top, companion_bottom])
        else:
            companion = companion_top
        sr = float(np.max(np.abs(np.linalg.eigvals(companion))))
        if sr < 0.85:
            break
        A_list = [a * 0.8 for a in A_list]

    sigma_chol = np.linalg.cholesky(
        np.eye(n) * sigma_scale + 0.01 * rng.standard_normal((n, n)) ** 2
    )
    eps = rng.standard_normal((T + 50, n)) @ sigma_chol.T  # 50 burn-in
    y = np.zeros((T + 50, n))
    for t in range(p, T + 50):
        for k in range(p):
            y[t] += A_list[k] @ y[t - 1 - k]
        y[t] += eps[t]
    return y[50:]  # drop burn-in


# =====================================================================
# rolling_connectedness
# =====================================================================

def test_rolling_basic_shapes():
    rng = np.random.default_rng(0)
    n, T = 3, 200
    returns = _stable_var_returns(rng, n, T)
    res = rolling_connectedness(returns, window=80, step=20, p=1, horizon=5)

    expected_n_windows = (T - 80) // 20 + 1
    assert len(res.window_end_indices) == expected_n_windows
    assert res.total_index.shape == (expected_n_windows,)
    assert res.from_others.shape == (expected_n_windows, n)
    assert res.to_others.shape == (expected_n_windows, n)
    assert res.net.shape == (expected_n_windows, n)
    # No tables retained by default
    assert res.tables is None


def test_rolling_returns_tables_when_requested():
    rng = np.random.default_rng(1)
    returns = _stable_var_returns(rng, 3, 150)
    res = rolling_connectedness(returns, window=60, step=30, p=1,
                                  horizon=5, return_tables=True)
    n_w = res.total_index.shape[0]
    assert res.tables is not None
    assert res.tables.shape == (n_w, 3, 3)


def test_rolling_window_too_large_rejected():
    rng = np.random.default_rng(2)
    returns = _stable_var_returns(rng, 3, 50)
    with pytest.raises(ValueError, match="exceeds"):
        rolling_connectedness(returns, window=100, step=1)


def test_rolling_step_must_be_positive():
    rng = np.random.default_rng(3)
    returns = _stable_var_returns(rng, 3, 100)
    with pytest.raises(ValueError, match="step must be"):
        rolling_connectedness(returns, window=50, step=0)


def test_rolling_total_index_in_valid_range():
    """Each window's total_index must be a percent in [0, 100*(N-1)/N approx]."""
    rng = np.random.default_rng(4)
    returns = _stable_var_returns(rng, 3, 200)
    res = rolling_connectedness(returns, window=80, step=10, p=1, horizon=5)
    assert np.all(res.total_index >= 0)
    assert np.all(res.total_index <= 100.0)


def test_rolling_window_end_indices_strictly_increasing():
    rng = np.random.default_rng(5)
    returns = _stable_var_returns(rng, 3, 200)
    res = rolling_connectedness(returns, window=60, step=15, p=1, horizon=5)
    assert np.all(np.diff(res.window_end_indices) > 0)


def test_rolling_skip_failures_keeps_going(monkeypatch):
    """When skip_failures=True, fit failures are counted, not raised.

    `rolling_connectedness` does a lazy `from diebold_yilmaz.fit import fit_var`
    at call time, so we patch the source module — that's where the binding
    is read from on every call.
    """
    rng = np.random.default_rng(6)
    returns = _stable_var_returns(rng, 3, 200)

    from diebold_yilmaz import fit as fit_mod
    real_fit = fit_mod.fit_var
    state = {"calls": 0}

    def flaky(*a, **kw):
        state["calls"] += 1
        if state["calls"] % 2 == 0:
            raise RuntimeError("simulated VAR fit failure")
        return real_fit(*a, **kw)

    monkeypatch.setattr(fit_mod, "fit_var", flaky)

    res = rolling_connectedness(returns, window=80, step=20, p=1, horizon=5,
                                skip_failures=True)
    assert res.n_failed > 0


# =====================================================================
# bootstrap_connectedness_ci
# =====================================================================

def test_bootstrap_basic_shapes():
    rng = np.random.default_rng(10)
    n, T = 3, 200
    returns = _stable_var_returns(rng, n, T)
    ci = bootstrap_connectedness_ci(
        returns, n_boot=20, block_len=10, p=1, horizon=5,
        rng=np.random.default_rng(99),
    )
    assert isinstance(ci.point_estimate.total_index, float)
    assert ci.total_index_samples.shape[0] <= 20
    assert ci.net_samples.shape[1] == n
    assert ci.net_ci.shape == (n, 2)
    assert ci.confidence_level == 0.95
    assert ci.n_boot == 20


def test_bootstrap_ci_contains_point_estimate_for_well_specified():
    """For a well-specified VAR with enough data, point estimate should fall
    within the bootstrap CI most of the time."""
    rng = np.random.default_rng(11)
    returns = _stable_var_returns(rng, 3, 400)  # plenty of data
    ci = bootstrap_connectedness_ci(
        returns, n_boot=100, block_len=15, p=1, horizon=5,
        rng=np.random.default_rng(123),
    )
    lo, hi = ci.total_index_ci
    # 95% CI on a well-specified VAR should bracket the point estimate
    pt = ci.point_estimate.total_index
    # Loose tolerance: bootstrap is noisy and point ≠ bootstrap median exactly
    margin = 0.5 * (hi - lo) + 5.0
    assert (lo - margin) <= pt <= (hi + margin)


def test_bootstrap_default_block_len_is_T_cube_root():
    """Default block_len = floor(T^(1/3))."""
    rng = np.random.default_rng(12)
    returns = _stable_var_returns(rng, 3, 1000)
    ci = bootstrap_connectedness_ci(
        returns, n_boot=10, p=1, horizon=5,
        rng=np.random.default_rng(7),
    )
    assert ci.block_len == int(np.floor(1000 ** (1.0 / 3.0)))


def test_bootstrap_rejects_too_few_resamples():
    rng = np.random.default_rng(13)
    returns = _stable_var_returns(rng, 3, 100)
    with pytest.raises(ValueError, match="n_boot must be"):
        bootstrap_connectedness_ci(returns, n_boot=1)


def test_bootstrap_rejects_invalid_confidence_level():
    rng = np.random.default_rng(14)
    returns = _stable_var_returns(rng, 3, 100)
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_connectedness_ci(returns, confidence_level=1.5)


def test_bootstrap_block_len_must_fit():
    """If block_len * 2 > T, error."""
    rng = np.random.default_rng(15)
    returns = _stable_var_returns(rng, 3, 50)
    with pytest.raises(ValueError, match="too short"):
        bootstrap_connectedness_ci(returns, n_boot=10, block_len=40)


def test_bootstrap_reproducible_with_seeded_rng():
    """Same RNG seed ⇒ same bootstrap draws."""
    rng = np.random.default_rng(16)
    returns = _stable_var_returns(rng, 3, 200)

    ci_a = bootstrap_connectedness_ci(
        returns, n_boot=20, block_len=10, p=1, horizon=5,
        rng=np.random.default_rng(42),
    )
    ci_b = bootstrap_connectedness_ci(
        returns, n_boot=20, block_len=10, p=1, horizon=5,
        rng=np.random.default_rng(42),
    )
    np.testing.assert_array_equal(ci_a.total_index_samples, ci_b.total_index_samples)
    np.testing.assert_array_equal(ci_a.net_samples, ci_b.net_samples)


# =====================================================================
# aggregate_by_group
# =====================================================================

def test_aggregate_basic_grouping():
    """Group 4 vars into 2 groups: total mass conserved, K×K shape."""
    rng = np.random.default_rng(20)
    returns = _stable_var_returns(rng, 4, 200)
    psi, sigma, _ = fit_var(returns, p=1, horizon=5)
    res = connectedness(psi, sigma, horizon=5)

    grouped = aggregate_by_group(res, ["A", "A", "B", "B"])
    assert grouped.group_table.shape == (2, 2)
    assert grouped.group_labels == ["A", "B"]
    np.testing.assert_array_equal(grouped.n_per_group, [2, 2])
    # Total mass preserved
    assert grouped.group_table.sum() == pytest.approx(res.table.sum(), rel=1e-9)


def test_aggregate_preserves_label_first_occurrence_order():
    rng = np.random.default_rng(21)
    returns = _stable_var_returns(rng, 4, 150)
    psi, sigma, _ = fit_var(returns, p=1, horizon=5)
    res = connectedness(psi, sigma, horizon=5)

    grouped = aggregate_by_group(res, ["Tech", "Banks", "Tech", "Energy"])
    assert grouped.group_labels == ["Tech", "Banks", "Energy"]
    assert grouped.group_table.shape == (3, 3)
    # Tech has 2 vars, Banks 1, Energy 1
    np.testing.assert_array_equal(grouped.n_per_group, [2, 1, 1])


def test_aggregate_with_weights():
    """Weighted aggregation should change the table values."""
    rng = np.random.default_rng(22)
    returns = _stable_var_returns(rng, 3, 150)
    psi, sigma, _ = fit_var(returns, p=1, horizon=5)
    res = connectedness(psi, sigma, horizon=5)

    g_unw = aggregate_by_group(res, ["A", "B", "B"])
    g_w = aggregate_by_group(res, ["A", "B", "B"],
                              weights=np.array([10.0, 1.0, 1.0]))
    # Group "A" only has var 0, weighted at 10× → its row should be 10× heavier
    assert g_w.group_table[0].sum() == pytest.approx(10 * g_unw.group_table[0].sum(),
                                                       rel=1e-9)


def test_aggregate_groups_length_mismatch():
    rng = np.random.default_rng(23)
    returns = _stable_var_returns(rng, 3, 150)
    psi, sigma, _ = fit_var(returns, p=1, horizon=5)
    res = connectedness(psi, sigma, horizon=5)
    with pytest.raises(ValueError, match="groups length"):
        aggregate_by_group(res, ["A", "B"])  # only 2 labels, N=3


def test_aggregate_weights_length_mismatch():
    rng = np.random.default_rng(24)
    returns = _stable_var_returns(rng, 3, 150)
    psi, sigma, _ = fit_var(returns, p=1, horizon=5)
    res = connectedness(psi, sigma, horizon=5)
    with pytest.raises(ValueError, match="weights length"):
        aggregate_by_group(res, ["A", "B", "B"], weights=np.array([1.0, 2.0]))


def test_aggregate_summary_metrics_consistent():
    rng = np.random.default_rng(25)
    returns = _stable_var_returns(rng, 4, 200)
    psi, sigma, _ = fit_var(returns, p=1, horizon=5)
    res = connectedness(psi, sigma, horizon=5)
    grouped = aggregate_by_group(res, ["X", "X", "Y", "Y"])

    # Per-group from + to summed match off-diagonal mass of group_table
    table = grouped.group_table
    diag = np.diag(table)
    expected_from = table.sum(axis=1) - diag
    expected_to = table.sum(axis=0) - diag
    np.testing.assert_array_almost_equal(grouped.group_from, expected_from)
    np.testing.assert_array_almost_equal(grouped.group_to, expected_to)
    np.testing.assert_array_almost_equal(grouped.group_net, expected_to - expected_from)


def test_aggregate_self_loop_when_one_group():
    """All variables in one group ⇒ 1×1 table = sum of all entries."""
    rng = np.random.default_rng(26)
    returns = _stable_var_returns(rng, 3, 150)
    psi, sigma, _ = fit_var(returns, p=1, horizon=5)
    res = connectedness(psi, sigma, horizon=5)
    grouped = aggregate_by_group(res, ["all"] * 3)
    assert grouped.group_table.shape == (1, 1)
    assert grouped.group_table[0, 0] == pytest.approx(res.table.sum(), rel=1e-9)
    # No off-diagonal mass with K=1 → total_index = 0
    assert grouped.group_total_index == pytest.approx(0.0, abs=1e-9)
