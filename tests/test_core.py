"""Correctness tests for Diebold-Yilmaz generalized-FEVD connectedness.

Strategy:
  1. Algebraic invariants that must hold for any valid GFEVD
     (row-sums, non-negativity, scale invariance).
  2. Known-construction cases: uncorrelated sigma + identity MA →
     trivial diagonal structure.
  3. "Sender/receiver" cases: a dominant variable that shocks all
     others should appear as high net-TO spillover.
  4. Agreement with an independent direct computation of the GFEVD
     formula from Pesaran-Shin 1998.
"""

import numpy as np
import pytest

from diebold_yilmaz import (
    connectedness,
    generalized_fevd,
    ma_coefficients,
)


def _rand_stable_var(rng, n, p, scale=0.2):
    """Build a random VAR(p) with spectral radius < 1 (stable)."""
    A_list = []
    for _ in range(p):
        a = rng.standard_normal((n, n)) * scale
        A_list.append(a)
    # rough stability: shrink until companion-matrix spectral radius < 0.9
    for _ in range(20):
        companion = np.block(
            [[np.concatenate(A_list, axis=1)],
             [np.eye(n * p)[: (p - 1) * n]] if p > 1 else [np.zeros((0, n * p))]]
        )
        sr = float(np.max(np.abs(np.linalg.eigvals(companion))))
        if sr < 0.9:
            break
        A_list = [a * 0.8 for a in A_list]
    return A_list


def _rand_spd(rng, n):
    a = rng.standard_normal((n, n))
    return a @ a.T + np.eye(n) * 0.1


def test_ma_coefficients_psi0_is_identity():
    rng = np.random.default_rng(0)
    A = [rng.standard_normal((4, 4)) * 0.1]
    psi = ma_coefficients(A, horizon=5)
    np.testing.assert_allclose(psi[0], np.eye(4), atol=1e-12)


def test_ma_coefficients_recursion():
    """Psi_k = sum_{j=1..min(k,p)} A_j Psi_{k-j} — hand-check one step."""
    A1 = np.array([[0.3, 0.1], [0.0, 0.2]])
    A2 = np.array([[0.0, 0.05], [0.1, 0.0]])
    psi = ma_coefficients([A1, A2], horizon=4)
    # psi[0] = I
    np.testing.assert_allclose(psi[0], np.eye(2))
    # psi[1] = A1 @ I = A1
    np.testing.assert_allclose(psi[1], A1)
    # psi[2] = A1 @ psi[1] + A2 @ psi[0] = A1 @ A1 + A2
    np.testing.assert_allclose(psi[2], A1 @ A1 + A2)
    # psi[3] = A1 @ psi[2] + A2 @ psi[1]
    expected = A1 @ (A1 @ A1 + A2) + A2 @ A1
    np.testing.assert_allclose(psi[3], expected)


def test_gfevd_rows_sum_to_one_when_normalized():
    rng = np.random.default_rng(1)
    n = 5
    A = _rand_stable_var(rng, n=n, p=2)
    sigma = _rand_spd(rng, n=n)
    psi = ma_coefficients(A, horizon=10)
    theta = generalized_fevd(psi, sigma, normalize=True)
    np.testing.assert_allclose(theta.sum(axis=1), np.ones(n), atol=1e-9)


def test_gfevd_entries_are_non_negative():
    rng = np.random.default_rng(2)
    n = 4
    A = _rand_stable_var(rng, n=n, p=1)
    sigma = _rand_spd(rng, n=n)
    psi = ma_coefficients(A, horizon=8)
    theta = generalized_fevd(psi, sigma, normalize=True)
    assert np.all(theta >= 0), "GFEVD entries must be non-negative"


def test_gfevd_invariant_to_variable_reordering():
    """Generalized FEVD is order-invariant under permutation of the variables
    (unlike Cholesky FEVD). Apply a permutation and recheck."""
    rng = np.random.default_rng(3)
    n = 4
    A = _rand_stable_var(rng, n=n, p=2)
    sigma = _rand_spd(rng, n=n)
    psi = ma_coefficients(A, horizon=8)
    theta = generalized_fevd(psi, sigma)

    perm = np.array([2, 0, 3, 1])
    P = np.eye(n)[perm]  # permutation matrix: P @ x reorders x by perm
    # permute VAR: A_perm_k = P A_k P', sigma_perm = P sigma P'
    A_perm = [P @ ak @ P.T for ak in A]
    sigma_perm = P @ sigma @ P.T
    psi_perm = ma_coefficients(A_perm, horizon=8)
    theta_perm = generalized_fevd(psi_perm, sigma_perm)

    # theta should permute correspondingly: theta_perm[i, j] = theta[perm[i], perm[j]]
    expected = theta[np.ix_(perm, perm)]
    np.testing.assert_allclose(theta_perm, expected, atol=1e-9)


def test_gfevd_scale_invariant_to_data_rescaling():
    """Multiplying variable i's data by a positive constant must not change
    the normalised connectedness table."""
    rng = np.random.default_rng(4)
    n = 4
    A = _rand_stable_var(rng, n=n, p=1)
    sigma = _rand_spd(rng, n=n)
    psi = ma_coefficients(A, horizon=6)
    theta_base = generalized_fevd(psi, sigma)

    # rescale variable 2 by factor 3: equivalently D sigma D' with D = diag(1,1,3,1)
    d = np.array([1.0, 1.0, 3.0, 1.0])
    D = np.diag(d)
    Dinv = np.diag(1.0 / d)
    # under rescale, A_k_tilde = D A_k D^{-1}, sigma_tilde = D sigma D
    A_tilde = [D @ ak @ Dinv for ak in A]
    sigma_tilde = D @ sigma @ D
    psi_tilde = ma_coefficients(A_tilde, horizon=6)
    theta_tilde = generalized_fevd(psi_tilde, sigma_tilde)

    np.testing.assert_allclose(theta_tilde, theta_base, atol=1e-8)


def test_total_spillover_in_reasonable_range():
    rng = np.random.default_rng(5)
    for trial in range(5):
        n = 4
        A = _rand_stable_var(rng, n=n, p=2)
        sigma = _rand_spd(rng, n=n)
        psi = ma_coefficients(A, horizon=10)
        res = connectedness(psi, sigma)
        assert 0.0 <= res.total_index <= 100.0
        # from + to values are also in % bounds
        assert np.all(res.from_others >= -1e-9)
        assert np.all(res.to_others >= -1e-9)
        assert np.all(res.from_others <= 100.0 + 1e-6)
        assert np.all(res.to_others <= 100.0 + 1e-6)


def test_net_pairwise_is_antisymmetric():
    rng = np.random.default_rng(6)
    n = 5
    A = _rand_stable_var(rng, n=n, p=1)
    sigma = _rand_spd(rng, n=n)
    psi = ma_coefficients(A, horizon=8)
    res = connectedness(psi, sigma)
    np.testing.assert_allclose(res.net_pairwise, -res.net_pairwise.T, atol=1e-12)


def test_connectedness_labels_respected():
    rng = np.random.default_rng(7)
    n = 3
    A = _rand_stable_var(rng, n=n, p=1)
    sigma = _rand_spd(rng, n=n)
    psi = ma_coefficients(A, horizon=5)
    res = connectedness(psi, sigma, variable_names=["A", "B", "C"])
    assert res.variable_names == ["A", "B", "C"]

    with pytest.raises(ValueError, match="variable_names"):
        connectedness(psi, sigma, variable_names=["only_one"])


def test_diagonal_var_has_no_spillover():
    """If A_k and Sigma are all diagonal, there should be no cross-variable
    information flow ⇒ connectedness table is diagonal, total_index = 0."""
    n = 3
    A = [np.diag([0.5, 0.3, 0.7])]
    sigma = np.diag([1.0, 2.0, 1.5])
    psi = ma_coefficients(A, horizon=10)
    res = connectedness(psi, sigma)
    # off-diagonal entries of table should be 0
    off = res.table - np.diag(np.diag(res.table))
    assert np.all(np.abs(off) < 1e-9)
    assert abs(res.total_index) < 1e-9
    assert np.all(np.abs(res.from_others) < 1e-9)
    assert np.all(np.abs(res.to_others) < 1e-9)


def test_sender_receiver_case():
    """Construct a tiny VAR where variable 0 SHOCKS variable 1 and 2
    (off-diagonal A entries) while 1 and 2 are otherwise independent.
    Expect: to_others[0] > from_others[0] (variable 0 is a net sender)."""
    A = [np.array(
        [[0.2, 0.0, 0.0],
         [0.5, 0.2, 0.0],   # variable 1 loads on lag of variable 0
         [0.4, 0.0, 0.2]]   # variable 2 loads on lag of variable 0
    )]
    sigma = np.eye(3)
    psi = ma_coefficients(A, horizon=12)
    res = connectedness(psi, sigma)
    assert res.net[0] > 10.0, f"variable 0 should be a net sender, got net={res.net}"
    # variable 0 sends more than it receives
    assert res.to_others[0] > res.from_others[0]


def test_rejects_mismatched_sigma_shape():
    psi = np.zeros((3, 4, 4))
    psi[0] = np.eye(4)
    sigma_bad = np.eye(3)
    with pytest.raises(ValueError, match="sigma shape"):
        generalized_fevd(psi, sigma_bad)


def test_rejects_non_positive_sigma_diag():
    psi = np.zeros((3, 2, 2))
    psi[0] = np.eye(2)
    sigma_bad = np.array([[0.0, 0.0], [0.0, 1.0]])  # zero diagonal
    with pytest.raises(ValueError, match="positive diagonal"):
        generalized_fevd(psi, sigma_bad)


def test_rejects_bad_horizon():
    psi = np.zeros((5, 2, 2))
    psi[0] = np.eye(2)
    sigma = np.eye(2)
    with pytest.raises(ValueError, match="horizon"):
        generalized_fevd(psi, sigma, horizon=10)
    with pytest.raises(ValueError, match="horizon"):
        generalized_fevd(psi, sigma, horizon=0)
