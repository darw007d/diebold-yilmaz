# diebold-yilmaz

**Diebold-Yilmaz (2012) connectedness** via **generalized** forecast-error variance decomposition. Pure numpy + scipy, no C++ deps.

## What and why

`statsmodels` provides VAR fitting and a FEVD method — but only the **Cholesky-orthogonalized** FEVD, which depends on the (arbitrary) ordering of your variables. The Diebold-Yilmaz (2012, 2014) connectedness framework requires the **generalized** FEVD of **Pesaran & Shin (1998)**, which is invariant to reordering. This library fills that gap.

Given VAR-MA coefficients `Ψ` and residual covariance `Σ`, it produces:

- the full **connectedness table** `C` (row-normalized N × N percent matrix)
- the **Total Spillover Index** `S = mean of off-diagonal C × 100 %`
- per-variable **directional spillovers**: `from_others`, `to_others`, `net = to − from`
- the **net pairwise** matrix `C − Cᵀ`

…all in one call.

## Install

```bash
pip install diebold-yilmaz
```

Python ≥ 3.9, numpy ≥ 1.23, scipy ≥ 1.10. `statsmodels` is **optional** (only needed if you use the `fit_var` convenience).

## Quickstart

```python
import numpy as np
from diebold_yilmaz import connectedness, ma_coefficients

# VAR(1) with variable 0 shocking 1 and 2
A1 = np.array(
    [[0.20, 0.00, 0.00],
     [0.50, 0.20, 0.00],
     [0.40, 0.00, 0.20]]
)
sigma = np.eye(3)

psi = ma_coefficients([A1], horizon=12)
result = connectedness(psi, sigma, variable_names=["SENDER", "RECV_1", "RECV_2"])

print(f"Total spillover index: {result.total_index:.2f}%")
print(f"Net per variable: {dict(zip(result.variable_names, result.net.round(2)))}")
# Total spillover index: 12.43%
# Net per variable: {'SENDER': 37.3, 'RECV_1': -22.0, 'RECV_2': -15.29}
```

## With statsmodels (optional)

```python
pip install diebold-yilmaz[statsmodels]
```

```python
from diebold_yilmaz import fit_var, connectedness

psi, sigma, p_used = fit_var(returns_TN, p=None, ic="aic", horizon=10)
result = connectedness(psi, sigma)
```

If you already have VAR outputs (from R, custom code, or `statsmodels.tsa.vector_ar.var_model.VARResults.ma_rep`), feed them directly to `connectedness` — `statsmodels` is never imported unless you call `fit_var`.

## The math (1-minute version)

Start with a stable VAR(p): `y_t = Σ_k A_k y_{t−k} + ε_t`, `ε ~ N(0, Σ)`. Its moving-average form is

```
y_t = Σ_h Ψ_h ε_{t−h},   Ψ_0 = I,   Ψ_h = Σ_{j=1..min(h,p)} A_j Ψ_{h−j}
```

Pesaran-Shin (1998) generalized FEVD at horizon `H`:

```
θ^g_{ij}(H) = σ_{jj}^{−1} · Σ_{h=0..H−1} (e_i' Ψ_h Σ e_j)²
                         / Σ_{h=0..H−1} (e_i' Ψ_h Σ Ψ_h' e_i)
```

This is **invariant to the ordering** of the variables — unlike the Cholesky FEVD — because it uses generalized impulse responses instead of orthogonalized ones. Row-normalize θ so rows sum to 1 and you get the Diebold-Yilmaz connectedness table.

From there: Total = mean off-diagonal; from_i = row_i off-diagonal sum; to_j = col_j off-diagonal sum; net = to − from.

## Correctness

14 unit tests cover:

- Ψ_0 = I identity, hand-checked MA recursion on a 2-step VAR(2)
- Row-sums equal 1.0 after normalization
- All entries non-negative (FEVD invariant)
- **Ordering invariance** (the whole point of generalized FEVD): reorder variables, FEVD permutes accordingly
- **Scale invariance**: rescale a variable by diag(D), connectedness table is unchanged
- Antisymmetry of net pairwise: `C − Cᵀ = −(Cᵀ − C)`
- Total spillover index lies in [0, 100] %
- Degenerate case: diagonal VAR + diagonal Σ ⇒ zero spillover
- Sender/receiver case: hand-constructed lead-lag VAR produces correctly signed `net`
- Input validation: shape mismatches, non-positive `σ_{jj}`, bad `horizon`

## Roadmap

**v0.2 (planned):**
- Rolling-window connectedness helper (Diebold-Yilmaz 2014 style, for time-varying spillover series)
- Block-bootstrap confidence intervals on `total_index` and `net`
- Community-/cluster-level aggregation on `net_pairwise`

## Authors

- **Pierre Samson** ([@darw007d](https://github.com/darw007d)) — idea, use-case, design decisions
- **Claude Opus** (Anthropic) — implementation and tests

Originally motivated by a quantitative-finance application (volatility-spillover measurement). Sister package to [phawkes](https://pypi.org/project/phawkes/) (Hawkes), [fisherrao](https://pypi.org/project/fisherrao/) (information geometry), and [tailcor](https://pypi.org/project/tailcor/) (tail-contagion decomposition). Same "small, tested, publishable" pattern.

## Citations

- Diebold, F.X. & Yilmaz, K. (2012). *Better to give than to receive: Predictive directional measurement of volatility spillovers.* Int. J. Forecasting 28(1), 57-66.
- Diebold, F.X. & Yilmaz, K. (2014). *On the network topology of variance decompositions: Measuring the connectedness of financial firms.* J. Econometrics 182(1), 119-134.
- Pesaran, M.H. & Shin, Y. (1998). *Generalized impulse response analysis in linear multivariate models.* Economics Letters 58(1), 17-29.

## License

MIT — see [LICENSE](LICENSE).
