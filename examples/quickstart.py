"""diebold-yilmaz quickstart.

Fits a small VAR(2) on synthetic data where variable 0 is a clear NET SENDER
(it leads all others by one lag). The Diebold-Yilmaz connectedness table
should flag it as such: net_to > net_from.
"""

import numpy as np

from diebold_yilmaz import connectedness, ma_coefficients

# Construct a sender/receiver VAR by hand:
#   y[t+1] depends on y[t] only through lag-1 loadings.
#   Variable 0 has lag-1 influence on 1 and 2 (rows 1,2, col 0 off-diagonal).
#   Variables 1 and 2 only self-persist.
A1 = np.array(
    [[0.20, 0.00, 0.00],
     [0.50, 0.20, 0.00],   # var 1 <- var 0
     [0.40, 0.00, 0.20]]   # var 2 <- var 0
)
sigma = np.eye(3)

psi = ma_coefficients([A1], horizon=12)
result = connectedness(psi, sigma, variable_names=["SENDER", "RECV_1", "RECV_2"])

print("Diebold-Yilmaz (2012) connectedness summary")
print(f"  horizon: {result.horizon}  variables: {result.variable_names}")
print()
print(f"Total spillover index: {result.total_index:.2f}%")
print()
print("Connectedness table (row = receives FROM, column = sends TO), %:")
for i, name in enumerate(result.variable_names):
    row = "  " + name.ljust(8) + " "
    for j in range(3):
        row += f"{result.table[i, j]:6.2f}  "
    print(row)
print()
print(f"{'':10} from_others    to_others    net")
for i, name in enumerate(result.variable_names):
    print(
        f"  {name:8s}  {result.from_others[i]:+8.2f}%   "
        f"{result.to_others[i]:+8.2f}%   {result.net[i]:+8.2f}%"
    )
