"""FD solution of the two-capital-with-shocks model across several sigma values,
overlaid. Panels: i^d, i^g, marginal values q_d, q_g, value slope v'(Z), C/Y."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import two_capital_shock_model as M
from fd_shock import solve_fd_shock

SIGMAS = [0.01, 0.1, 0.2]
COLORS = ["b", "g", "r"]
OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OD, exist_ok=True)

sols = []
for s in SIGMAS:
    P = M.load_calibration("A_g_prime_prime")
    P["sigma_d"] = s; P["sigma_g"] = s
    o = solve_fd_shock(P, n=1200)
    Z = o["Z"]
    o["q_d"] = 1.0 - Z * o["slope"]
    o["q_g"] = 1.0 + (1.0 - Z) * o["slope"]
    o["Abar"] = (1.0 - Z) * P["A_d"] + Z * P["A_g"]
    o["C_over_Y"] = o["c"] / o["Abar"]
    sols.append(o)
    print(f"sigma={s}: iters={o['iters']} max|resid|={o['max_abs_residual']:.2e}")

plo, phi = 0.10, 0.90


def pts(o, key, n=33):
    m = (o["Z"] >= plo) & (o["Z"] <= phi)
    Z, y = o["Z"][m], o[key][m]
    st = max(1, len(Z) // n)
    return Z[::st], y[::st]


fig, ax = plt.subplots(2, 3, figsize=(16, 9))
panels = [("i_d", r"Dirty investment $i^d$", r"$i^d$"),
          ("i_g", r"Green investment $i^g$", r"$i^g$"),
          ("q_d", r"Marginal value dirty $q_d=1-Zv'$", r"$q_d$"),
          ("q_g", r"Marginal value green $q_g=1+(1-Z)v'$", r"$q_g$"),
          ("slope", r"Value slope $v'(Z)$", r"$v'$"),
          ("C_over_Y", r"Consumption/output $C/Y$", "C/Y")]
for a, (key, title, ylab) in zip(ax.ravel(), panels):
    for o, s, col in zip(sols, SIGMAS, COLORS):
        z, y = pts(o, key)
        a.plot(z, y, col + "-o", ms=3.5, lw=1.4, label=f"sigma={s}")
    a.set_xlabel("Z (green capital share)"); a.set_ylabel(ylab)
    a.set_title(title); a.legend(); a.grid(alpha=0.3)

fig.suptitle("Two-capital WITH shocks: FD across sigma (A_d=0.1303, A_g=0.1567)", fontsize=13)
fig.tight_layout()
path = os.path.join(OD, "shock_FD_multisigma.png")
fig.savefig(path, dpi=150)
print("saved", path)
