"""
Plot the two-capital-with-shocks FD solution (and NN overlay when available).
Panels: i^d, i^g, marginal values q_d & q_g, aggregate productivity Abar(Z),
consumption/output C/Y, and the value slope v'(Z).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import two_capital_shock_model as M
from fd_shock import solve_fd_shock

P = M.load_calibration("A_g_prime_prime")
OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OD, exist_ok=True)


def augment(d):
    Z = d["Z"]
    d["q_d"] = 1.0 - Z * d["slope"]
    d["q_g"] = 1.0 + (1.0 - Z) * d["slope"]
    d["Abar"] = (1.0 - Z) * P["A_d"] + Z * P["A_g"]
    d["C_over_Y"] = d["c"] / d["Abar"]
    return d


fd = augment(solve_fd_shock(P, n=1500))
print(f"FD-shock: iters={fd['iters']}  max|resid|={fd['max_abs_residual']:.2e}")

# Optional NN overlay if a saved eval exists
nn = None
nn_npz = os.path.join(OD, "nn_eval_shock_A_g_prime_prime.npz")
if os.path.exists(nn_npz):
    z = np.load(nn_npz)
    nn = augment({k: z[k] for k in z.files})

plo, phi = 0.10, 0.90


def pts(o, key, n=33):
    m = (o["Z"] >= plo) & (o["Z"] <= phi)
    Z, y = o["Z"][m], o[key][m]
    s = max(1, len(Z) // n)
    return Z[::s], y[::s]


fig, ax = plt.subplots(2, 3, figsize=(16, 9))


def cmp(a, key, title, ylab, color="b", style="-o"):
    zf, yf = pts(fd, key)
    a.plot(zf, yf, color + style, ms=4, lw=1.5, label="FD")
    if nn is not None:
        zn, yn = pts(nn, key)
        a.plot(zn, yn, "r--s", ms=4, lw=1.5, label="NN")
    a.set_xlabel("Z (green capital share)"); a.set_ylabel(ylab)
    a.set_title(title); a.legend(); a.grid(alpha=0.3)


cmp(ax[0, 0], "i_d", r"Dirty investment rate $i^d$", r"$i^d$")
cmp(ax[0, 1], "i_g", r"Green investment rate $i^g$", r"$i^g$")

# marginal values panel: q_d and q_g
a = ax[0, 2]
zd, qd = pts(fd, "q_d"); zg, qg = pts(fd, "q_g")
a.plot(zd, qd, "b-o", ms=4, lw=1.5, label=r"$q_d$ (dirty), FD")
a.plot(zg, qg, "g-^", ms=4, lw=1.5, label=r"$q_g$ (green), FD")
if nn is not None:
    znd, nqd = pts(nn, "q_d"); zng, nqg = pts(nn, "q_g")
    a.plot(znd, nqd, "r--s", ms=3, lw=1.2, label=r"$q_d$ NN")
    a.plot(zng, nqg, "m--d", ms=3, lw=1.2, label=r"$q_g$ NN")
a.set_xlabel("Z (green capital share)"); a.set_ylabel("marginal value")
a.set_title(r"Marginal values of capital $q_d=1-Zv'$, $q_g=1+(1-Z)v'$")
a.legend(); a.grid(alpha=0.3)

ax[1, 0].plot(fd["Z"], fd["Abar"], "k-", lw=2)
ax[1, 0].set_xlabel("Z"); ax[1, 0].set_ylabel(r"$\bar A(Z)$")
ax[1, 0].set_title(r"Aggregate productivity $\bar A(Z)$"); ax[1, 0].grid(alpha=0.3)
cmp(ax[1, 1], "C_over_Y", "Consumption / output  C/Y", "C/Y")
cmp(ax[1, 2], "slope", r"Value slope $v'(Z)$", r"$v'(Z)$")

fig.suptitle(f"Two-capital WITH shocks: A_d={P['A_d']}, A_g={P['A_g']}, "
             f"sigma_d=sigma_g={P['sigma_d']}", fontsize=13)
fig.tight_layout()
path = os.path.join(OD, "two_capital_shock_A_g_prime_prime.png")
fig.savefig(path, dpi=150)
print("saved", path)
