"""Three-way comparison: FD (ground truth) vs NN forwardnet vs NN dgm (both
FD-supervised). Loads the saved NN eval npz; recomputes FD locally (no TF)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import two_capital_model as M
from reference_solver import solve_newton

P = M.load_calibration("A_g_prime_prime")
OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def augment(d):
    d["Abar"] = (1 - d["Z"]) * P["A_d"] + d["Z"] * P["A_g"]
    d["C_over_Y"] = d["c"] / d["Abar"]
    return d


Zf, vf, pf, sol = solve_newton(P, n=1000)
ps = savgol_filter(pf, 31, 3)
i_d, i_g, c = M.controls(Zf, ps, P)
fd = augment({"Z": Zf, "slope": ps, "i_d": i_d, "i_g": i_g, "c": c, "ratio": i_g / i_d})


def load(name):
    z = np.load(os.path.join(OD, name))
    return augment({k: z[k] for k in z.files})


fn = load("nn_eval_A_g_prime_prime_forwardnet_fdsup.npz")
dg = load("nn_eval_A_g_prime_prime_dgm_fdsup.npz")

plo, phi = 0.10, 0.90
series = [(fd, "b-o", "FD (ground truth)"), (fn, "r--s", "NN forwardnet"), (dg, "g-.^", "NN dgm gated")]


def pts(o, key, n=33):
    m = (o["Z"] >= plo) & (o["Z"] <= phi)
    Z, y = o["Z"][m], o[key][m]
    s = max(1, len(Z) // n)
    return Z[::s], y[::s]


fig, ax = plt.subplots(2, 3, figsize=(16, 9))


def cmp(a, key, title, yl):
    for o, st, lb in series:
        zz, yy = pts(o, key)
        a.plot(zz, yy, st, ms=4, lw=1.5, label=lb)
    a.set_xlabel("Z (green capital share)")
    a.set_ylabel(yl); a.set_title(title); a.legend(); a.grid(alpha=0.3)


cmp(ax[0, 0], "i_d", r"Dirty investment rate $i^d$ vs $Z$", r"$i^d$")
cmp(ax[0, 1], "i_g", r"Green investment rate $i^g$ vs $Z$", r"$i^g$")
cmp(ax[0, 2], "slope", r"Value slope $v'(Z)$", r"$v'(Z)$")
ax[1, 0].plot(fd["Z"], fd["Abar"], "k-", lw=2)
ax[1, 0].set_xlabel("Z"); ax[1, 0].set_ylabel(r"$\bar A(Z)$")
ax[1, 0].set_title(r"Aggregate productivity $\bar A(Z)$"); ax[1, 0].grid(alpha=0.3)
cmp(ax[1, 1], "C_over_Y", "Consumption / output  C/Y", "C/Y")
ax[1, 2].axis("off")   # 6th panel (i^g/i^d ratio) removed

fig.suptitle("Deterministic two-capital: FD vs NN (forwardnet vs dgm gated), FD-supervised", fontsize=13)
fig.tight_layout()
path = os.path.join(OD, "three_way_A_g_prime_prime.png")
fig.savefig(path, dpi=150)
print("saved", path)
for key in ("i_d", "i_g", "slope"):
    zc = np.linspace(0.1, 0.9, 9)
    a = np.interp(zc, fd["Z"], fd[key]); b = np.interp(zc, fn["Z"], fn[key]); c2 = np.interp(zc, dg["Z"], dg[key])
    print(f"max|{key}|: forwardnet={np.max(np.abs(a-b)):.3e}  dgm={np.max(np.abs(a-c2)):.3e}")
