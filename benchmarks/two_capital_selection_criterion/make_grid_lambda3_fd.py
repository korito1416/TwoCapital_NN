"""PI evidence figure, Mike's format: the lambda3 distorted-belief histogram grid
(rows = initializations, columns = xi) with ONE ROW APPENDED at the bottom:
the FD (verified) worst-case belief computed from the five frozen terminal V^l.

Row 1-3: each initialization's own simulated lambda3_weights_distorted (as in the
internal init-comparison grid). Row 4 (NEW): FD belief softmax_l(-V^l/xi) with the
five PIBYS FD solutions evaluated at the same year-60 path state (logK_60, Z_60,
Y = y_upper = 2.5); the pre-jump value cancels in the normalized weight, so the FD
row needs no pre-damage solve and is verifiable (Richardson + analytic limits).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OD = os.path.join(ROOT, "benchmarks", "post_damage_post_tech", "outputs")
REF = os.path.join(ROOT, "output_xiprofile_ref_20260714")
P1M = os.path.join(ROOT, "output_warmstart_1M_20260714")

TAGS = ["0000", "0083", "0167", "0250", "0333"]
LAM3 = np.array([0.0, 1/12, 1/6, 1/4, 1/3])
XIS = [("0.300", "ξ = 0.3", 0.3), ("0.100", "ξ = 0.1", 0.1), ("0.050", "ξ = 0.05", 0.05)]

INITS = [
    ("reference",    [REF]),
    ("warm start A", [f"{P1M}/nber_s1", f"{P1M}/nber_s2"]),
    ("warm start B", [f"{P1M}/perturb_s1", f"{P1M}/perturb_s2"]),
]

def sim(root, xi, name):
    d = os.path.join(root, "SimulationDeterministic", f"SimulationOutputs_ξ_{xi}")
    return np.loadtxt(os.path.join(d, f"{name}.txt"))

def fd_weights(xi_val, lk, z):
    """FD worst-case lambda3 belief at state (lk, z, Y=2.5)."""
    Vl = []
    for t in TAGS:
        d = np.load(os.path.join(OD, f"fd_pdpt_v5_stable_lam3_{t}_xi148.npz"))
        f = RGI((d["logK"], d["Z"], d["Y"]), d["v"], bounds_error=False, fill_value=None)
        Vl.append(float(f((lk, z, 2.5))))
    Vl = np.array(Vl)
    w = np.exp(-(Vl - Vl.min()) / xi_val)
    return w / w.sum()

plt.rcParams.update({"font.size": 12})
nrows = len(INITS) + 1
fig, axes = plt.subplots(nrows, 3, figsize=(15, 3.4 * nrows), sharex=True, sharey=True)
base = np.ones(len(LAM3)) / len(LAM3)
edges = np.linspace(0.0, 1/3, len(LAM3) + 1)

def draw(a, w, label_handles=False):
    a.hist(LAM3, weights=base, bins=edges, color="C3", alpha=0.5, ec="darkgrey",
           label=("baseline" if label_handles else None))
    a.hist(LAM3, weights=w, bins=edges, color="C0", alpha=0.5, ec="darkgrey",
           label=("distorted" if label_handles else None))

for ri, (ilab, roots) in enumerate(INITS):
    for ci, (xi_s, xi_lab, xi_v) in enumerate(XIS):
        a = axes[ri, ci]
        w = np.mean([sim(r, xi_s, "lambda3_weights_distorted") for r in roots], axis=0)
        draw(a, w, label_handles=(ri == 0 and ci == 0))
        if ri == 0: a.set_title(xi_lab, fontsize=13)
    axes[ri, 0].annotate(ilab, xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90,
                         va="center", ha="center", fontsize=14, fontweight="bold")

# ---- the appended FD row: verified worst-case belief at the same year-60 state ----
for ci, (xi_s, xi_lab, xi_v) in enumerate(XIS):
    a = axes[len(INITS), ci]
    t = sim(REF, xi_s, "t"); n = min(len(t), len(sim(REF, xi_s, "logK")))
    m = t[:n] <= 60.0
    lk60 = sim(REF, xi_s, "logK")[:n][m][-1]
    z60 = sim(REF, xi_s, "Z")[:n][m][-1]
    draw(a, fd_weights(xi_v, lk60, z60))
    a.set_xlabel(r"$\lambda_3$")
axes[len(INITS), 0].annotate("FD (verified)", xy=(-0.30, 0.5), xycoords="axes fraction",
                             rotation=90, va="center", ha="center", fontsize=14,
                             fontweight="bold", color="#B22222")

h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=13,
           bbox_to_anchor=(0.5, -0.004))
fig.tight_layout(rect=(0.03, 0.03, 1, 1))
out = os.path.join(HERE, "figures", "grid_lambda3_with_fd.png")
fig.savefig(out, dpi=145, bbox_inches="tight")
print("wrote", out)
