"""Welfare cost of robustness (discounted relative entropy = dV/dxi by the envelope
theorem), decomposed by uncertainty channel, across xi. Stacked bars show the jump
channels dominate and the diffusion channels (esp. climate) vanish."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = os.path.join(ROOT, "output_xiprofile_ref_20260714", "SimulationDeterministic")
XIS = [("0.050", 0.05), ("0.100", 0.1), ("0.300", 0.3)]
d, r1, r2, ylo, L = 0.01, 1.5, 0.36, 1.5, 5

def load(xi, n):
    return np.loadtxt(f"{BASE}/SimulationOutputs_ξ_{xi}/{n}.txt")
def dint(t, f):
    w = np.exp(-d * t); n = min(len(t), len(f)); m = t[:n] <= 60
    return np.trapz(w[:n][m] * f[:n][m], t[:n][m])

CH = ["capital", "knowledge", "climate", "damage jump", "tech jump"]
COL = ["#999999", "#CC79A7", "#56B4E9", "#D55E00", "#0072B2"]
vals = {c: [] for c in CH}
for xi, _ in XIS:
    t = load(xi, "t"); Y = load(xi, "Y")
    hd, hg, hr, hy = load(xi, "h_d"), load(xi, "h_g"), load(xi, "h_r"), load(xi, "h_y")
    gpost = load(xi, "g_post"); wdist = load(xi, "lambda3_weights_distorted")
    dmgI, techI = load(xi, "dmg_jump_intensity"), load(xi, "tech_jump_intensity")
    vals["capital"].append(dint(t, 0.5 * (hd**2 + hg**2)))
    vals["knowledge"].append(dint(t, 0.5 * hr**2))
    vals["climate"].append(dint(t, 0.5 * hy**2))
    Jn = r1 * (np.exp(r2 / 2 * np.maximum(Y - ylo, 0)**2) - 1) * (Y >= ylo)
    gbar = np.where(Jn > 1e-12, dmgI / np.maximum(Jn, 1e-12), 1.0)
    n = min(len(t), len(Y)); ent = np.zeros(n)
    for l in range(5):
        gl = L * gbar[:n] * wdist[l]
        ent += (1.0 / L) * (gl * np.log(np.maximum(gl, 1e-12)) - gl + 1.0)
    vals["damage jump"].append(dint(t[:n], Jn[:n] * ent))
    n2 = min(len(t), len(gpost)); Jg = np.where(gpost[:n2] > 1e-9, techI[:n2] / gpost[:n2], 0.0)
    vals["tech jump"].append(dint(t[:n2], Jg * (gpost[:n2] * np.log(np.maximum(gpost[:n2], 1e-12)) - gpost[:n2] + 1.0)))

plt.rcParams.update({"font.size": 14})
fig, ax = plt.subplots(figsize=(8.4, 4.6))
x = np.arange(len(XIS)); bottom = np.zeros(len(XIS))
for c, col in zip(CH, COL):
    ax.bar(x, vals[c], 0.55, bottom=bottom, color=col, label=c)
    bottom += np.array(vals[c])
ax.set_xticks(x); ax.set_xticklabels([f"ξ = {l}" for _, l in XIS])
ax.set_ylabel("welfare cost of robustness  (discounted relative entropy)")
ax.legend(frameon=False, ncol=1, loc="upper right")
ax.grid(axis="y", alpha=.25)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "figures", "welfare_decomp.png"),
            dpi=160, bbox_inches="tight")
tot = sum(np.array(vals[c]) for c in CH)
print("shares @xi=0.05:", {c: round(vals[c][0] / tot[0] * 100, 1) for c in CH})
print("total by xi:", np.round(tot, 3))
