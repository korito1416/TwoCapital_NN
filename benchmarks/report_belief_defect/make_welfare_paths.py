"""Path (time-series) view of the welfare cost of robustness. Instead of one summary
bar, show how the discounted per-channel relative entropy ACCUMULATES over the 60-year
path (left, stacked area -> total) and WHEN it is incurred (right, the flow/integrand).
Same channels and numbers as make_welfare.py; ξ = 0.05."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = os.path.join(ROOT, "output_xiprofile_ref_20260714", "SimulationDeterministic")
d, r1, r2, ylo, L = 0.01, 1.5, 0.36, 1.5, 5
XI = "0.050"

def load(n):
    return np.loadtxt(f"{BASE}/SimulationOutputs_ξ_{XI}/{n}.txt")

t = load("t"); Y = load("Y")
hd, hg, hr, hy = load("h_d"), load("h_g"), load("h_r"), load("h_y")
gpost = load("g_post"); wdist = load("lambda3_weights_distorted")
dmgI, techI = load("dmg_jump_intensity"), load("tech_jump_intensity")
n = min(len(t), len(Y)); t = t[:n]; m = t <= 60
w = np.exp(-d * t)

# per-channel discounted entropy FLOW (integrand)
flow = {}
flow["capital"] = w * 0.5 * (hd[:n]**2 + hg[:n]**2)
flow["knowledge"] = w * 0.5 * hr[:n]**2
flow["climate"] = w * 0.5 * hy[:n]**2
Jn = r1 * (np.exp(r2 / 2 * np.maximum(Y[:n] - ylo, 0)**2) - 1) * (Y[:n] >= ylo)
gbar = np.where(Jn > 1e-12, dmgI[:n] / np.maximum(Jn, 1e-12), 1.0)
entd = np.zeros(n)
for l in range(5):
    gl = L * gbar * wdist[l]
    entd += (1.0 / L) * (gl * np.log(np.maximum(gl, 1e-12)) - gl + 1.0)
flow["damage jump"] = w * Jn * entd
Jg = np.where(gpost[:n] > 1e-9, techI[:n] / gpost[:n], 0.0)
flow["tech jump"] = w * Jg * (gpost[:n] * np.log(np.maximum(gpost[:n], 1e-12)) - gpost[:n] + 1.0)

CH = ["capital", "knowledge", "climate", "damage jump", "tech jump"]
COL = ["#999999", "#CC79A7", "#56B4E9", "#D55E00", "#0072B2"]

def cum(f):  # cumulative discounted integral along the path
    c = np.zeros(n)
    c[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))
    return c

plt.rcParams.update({"font.size": 16})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.9))

# LEFT: cumulative, stacked area -> total welfare cost of robustness
cums = {c: cum(flow[c]) for c in CH}
base = np.zeros(n)
for c, col in zip(CH, COL):
    ax[0].fill_between(t[m], base[m], (base + cums[c])[m], color=col, label=c, alpha=.9)
    base = base + cums[c]
ax[0].set_title("cumulative welfare cost of robustness")
ax[0].set_xlabel("year"); ax[0].set_ylabel("discounted relative entropy, accrued to $t$")
ax[0].set_xlim(0, 60); ax[0].grid(alpha=.25)
ax[0].legend(frameon=False, loc="upper left", fontsize=11.5)
ax[0].annotate(f"total = {base[m][-1]:.2f}", (60, base[m][-1]), xytext=(-4, 4),
               textcoords="offset points", ha="right", fontsize=11)

# RIGHT: the flow (when the cost is incurred) -- jumps dominate, diffusion hugs zero
for c, col in zip(CH, COL):
    ax[1].plot(t[m], flow[c][m], color=col, lw=2.2, label=c)
ax[1].set_title("flow: where along the path the cost is incurred")
ax[1].set_xlabel("year"); ax[1].set_ylabel("discounted entropy rate")
ax[1].set_xlim(0, 60); ax[1].grid(alpha=.25)

fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "figures", "welfare_paths.png"),
            dpi=160, bbox_inches="tight")
print("welfare_paths.png written; totals by channel (xi=0.05):",
      {c: round(cums[c][m][-1], 4) for c in CH})
print("grand total:", round(base[m][-1], 3))
