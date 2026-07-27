"""BRICK 0 -- the level-under-identification WITNESS (no new PDE solve).

Take the verified FD solution of the terminal Post-Damage Post-Tech regime (logK,Z,Y),
add a uniform welfare-level shift Delta while HOLDING THE POLICY FIXED, and recompute the
true HJB residual (fd_pdpt_v5._residual) on the economic interior box. Because a constant
level shift leaves every derivative untouched and only moves the -delta*V source, the
residual RMS traces an EXACT hyperbola with asymptote slope delta = 0.01: a welfare-level
error of order (NN residual floor)/delta ~ 0.1 units is INVISIBLE to the equation error.
This panel is grid- and FD-accuracy-independent; it depends only on delta.

Message for Mike & Lars: the HJB equation error cannot rank solutions that differ in
welfare level; the FD solution supplies the unique verified level; distance-to-FD is the
missing selection criterion.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FDDIR = os.path.join(HERE, "..", "post_damage_post_tech")
sys.path.insert(0, FDDIR)
import fd_pdpt_v5 as FD

NN_FLOOR = 1e-3          # paper-grade NN loss_v (root-mean-square HJB residual)
LAM3 = 1.0 / 6.0
d = np.load(os.path.join(FDDIR, "outputs", "fd_pdpt_v5_stable_lam3_0167_xi148.npz"))
logK, Z, Y = d["logK"], d["Z"], d["Y"]
v, i_d, i_g, c = d["v"], d["i_d"], d["i_g"], d["c"]
p = FD.P
dK, dZ, dY = logK[1] - logK[0], Z[1] - Z[0], Y[1] - Y[0]
LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
E = p["eta"] * p["A_d"] * (1 - ZZ) * np.exp(LK)
lNy = p["l1"] + p["l2"] * YY + LAM3 * (YY - p["y_up"])
lNyy = p["l2"] + LAM3

# _residual returns R on the trimmed interior grid logK[1:-1], Z[1:-1], Y[1:-2].
# Build the economic-interior box mask on THOSE trimmed coordinates.
logK_r, Z_r, Y_r = logK[1:-1], Z[1:-1], Y[1:-2]
lki = (logK_r >= 4.3) & (logK_r <= 6.7); zi = (Z_r >= 0.30) & (Z_r <= 0.95); yi = (Y_r >= 0.5) & (Y_r <= 4.0)
box = np.ix_(lki, zi, yi)

def resid_rms(delta):
    R = FD._residual(v + delta, i_d, i_g, c, ZZ, E, lNy, lNyy, LAM3, LK, dK, dZ, dY, p)
    return float(np.sqrt(np.mean(R[box] ** 2)))

deltas = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0])
deltas = np.unique(np.concatenate([-deltas, deltas]))
rms = np.array([resid_rms(dl) for dl in deltas])
base = resid_rms(0.0)
# half-width where the shifted-FD residual first exceeds the NN floor (un-rankable band)
pos = deltas > 0
hw = np.interp(NN_FLOOR, rms[pos], deltas[pos]) if rms[pos].max() > NN_FLOOR else np.nan

print(f"baseline residual RMS on box = {base:.3e}   (NN floor = {NN_FLOOR:.0e})")
print(f"delta -> residual RMS:")
for dl, r in zip(deltas, rms):
    print(f"  Delta={dl:+7.2f}  RMS={r:.3e}   (exact delta*|Delta|={p['delta']*abs(dl):.3e})")
print(f"un-ranked level half-width |Delta*| where RMS crosses NN floor = {hw:.3f} welfare units")

plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(8.4, 5.6))
dfine = np.linspace(deltas.min(), deltas.max(), 400)
ax.plot(dfine, np.sqrt(base ** 2 + (p["delta"] * dfine) ** 2), "-", color="#0072B2", lw=2.4,
        label=r"$\sqrt{R_0^2 + (\delta\,\Delta)^2}$  (exact, slope $\delta=0.01$)")
ax.plot(deltas, rms, "o", color="0.1", ms=7, label="FD residual RMS (recomputed)")
ax.axhspan(0, NN_FLOOR, color="#D55E00", alpha=0.14, lw=0)
ax.axhline(NN_FLOOR, color="#D55E00", lw=1.3, ls="--", label=f"NN residual floor ({NN_FLOOR:.0e})")
if np.isfinite(hw):
    ax.axvspan(-hw, hw, color="#009E73", alpha=0.12, lw=0)
    ax.annotate(f"level error $|\\Delta|\\lesssim{hw:.2f}$ welfare units\ninvisible to the equation error",
                xy=(0, NN_FLOOR), xytext=(0.15, NN_FLOOR * 3.2), ha="left", fontsize=11.5,
                arrowprops=dict(arrowstyle="->", color="0.3"))
ax.set_yscale("log")
ax.set_xlabel(r"uniform welfare-level shift $\Delta$ (utils)")
ax.set_ylabel("HJB residual RMS on the economic interior box")
ax.set_title("A welfare-level error is nearly invisible to the HJB equation error")
ax.grid(alpha=.25, which="both", lw=.5); ax.legend(fontsize=11, loc="upper center")
fig.tight_layout()
out = os.path.join(HERE, "figures", "level_witness.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)
