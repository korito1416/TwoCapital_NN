"""
FINAL figure for the corrected FD (fd_pdpt_v5 = PIBYS, grid-converged, de-investment) vs the validated NN.
Tells the whole story in one 2x3:
  Row 1 (climate response, vs temperature Y at Z=0.7): i_d, i_g, c -- the FD de-invests as Y rises; NN is flat.
  Row 2: i_d vs Z and v_logK vs Z at Y=3 (FD vs NN), and the GRID-CONVERGENCE panel proving the FD is correct
         (PIBYS v_logK is flat under Z-refinement; the old upwind v3 / ADI v2 drift and never converge).
Loads outputs/fd_pdpt_v5_lam3_0167_xi148.npz (no recompute); NN via plot_pretrained_climate (TF).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_pretrained_climate as C

OD = C.OD
LOGK0 = C.logK_fix

# ---- grid-convergence data (v_logK at the reference under refinement of the advection-dominated Z) ----
# PIBYS v5 (job 51119266): flat. Upwind v3 / ADI v2: drift away, never converge.
CONV = {
    "FD PIBYS (v5, corrected)": dict(nZ=[21, 41, 81, 121], vlK=[0.4074, 0.4074, 0.4073, 0.4074], c="b", m="o"),
    "FD upwind (v3)":           dict(nZ=[21, 31, 41],       vlK=[0.296, 0.329, 0.342],            c="orange", m="s"),
    "FD ADI (v2)":              dict(nZ=[25, 51, 80],       vlK=[0.563, 0.685, 0.793],            c="r", m="^"),
}
RICHARDSON_VLK = None   # filled from fd_v5_richardson once available; drawn as the converged limit


def fd_at_logK(out, key, logk):
    arr = out[key]; lk = out["logK"]
    i = np.searchsorted(lk, logk); i = min(max(i, 1), len(lk) - 1)
    w = (logk - lk[i - 1]) / (lk[i] - lk[i - 1])
    return (1 - w) * arr[i - 1] + w * arr[i]


def main():
    d = np.load(os.path.join(OD, "fd_pdpt_v5_lam3_0167_xi148.npz"))
    out = {k: d[k] for k in d.files}
    Zf, Yf = out["Z"], out["Y"]
    fd = {k: fd_at_logK(out, k, LOGK0) for k in ("i_d", "i_g", "vlK", "c", "vY")}
    lNy = 0.00017675 + 2 * 0.0022 * Yf + (1 / 6.0) * (Yf - 2.5)
    fd_VY = fd["vY"] - lNy[None, :]

    net = C.build_and_load()
    def nn_eval(Zv, Yv):
        return C.evaluate(*net, np.asarray(Zv), np.asarray(Yv), logK=LOGK0, lam3=1 / 6.0, logxi=5.0)

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.2))
    Zg = np.linspace(0.1, 0.9, 41); Yg = np.linspace(0.0, 4.0, 41)
    iZ07 = np.argmin(np.abs(Zf - 0.7)); iY3 = np.argmin(np.abs(Yf - 3.0))

    # Row 1: climate response vs Y at Z=0.7
    nnY = nn_eval(np.full_like(Yg, 0.7), Yg)
    for a, (k, nk, ylab) in zip(ax[0], [("i_d", "i_d", r"$i^d$ (dirty investment)"),
                                         ("i_g", "i_g", r"$i^g$ (green investment)"), ("c", "c", r"$c$ (consumption)")]):
        fdv = np.array([np.interp(y, Yf, fd[k][iZ07, :]) for y in Yg])
        a.plot(Yg, fdv, "b-o", ms=3, lw=1.6, label="FD (PIBYS, corrected)")
        a.plot(Yg, nnY[nk], "r--s", ms=3, lw=1.6, label="NN")
        a.axhline(0, color="k", lw=0.7, alpha=0.5)
        a.set_title(fr"{ylab} vs $Y$   ($Z=0.7$)"); a.set_xlabel("temperature $Y$"); a.legend(); a.grid(alpha=0.3)
    ax[0, 0].annotate("FD de-invests\nas $Y$ rises", xy=(3.2, -0.03), xytext=(2.0, -0.045),
                      fontsize=9, color="b", arrowprops=dict(arrowstyle="->", color="b"))

    # Row 2a,b: vs Z at Y=3
    nnZ = nn_eval(Zg, np.full_like(Zg, 3.0))
    for a, (k, nk, ylab) in zip(ax[1, :2], [("i_d", "i_d", r"$i^d$"), ("vlK", "v_logK", r"$v_{\log K}$")]):
        fdv = np.array([np.interp(z, Zf, fd[k][:, iY3]) for z in Zg])
        a.plot(Zg, fdv, "b-o", ms=3, lw=1.6, label="FD (PIBYS)")
        a.plot(Zg, nnZ[nk], "r--s", ms=3, lw=1.6, label="NN")
        a.axhline(0, color="k", lw=0.7, alpha=0.5)
        a.set_title(fr"{ylab} vs $Z$   ($Y=3$)"); a.set_xlabel("green share $Z$"); a.legend(); a.grid(alpha=0.3)

    # Row 2c: grid convergence
    a = ax[1, 2]
    for lab, s in CONV.items():
        a.plot(s["nZ"], s["vlK"], marker=s["m"], color=s["c"], lw=1.6, ms=6, label=lab)
    if RICHARDSON_VLK is not None:
        a.axhline(RICHARDSON_VLK, color="b", ls=":", lw=1.4, alpha=0.8,
                  label=f"Richardson limit {RICHARDSON_VLK:.3f}")
    a.set_xscale("log"); a.set_xlabel(r"$n_Z$ (Z-grid points, advection-dominated dir.)")
    a.set_title(r"grid convergence of $v_{\log K}$ at ref."); a.set_ylabel(r"$v_{\log K}$")
    a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
    a.text(0.5, 0.06, "PIBYS: FLAT (converged)\nupwind/ADI: drift, never converge",
           transform=a.transAxes, fontsize=8.5, ha="center", color="dimgray")

    fig.suptitle(f"Post-damage post-tech: corrected FD (PIBYS, grid-converged, de-investment) vs validated NN   "
                 f"[logK={LOGK0:.2f}, $\\lambda_3$=1/6, $\\xi$=148.4]", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OD, "fd_final.png"); fig.savefig(p, dpi=150); print("saved", p, flush=True)


if __name__ == "__main__":
    main()
