"""
Plot the CORRECTED FD (fd_pdpt_v5 = Policy Iteration By Simulation: exact-characteristic policy
evaluation, zero artificial diffusion, grid-converged in the advection-dominated Z) against the
validated DGM NN. lambda3=1/6, xi=148.4. The FD gives DE-INVESTMENT (i_d<0) because it integrates
the high-Y climate cost; the NN (confined to Y<=4) gives invest -- the figure shows the gap and why.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_pretrained_climate as C   # NN (TF): build_and_load, evaluate, logK_fix

OD = C.OD
LOGK0 = C.logK_fix      # log(880) ~ 6.78, the NN's fixed slice


def fd_at_logK(out, key, logk):
    """Linear-interpolate an FD field (nK,nZ,nY) to a single logK plane -> (nZ,nY)."""
    arr = out[key]; lk = out["logK"]
    i = np.searchsorted(lk, logk); i = min(max(i, 1), len(lk) - 1)
    w = (logk - lk[i - 1]) / (lk[i] - lk[i - 1])
    return (1 - w) * arr[i - 1] + w * arr[i]


def main():
    print("=== load corrected FD (PIBYS v5, grid-converged 31x61x31) ===", flush=True)
    d = np.load(os.path.join(OD, "fd_pdpt_v5_lam3_0167_xi148.npz"))
    out = {k: d[k] for k in d.files}
    out["max_abs_residual"] = 1.4e-3      # from the converged 31x61x31 run

    # FD planes at logK = LOGK0
    Zf, Yf = out["Z"], out["Y"]
    fd = {k: fd_at_logK(out, k, LOGK0) for k in ("i_d", "i_g", "vlK", "c", "vY")}
    lNy = 0.00017675 + 2 * 0.0022 * Yf + (1/6.0) * (Yf - 2.5)            # (logN)_Y on the Y grid
    fd_VY = fd["vY"] - lNy[None, :]

    # NN at the same plane
    net = C.build_and_load()

    def nn_eval(Zv, Yv):
        return C.evaluate(*net, np.asarray(Zv), np.asarray(Yv), logK=LOGK0, lam3=1/6.0, logxi=5.0)

    # reference-point comparison
    o0 = nn_eval([0.7], [3.0])
    iZ = np.argmin(np.abs(Zf - 0.7)); iY = np.argmin(np.abs(Yf - 3.0))
    print(f"[ref logK={LOGK0:.2f},Z=0.7,Y=3.0]  FD: i_d={fd['i_d'][iZ,iY]:+.4f} i_g={fd['i_g'][iZ,iY]:+.4f} "
          f"vlK={fd['vlK'][iZ,iY]:.3f} c={fd['c'][iZ,iY]:.4f} V_Y={fd_VY[iZ,iY]:+.4f}", flush=True)
    print(f"                              NN: i_d={o0['i_d'][0]:+.4f} i_g={o0['i_g'][0]:+.4f} "
          f"vlK={o0['v_logK'][0]:.3f} c={o0['c'][0]:.4f} V_Y={o0['V_Y'][0]:+.4f}", flush=True)

    # interior max-abs differences over Z in [0.2,0.8], Y in [0.5,3.5]
    zc = np.linspace(0.2, 0.8, 13)
    for Yv in (1.0, 2.0, 3.0):
        for k, nk in [("i_d", "i_d"), ("i_g", "i_g"), ("vlK", "v_logK")]:
            a = np.array([np.interp(z, Zf, fd[k][:, np.argmin(np.abs(Yf - Yv))]) for z in zc])
            b = nn_eval(zc, np.full_like(zc, Yv))[nk]
            print(f"  Y={Yv}: max|{k}_FD-NN| on Z[0.2,0.8] = {np.max(np.abs(a-b)):.3e}", flush=True)

    # ---- plot FD vs NN ----
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    Zg = np.linspace(0.1, 0.9, 41); Yg = np.linspace(0.0, 4.0, 41)
    # top: vs Z at Y=3
    nnZ = nn_eval(Zg, np.full_like(Zg, 3.0))
    for a, (k, nk, ylab) in zip(ax[0], [("i_d", "i_d", r"$i^d$"), ("i_g", "i_g", r"$i^g$"), ("vlK", "v_logK", r"$v_{\log K}$")]):
        fdv = np.array([np.interp(z, Zf, fd[k][:, np.argmin(np.abs(Yf - 3.0))]) for z in Zg])
        a.plot(Zg, fdv, "b-o", ms=3, lw=1.4, label="FD (PIBYS)"); a.plot(Zg, nnZ[nk], "r--s", ms=3, lw=1.4, label="NN")
        a.axhline(0, color="k", lw=0.6, alpha=0.4)
        a.set_title(fr"{ylab} vs Z  ($Y=3$)"); a.set_xlabel("Z"); a.legend(); a.grid(alpha=0.3)
    # bottom: vs Y at Z=0.7
    nnY = nn_eval(np.full_like(Yg, 0.7), Yg)
    for a, (k, nk, ylab) in zip(ax[1], [("i_d", "i_d", r"$i^d$"), ("i_g", "i_g", r"$i^g$"), ("c", "c", r"$c$")]):
        fdv = np.array([np.interp(y, Yf, fd[k][np.argmin(np.abs(Zf - 0.7)), :]) for y in Yg])
        a.plot(Yg, fdv, "b-o", ms=3, lw=1.4, label="FD (PIBYS)"); a.plot(Yg, nnY[nk], "r--s", ms=3, lw=1.4, label="NN")
        a.axhline(0, color="k", lw=0.6, alpha=0.4)
        a.set_title(fr"{ylab} vs Y  ($Z=0.7$)"); a.set_xlabel("Y"); a.legend(); a.grid(alpha=0.3)
    fig.suptitle(f"Post-damage post-tech: corrected FD (PIBYS, grid-converged, de-invest) vs validated NN  "
                 f"[logK={LOGK0:.2f}, $\\lambda_3$=1/6, $\\xi$=148.4]  FD max|resid|={out['max_abs_residual']:.1e}",
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(OD, "fd_vs_nn.png"); fig.savefig(p, dpi=150); print("saved", p, flush=True)


if __name__ == "__main__":
    main()
