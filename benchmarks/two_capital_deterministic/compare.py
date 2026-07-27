"""
Solve the deterministic two-capital model with (1) an accurate finite-difference
reference (central-difference Newton) and (2) the project's DGM-PIA neural net
(value + i_d + i_g networks, 3 losses), then plot and compare.

Plots i^d vs Z and i^g vs Z (FD vs NN), plus aggregate productivity Abar(Z) and
the consumption/output ratio C/Y, and the value slope v'(Z).

Usage:
    python compare.py [--a-g A_g_prime_prime] [--nn-iters 200000] [--no-nn]
"""

import argparse
import os

import numpy as np

import two_capital_model as M
from reference_solver import solve_newton, true_residual


def _augment(out, P):
    """Add aggregate productivity Abar(Z) and consumption/output C/Y."""
    Z = out["Z"]
    out["Abar"] = (1.0 - Z) * P["A_d"] + Z * P["A_g"]        # output/K = Abar(Z)
    out["C_over_Y"] = out["c"] / out["Abar"]                  # C/Y = (C/K)/(Y/K)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-g", default="A_g_prime_prime",
                    choices=["A_g", "A_g_prime", "A_g_prime_prime"])
    ap.add_argument("--nn-iters", type=int, default=200000)
    ap.add_argument("--fd-n", type=int, default=600)
    ap.add_argument("--no-nn", action="store_true")
    ap.add_argument("--reuse-nn", action="store_true",
                    help="load the saved NN eval npz instead of retraining (no TF needed)")
    ap.add_argument("--arch", default="forwardnet", choices=["forwardnet", "dgm"],
                    help="NN architecture: forwardnet (models/) or dgm (models_dgm/ gated)")
    ap.add_argument("--fd-supervise", action="store_true",
                    help="warm-start the NN on the FD reference before residual training")
    ap.add_argument("--precond", action="store_true",
                    help="precondition the HJB residual by 1/(|mu|+eps) to fix v' conditioning")
    ap.add_argument("--precond-eps", type=float, default=1e-3)
    ap.add_argument("--lbfgs-iters", type=int, default=5000)
    ap.add_argument("--lr-v", type=float, default=1e-4)
    ap.add_argument("--lr-c", type=float, default=4e-3)
    ap.add_argument("--label", default="", help="extra suffix to disambiguate output files")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    P = M.load_calibration(args.a_g)
    print("PARAM SOURCE = parent models/params.py")
    print(f"  A_d={P['A_d']}  A_g={P['A_g']} ({args.a_g})  delta={P['delta']}  "
          f"Gamma={P['Gamma_d']}  theta={P['theta_d']}  alpha={P['alpha_d']}", flush=True)

    # --- Finite difference (accurate central-difference Newton) ---
    Zf, vf, pf, sol = solve_newton(P, n=args.fd_n)
    rf = true_residual(Zf, vf, P)              # HJB loss on the RAW solved residual
    fd_loss_l2 = float(np.sqrt(np.mean(rf[2:-2] ** 2)))
    fd_loss_max = float(np.max(np.abs(rf[2:-2])))
    # The central-difference slope carries grid-scale odd-even oscillation; denoise
    # it (Savitzky-Golay) for the displayed slope/controls (trend is preserved).
    try:
        from scipy.signal import savgol_filter
        win = min(31, (len(pf) // 2) * 2 - 1)
        ps = savgol_filter(pf, win, 3)
    except Exception:
        ps = pf
    i_d, i_g, c = M.controls(Zf, ps, P)
    fd = {"method": "FD", "Z": Zf, "v": vf, "slope": ps, "i_d": i_d, "i_g": i_g,
          "c": c, "ratio": i_g / i_d, "residual": rf}
    _augment(fd, P)
    print(f"[FD] central-Newton converged={sol.success}  "
          f"HJB-loss L2={fd_loss_l2:.2e}  max={fd_loss_max:.2e}", flush=True)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    sup = "_fdsup" if args.fd_supervise else ""
    pc = "_pc" if args.precond else ""
    lrtag = "" if abs(args.lr_v - 1e-4) < 1e-12 else f"_lrv{args.lr_v:g}"
    lbl = f"_{args.label}" if args.label else ""
    tag = f"{args.a_g}_{args.arch}{sup}{pc}{lrtag}{lbl}"
    nn_npz = os.path.join(out_dir, f"nn_eval_{tag}.npz")

    # --- Neural network (project DGM-PIA, 3 losses) ---
    nn = None
    if not args.no_nn:
        if args.reuse_nn and os.path.exists(nn_npz):
            z = np.load(nn_npz)
            nn = {k: z[k] for k in z.files}
            nn["arch"] = str(nn.get("arch", args.arch))
            print(f"[NN] loaded saved eval from {nn_npz}", flush=True)
        else:
            from nn_dgm_solver import solve_nn_dgm
            nn = solve_nn_dgm(P, iters=args.nn_iters, seed=args.seed, verbose=True,
                              arch=args.arch, fd_ref=(fd if args.fd_supervise else None),
                              precond=args.precond, precond_eps=args.precond_eps,
                              lbfgs_iters=args.lbfgs_iters, lr_v=args.lr_v, lr_c=args.lr_c)
            np.savez(nn_npz, Z=nn["Z"], v=nn["v"], slope=nn["slope"], i_d=nn["i_d"],
                     i_g=nn["i_g"], c=nn["c"], ratio=nn["ratio"], residual=nn["residual"],
                     final_losses=np.array(nn["final_losses"]), arch=args.arch)
            print(f"[NN] saved eval to {nn_npz}", flush=True)
        _augment(nn, P)
        if "final_losses" in nn:
            hjb, fd_l, fg_l = [float(x) for x in np.array(nn["final_losses"]).ravel()]
            print(f"[NN/{args.arch}{sup}] final losses  HJB={hjb:.2e}  FOC_d={fd_l:.2e}  FOC_g={fg_l:.2e}", flush=True)
    _plot(P, fd, nn, tag, out_dir)
    _dump_csv(fd, nn, tag, out_dir)

    if nn is not None:
        Zc = np.linspace(0.05, 0.95, 19)
        for key in ("i_d", "i_g"):
            a = np.interp(Zc, fd["Z"], fd[key]); b = np.interp(Zc, nn["Z"], nn[key])
            print(f"max |{key}_FD - {key}_NN| on [0.05,0.95] = {np.max(np.abs(a-b)):.3e}")


def _plot(P, fd, nn, tag, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    # Restrict the control/value panels to the reliable interior; the two ends are
    # degenerate (mu_Z -> 0) one-capital limits where the central-difference slope
    # is numerically singular. Aggregate productivity Abar(Z) is exact, plotted full.
    plo, phi = 0.10, 0.90

    def _pts(o, key, npts=33):
        m = (o["Z"] >= plo) & (o["Z"] <= phi)
        Z, y = o["Z"][m], o[key][m]
        step = max(1, len(Z) // npts)
        return Z[::step], y[::step]   # sparse sample, connected by straight lines

    nn_label = "NN (DGM-PIA)"
    if nn is not None:
        nn_label = f"NN ({str(nn.get('arch', 'forwardnet'))})"

    def cmp(a, key, title, ylab):
        zf, yf = _pts(fd, key)
        a.plot(zf, yf, "b-o", ms=4, lw=1.5, label="FD (central Newton)")
        if nn is not None:
            zn, yn = _pts(nn, key)
            a.plot(zn, yn, "r--s", ms=4, lw=1.5, label=nn_label)
        a.set_xlabel("Z (green capital share)"); a.set_ylabel(ylab)
        a.set_title(title); a.legend(); a.grid(alpha=0.3)

    cmp(ax[0, 0], "i_d", r"Dirty investment rate $i^d$ vs $Z$", r"$i^d$")
    cmp(ax[0, 1], "i_g", r"Green investment rate $i^g$ vs $Z$", r"$i^g$")
    cmp(ax[0, 2], "slope", r"Value slope $v'(Z)$", r"$v'(Z)$")
    # Abar(Z): same for both methods (pure productivity)
    ax[1, 0].plot(fd["Z"], fd["Abar"], "k-", lw=2)
    ax[1, 0].set_xlabel("Z"); ax[1, 0].set_ylabel(r"$\bar A(Z)$")
    ax[1, 0].set_title(r"Aggregate (average) productivity $\bar A(Z)=(1-Z)A_d+Z A_g$")
    ax[1, 0].grid(alpha=0.3)
    cmp(ax[1, 1], "C_over_Y", "Consumption / output  C/Y", "C/Y")
    ax[1, 2].axis("off")   # 6th panel (i^g/i^d ratio) removed

    fig.suptitle(f"Deterministic two-capital: FD vs NN (DGM-PIA)   "
                 f"A_d={P['A_d']}, A_g={P['A_g']} [{tag}]", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, f"two_capital_FDvsNN_{tag}.png")
    fig.savefig(path, dpi=150)
    print(f"saved figure: {path}", flush=True)


def _dump_csv(fd, nn, tag, out_dir):
    Z = fd["Z"]
    cols = {"Z": Z, "vprime_FD": fd["slope"], "i_d_FD": fd["i_d"], "i_g_FD": fd["i_g"],
            "c_FD": fd["c"], "Abar": fd["Abar"], "C_over_Y_FD": fd["C_over_Y"]}
    if nn is not None:
        for k_out, k_in in [("vprime_NN", "slope"), ("i_d_NN", "i_d"), ("i_g_NN", "i_g"),
                            ("c_NN", "c"), ("C_over_Y_NN", "C_over_Y")]:
            cols[k_out] = np.interp(Z, nn["Z"], nn[k_in])
    keys = list(cols.keys())
    arr = np.column_stack([cols[k] for k in keys])
    path = os.path.join(out_dir, f"two_capital_FDvsNN_{tag}.csv")
    np.savetxt(path, arr, delimiter=",", header=",".join(keys), comments="")
    print(f"saved csv: {path}", flush=True)


if __name__ == "__main__":
    main()
