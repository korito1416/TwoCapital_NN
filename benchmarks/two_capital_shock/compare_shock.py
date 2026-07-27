"""
Shock-model driver: FD (ground truth) vs NN (DGM-PIA with v''), at a chosen sigma,
with/without FD supervision. Plots i^d, i^g, marginal values q_d/q_g, v', C/Y.

Usage:
  python compare_shock.py --sigma 0.2 [--fd-supervise] [--precond] [--no-nn]
                          [--reuse-nn] [--arch forwardnet|dgm] [--label TAG]
"""
import argparse
import os
import numpy as np

import two_capital_shock_model as M
from fd_shock import solve_fd_shock


def augment(d, P):
    Z = d["Z"]
    d["q_d"] = 1.0 - Z * d["slope"]
    d["q_g"] = 1.0 + (1.0 - Z) * d["slope"]
    d["Abar"] = (1.0 - Z) * P["A_d"] + Z * P["A_g"]
    d["C_over_Y"] = d["c"] / d["Abar"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--a-g", default="A_g_prime_prime")
    ap.add_argument("--arch", default="forwardnet", choices=["forwardnet", "dgm"])
    ap.add_argument("--fd-supervise", action="store_true")
    ap.add_argument("--precond", action="store_true")
    ap.add_argument("--nn-iters", type=int, default=120000)
    ap.add_argument("--lbfgs-iters", type=int, default=0)
    ap.add_argument("--fd-n", type=int, default=1500)
    ap.add_argument("--no-nn", action="store_true")
    ap.add_argument("--reuse-nn", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    P = M.load_calibration(args.a_g)
    P["sigma_d"] = args.sigma; P["sigma_g"] = args.sigma
    print(f"sigma={args.sigma}, arch={args.arch}, fd_supervise={args.fd_supervise}, "
          f"precond={args.precond}, lbfgs={args.lbfgs_iters}", flush=True)

    fd = augment(solve_fd_shock(P, n=args.fd_n), P)
    print(f"[FD] iters={fd['iters']} max|resid|={fd['max_abs_residual']:.2e}", flush=True)

    OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(OD, exist_ok=True)
    sup = "_fdsup" if args.fd_supervise else "_nosup"
    pc = "_pc" if args.precond else ""
    lbl = f"_{args.label}" if args.label else ""
    tag = f"sigma{args.sigma:g}_{args.arch}{sup}{pc}{lbl}"
    nn_npz = os.path.join(OD, f"nn_eval_shock_{tag}.npz")

    nn = None
    if not args.no_nn:
        if args.reuse_nn and os.path.exists(nn_npz):
            z = np.load(nn_npz); nn = augment({k: z[k] for k in z.files}, P)
            print(f"[NN] loaded {nn_npz}", flush=True)
        else:
            from nn_dgm_shock import solve_nn_shock
            nn = solve_nn_shock(P, iters=args.nn_iters, seed=args.seed, verbose=True,
                                arch=args.arch, fd_ref=(fd if args.fd_supervise else None),
                                precond=args.precond, lbfgs_iters=args.lbfgs_iters)
            np.savez(nn_npz, Z=nn["Z"], v=nn["v"], slope=nn["slope"], i_d=nn["i_d"],
                     i_g=nn["i_g"], c=nn["c"], final_losses=np.array(nn["final_losses"]), arch=args.arch)
            nn = augment(nn, P)
            print(f"[NN] saved {nn_npz}", flush=True)
        hjb, fdl, fgl = [float(x) for x in np.array(nn["final_losses"]).ravel()]
        print(f"[NN/{tag}] final HJB={hjb:.2e} FOC_d={fdl:.2e} FOC_g={fgl:.2e}", flush=True)
        for key in ("i_d", "i_g", "q_g", "slope"):
            zc = np.linspace(0.1, 0.9, 17)
            a = np.interp(zc, fd["Z"], fd[key]); b = np.interp(zc, nn["Z"], nn[key])
            print(f"  max|{key}_FD - {key}_NN| on [0.1,0.9] = {np.max(np.abs(a-b)):.3e}", flush=True)

    _plot(P, fd, nn, tag, OD, args.sigma)


def _plot(P, fd, nn, tag, OD, sigma):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plo, phi = 0.10, 0.90

    def pts(o, key, n=33):
        m = (o["Z"] >= plo) & (o["Z"] <= phi)
        Z, y = o["Z"][m], o[key][m]
        s = max(1, len(Z) // n)
        return Z[::s], y[::s]

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    panels = [("i_d", r"Dirty investment $i^d$", r"$i^d$"),
              ("i_g", r"Green investment $i^g$", r"$i^g$"),
              ("q_d", r"Marginal value dirty $q_d$", r"$q_d$"),
              ("q_g", r"Marginal value green $q_g$", r"$q_g$"),
              ("slope", r"Value slope $v'(Z)$", r"$v'$"),
              ("C_over_Y", r"Consumption/output $C/Y$", "C/Y")]
    for a, (key, title, ylab) in zip(ax.ravel(), panels):
        zf, yf = pts(fd, key); a.plot(zf, yf, "b-o", ms=4, lw=1.5, label="FD")
        if nn is not None:
            zn, yn = pts(nn, key); a.plot(zn, yn, "r--s", ms=4, lw=1.5, label="NN")
        a.set_xlabel("Z"); a.set_ylabel(ylab); a.set_title(title); a.legend(); a.grid(alpha=0.3)
    fig.suptitle(f"Two-capital WITH shocks  [{tag}]  sigma={sigma}", fontsize=13)
    fig.tight_layout()
    path = os.path.join(OD, f"shock_FDvsNN_{tag}.png")
    fig.savefig(path, dpi=150)
    print("saved figure:", path, flush=True)


if __name__ == "__main__":
    main()
