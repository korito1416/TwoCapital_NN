"""
Driver: robust two-capital DGM with log(xi) pseudo-state vs FD reference.

Builds an FD reference on a grid of xi (the worst-case BVP solved separately at each
xi), uses it to warm-start the network, trains the DGM on the (Z, logxi) slab, then
compares NN vs FD at the canonical xi in {148.4, 0.1, 0.05}. Plots i_d, v' (FD solid,
NN dashed) per xi.

Usage: python compare_uncertainty_nn.py --sigma 0.2 [--fd-supervise] [--precond]
                                        [--nn-iters N] [--label TAG]
"""
import argparse
import os
import sys
import numpy as np

_SHOCK = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "two_capital_shock"))
if _SHOCK not in sys.path:
    sys.path.insert(0, _SHOCK)
import two_capital_shock_model as M          # noqa: E402
from fd_shock import solve_fd_shock          # noqa: E402

EVAL_XIS = [148.4, 0.1, 0.05]


def fd_at(sigma, xi, n=1500):
    P = M.load_calibration("A_g_prime_prime")
    P["sigma_d"] = P["sigma_g"] = sigma; P["xi"] = xi
    return solve_fd_shock(P, n=n), P


def build_fd_ref(sigma, n=1500, n_xi=9, zlo=0.05, zhi=0.95):
    """Stack FD solutions over a logxi grid into supervision arrays."""
    Zc, LXc, Vc, IDc, IGc = [], [], [], [], []
    for lx in np.linspace(-3.0, 5.0, n_xi):
        o, _ = fd_at(sigma, float(np.exp(lx)), n=n)
        m = (o["Z"] >= zlo) & (o["Z"] <= zhi)
        Zc.append(o["Z"][m]); LXc.append(np.full(m.sum(), lx))
        Vc.append(o["v"][m]); IDc.append(o["i_d"][m]); IGc.append(o["i_g"][m])
    return {"Z": np.concatenate(Zc), "logxi": np.concatenate(LXc),
            "v": np.concatenate(Vc), "i_d": np.concatenate(IDc), "i_g": np.concatenate(IGc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--arch", default="forwardnet", choices=["forwardnet", "dgm"])
    ap.add_argument("--fd-supervise", action="store_true")
    ap.add_argument("--precond", action="store_true")
    ap.add_argument("--nn-iters", type=int, default=150000)
    ap.add_argument("--num-neurons", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--fd-n", type=int, default=1500)
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"sigma={args.sigma} arch={args.arch} fd_sup={args.fd_supervise} precond={args.precond} "
          f"iters={args.nn_iters}", flush=True)
    fd_ref = build_fd_ref(args.sigma, n=args.fd_n)
    print(f"[FD ref] {fd_ref['Z'].size} supervision points over 9 xi values", flush=True)
    # FD at the eval xis (the comparison reference)
    fd_eval = {xi: fd_at(args.sigma, xi, n=args.fd_n)[0] for xi in EVAL_XIS}

    OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(OD, exist_ok=True)
    sup = "_fdsup" if args.fd_supervise else "_nosup"
    pc = "_pc" if args.precond else ""
    lbl = f"_{args.label}" if args.label else ""
    tag = f"sigma{args.sigma:g}_{args.arch}{sup}{pc}{lbl}"

    from nn_dgm_uncertainty import solve_nn_uncertainty
    P = M.load_calibration("A_g_prime_prime"); P["sigma_d"] = P["sigma_g"] = args.sigma
    nn = solve_nn_uncertainty(P, iters=args.nn_iters, seed=args.seed, verbose=True,
                              arch=args.arch, precond=args.precond,
                              num_neurons=args.num_neurons, num_layers=args.num_layers,
                              fd_ref=(fd_ref if args.fd_supervise else None), eval_xis=EVAL_XIS)
    hjb, fdl, fgl = nn["final_losses"]
    print(f"[NN/{tag}] final HJB={hjb:.2e} FOC_d={fdl:.2e} FOC_g={fgl:.2e}", flush=True)

    Znn = np.asarray(nn["Z"]).ravel()
    print(f"[shapes] nn['Z']={Znn.shape}", flush=True)
    for xi in EVAL_XIS:
        no = nn["by_xi"][xi]
        print(f"  nn xi={xi}: " + " ".join(f"{k}={np.asarray(no[k]).shape}" for k in ("slope", "i_d", "i_g")), flush=True)
        fo = fd_eval[xi]
        print(f"  FD xi={xi}: Z={np.asarray(fo['Z']).shape} " +
              " ".join(f"{k}={np.asarray(fo[k]).shape}" for k in ("slope", "i_d", "i_g")), flush=True)

    # ---- SAVE the (expensive) NN result FIRST, before any fragile post-processing ----
    npz_path = os.path.join(OD, f"nn_eval_uncertainty_{tag}.npz")
    np.savez(npz_path, Z=Znn, final_losses=np.array(nn["final_losses"]),
             **{f"slope_xi{xi:g}_NN": np.asarray(nn["by_xi"][xi]["slope"]).ravel() for xi in EVAL_XIS},
             **{f"i_d_xi{xi:g}_NN": np.asarray(nn["by_xi"][xi]["i_d"]).ravel() for xi in EVAL_XIS},
             **{f"i_g_xi{xi:g}_NN": np.asarray(nn["by_xi"][xi]["i_g"]).ravel() for xi in EVAL_XIS})
    print(f"[NN] saved {npz_path}", flush=True)

    # accuracy at each eval xi on [0.1,0.9] (robust to any length quirks)
    zc = np.linspace(0.1, 0.9, 17)
    for xi in EVAL_XIS:
        fo = fd_eval[xi]; no = nn["by_xi"][xi]
        fZ = np.asarray(fo["Z"]).ravel()
        for k in ("i_d", "i_g", "slope"):
            fk = np.asarray(fo[k]).ravel(); nk = np.asarray(no[k]).ravel()
            a = np.interp(zc, fZ, fk); b = np.interp(zc, Znn, nk)
            print(f"  xi={xi}: max|{k}_FD-NN| = {np.max(np.abs(a-b)):.3e}", flush=True)

    _plot(fd_eval, nn, tag, OD, args.sigma)


def _plot(fd_eval, nn, tag, OD, sigma):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plo, phi = 0.1, 0.9

    def pts(Z, y, npts=33):
        Z = np.asarray(Z).ravel(); y = np.asarray(y).ravel()
        n = min(len(Z), len(y)); Z, y = Z[:n], y[:n]
        m = (Z >= plo) & (Z <= phi)
        Zc, yc = Z[m], y[m]; s = max(1, len(Zc) // npts)
        return Zc[::s], yc[::s]

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for j, xi in enumerate(EVAL_XIS):
        fo = fd_eval[xi]; no = nn["by_xi"][xi]
        for r, key, ylab in [(0, "i_d", r"$i^d$"), (1, "slope", r"$v'$")]:
            a = ax[r, j]
            zf, yf = pts(fo["Z"], fo[key]); a.plot(zf, yf, "b-o", ms=4, lw=1.5, label="FD")
            zn, yn = pts(nn["Z"], no[key]); a.plot(zn, yn, "r--s", ms=4, lw=1.5, label="NN")
            a.set_title(fr"$\xi={xi:g}$: {ylab}"); a.set_xlabel("Z"); a.set_ylabel(ylab)
            a.legend(); a.grid(alpha=0.3)
    fig.suptitle(f"Robust two-capital: DGM(logxi) vs FD  [{tag}]  sigma={sigma}", fontsize=13)
    fig.tight_layout()
    p = os.path.join(OD, f"uncertainty_FDvsNN_{tag}.png")
    fig.savefig(p, dpi=150); print("saved figure:", p, flush=True)


if __name__ == "__main__":
    main()
