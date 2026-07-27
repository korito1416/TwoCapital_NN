"""
FD solution of the robust two-capital model across the robustness multiplier xi.

Robustness adds the drag -1/(2 xi)[(1-Z)^2 sd^2 q_d^2 + Z^2 sg^2 q_g^2] to the shock
HJB (derivation.tex, referee-passed); xi -> inf recovers the non-robust shock model.
We solve at xi in {148.4 (~ no robustness), 0.1, 0.05 (strong)} and report the
worst-case Brownian drift distortions h_d*, h_g* = -(1/xi)(...)q_j recovered ex post.
The robustness effect scales as sigma^2/xi, so it is tiny at the calibrated sigma=0.01;
a sigma=0.2 figure is added to make the mechanism visible.

This is the FD reference for the DGM neural net (which will carry log(xi) as a pseudo
state). Login node: module load python/anaconda-2021.05.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# the shock model + FD solver (now carry optional xi robustness, default off)
_SHOCK = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "two_capital_shock"))
if _SHOCK not in sys.path:
    sys.path.insert(0, _SHOCK)
import two_capital_shock_model as M          # noqa: E402
from fd_shock import solve_fd_shock          # noqa: E402

OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OD, exist_ok=True)

XIS = [148.4, 0.1, 0.05]
COLORS = {148.4: "b", 0.1: "g", 0.05: "r"}


def augment(d, P):
    Z = d["Z"]
    d["q_d"] = 1.0 - Z * d["slope"]
    d["q_g"] = 1.0 + (1.0 - Z) * d["slope"]
    d["Abar"] = (1.0 - Z) * P["A_d"] + Z * P["A_g"]
    d["C_over_Y"] = d["c"] / d["Abar"]
    h_d, h_g = M.worst_case_drifts(Z, d["slope"], P)
    d["h_d"], d["h_g"] = h_d, h_g
    return d


def solve(sigma, xi, n=2000):
    P = M.load_calibration("A_g_prime_prime")
    P["sigma_d"] = P["sigma_g"] = sigma
    P["xi"] = xi
    o = augment(solve_fd_shock(P, n=n), P)
    o["resid"] = o["max_abs_residual"]
    return o, P


def compat_check():
    """xi=inf (default) must reproduce the non-robust shock model exactly."""
    P = M.load_calibration("A_g_prime_prime"); P["sigma_d"] = P["sigma_g"] = 0.2
    base = solve_fd_shock(P, n=2000)                       # no xi -> default inf
    P2 = dict(P); P2["xi"] = np.inf
    rob = solve_fd_shock(P2, n=2000)
    d = max(abs(np.interp(z, base["Z"], base["slope"]) - np.interp(z, rob["Z"], rob["slope"]))
            for z in np.linspace(0.1, 0.9, 17))
    print(f"[compat] xi=inf vs no-xi (sigma=0.2): max|slope diff| = {d:.2e}", flush=True)


def make_fig(sigma, tag):
    sols = {xi: solve(sigma, xi)[0] for xi in XIS}
    for xi in XIS:
        o = sols[xi]
        print(f"  sigma={sigma} xi={xi}: resid={o['resid']:.2e} "
              f"i_d(.7)={np.interp(0.7,o['Z'],o['i_d']):+.5f} "
              f"v'(.7)={np.interp(0.7,o['Z'],o['slope']):+.4f} "
              f"max|h_d|={np.max(np.abs(o['h_d'])):.2e} max|h_g|={np.max(np.abs(o['h_g'])):.2e}",
              flush=True)
    plo, phi = 0.1, 0.9

    def pts(o, key, npts=33):
        m = (o["Z"] >= plo) & (o["Z"] <= phi)
        Z, y = o["Z"][m], o[key][m]
        s = max(1, len(Z) // npts)
        return Z[::s], y[::s]

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    panels = [("i_d", r"Dirty investment $i^d$"), ("i_g", r"Green investment $i^g$"),
              ("q_d", r"Marginal value dirty $q_d$"), ("slope", r"Value slope $v'(Z)$"),
              ("h_d", r"Worst-case dirty drift $h_d^\ast$"),
              ("h_g", r"Worst-case green drift $h_g^\ast$")]
    for a, (key, title) in zip(ax.ravel(), panels):
        for xi in XIS:
            z, y = pts(sols[xi], key)
            a.plot(z, y, COLORS[xi] + "-o", ms=3.5, lw=1.4, label=fr"$\xi={xi:g}$")
        a.axhline(0, color="k", lw=0.6, ls=":")
        a.set_xlabel("Z (green capital share)"); a.set_title(title)
        a.legend(); a.grid(alpha=0.3)
    fig.suptitle(f"Robust two-capital (FD): xi sweep at sigma={sigma}  "
                 f"(xi=148.4 ~ no robustness, xi=0.05 strong)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(OD, f"uncertainty_FD_{tag}.png")
    fig.savefig(p, dpi=150); print("saved", p, flush=True)


if __name__ == "__main__":
    compat_check()
    print("\n=== sigma=0.01 (calibration; robustness effect ~ sigma^2/xi, small) ===")
    make_fig(0.01, "sigma0.01")
    print("\n=== sigma=0.2 (robustness effect made visible) ===")
    make_fig(0.2, "sigma0.2")
