"""
Operator-metric-richness diagnostic bridge (diagonal -> block-Gram -> full Gram).

PURPOSE (analytic, FD-discretization only -- NO NN training noise):
  Prove that the precision floor on the two-capital benchmark is set by HOW MUCH of
  the operator Gram J^T J is inverted in the update metric, NOT by hyperparameters.
  We solve the EXACT FD-discretized 3-loss system
        F(u) = [ R(u) ; FOC_d(u) ; FOC_g(u) ],   u = [v(0..n-1), i_d, i_g]
  by an iterative update u <- u - step, where 'step' uses progressively richer metrics:

    TIER 0  gradient        : step = lr * J^T F                (no metric)
    TIER 1  diagonal (#1)    : step = lr * diag(W) J^T F,  W = 1/(|mu_Z|+eps) on R rows,
                               i.e. the diagonal-1/|mu_Z| preconditioner = the rank-1
                               truncation of the right metric (Lars's bridge claim).
    TIER 2  block-Gram       : per-block damped Gauss-Newton: for each residual block b
                               solve (J_b^T J_b + lam I) s_b = J_b^T F_b on its OWN
                               column support; sum the three contributions (off-diagonal
                               cross-block couplings DROPPED).
    TIER 3  full Gram        : full damped Newton  (J^T J + lam I)^{-1} J^T F
                               (ALL off-diagonal couplings kept).

  TRUE-ERROR METRIC (never the residual/loss norm):
        max_Z |i_d - i_d_FD|  and  ||v - v_fd||_inf   vs solve_fd ground truth.

  Expectation (the ENGD mechanism): error decreases monotonically with metric richness,
  TIER 3 (full Gram) reaching the lowest floor; the gap TIER1->TIER3 is exactly the
  off-diagonal operator coupling the diagonal 1/|mu_Z| preconditioner cannot see.

Runs login-node / srun (numpy+scipy only). float64 throughout.
"""

import os
import numpy as np

import two_capital_model as M
from theta_sensitivity import solve_fd
from precond_conditioning_study import (make_P, residuals, jacobian)

OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OD, exist_ok=True)

np.set_printoptions(precision=4, suppress=False)


# ---------------------------------------------------------------------------
# FD ground truth interpolated onto the coarse working grid Zc
# ---------------------------------------------------------------------------
def fd_truth(P, Zc):
    fine = solve_fd(P, n=4000)
    Zf = fine["Z"]
    return {
        "v": np.interp(Zc, Zf, fine["v"]),
        "slope": np.interp(Zc, Zf, fine["slope"]),
        "i_d": np.interp(Zc, Zf, fine["i_d"]),
        "i_g": np.interp(Zc, Zf, fine["i_g"]),
        "fine": fine,
    }


def true_errors(u, Zc, fd):
    n = len(Zc)
    v = u[:n]; i_d = u[n:2 * n]; i_g = u[2 * n:]
    return {
        "v_inf": float(np.max(np.abs(v - fd["v"]))),
        "id_inf": float(np.max(np.abs(i_d - fd["i_d"]))),
        "ig_inf": float(np.max(np.abs(i_g - fd["i_g"]))),
    }


# ---------------------------------------------------------------------------
# The update tiers.  Each returns the step vector to SUBTRACT from u.
# ---------------------------------------------------------------------------
def step_gradient(J, F, lr):
    return lr * (J.T @ F)


def step_diagonal(J, F, mu, lr, eps=1e-4):
    """TIER 1: diagonal 1/|mu_Z| weight on the HJB-residual (R) rows = rank-1 metric.
    Weighted Gauss-Newton with a DIAGONAL row metric W (no column coupling):
        step = lr * J^T W F ,  W=diag(w),  w[R]=1/(|mu|+eps), w[FOC]=1.
    """
    n = len(mu)
    w = np.ones(3 * n)
    w[:n] = 1.0 / (np.abs(mu) + eps)
    # normalise so the overall step scale is comparable to TIER 0
    w = w / np.mean(w)
    return lr * (J.T @ (w * F))


def step_block_gram(J, F, n, lam):
    """TIER 2: block-diagonal Gram. For each residual block b in {R, FOC_d, FOC_g},
    take its rows J_b and solve a damped GN on the FULL variable set but using ONLY
    that block's rows; SUM the three steps. This keeps each block's own column Gram
    (incl. coupling among that block's own contributions) but DROPS cross-block
    off-diagonal Gram entries J_a^T J_b (a != b)."""
    N = J.shape[1]
    step = np.zeros(N)
    for b in range(3):
        sl = slice(b * n, (b + 1) * n)
        Jb = J[sl, :]
        Fb = F[sl]
        G = Jb.T @ Jb
        g = Jb.T @ Fb
        step += np.linalg.solve(G + lam * np.eye(N), g)
    return step


def step_full_gram(J, F, lam):
    """TIER 3: full damped Newton  (J^T J + lam I)^{-1} J^T F."""
    N = J.shape[1]
    G = J.T @ J
    g = J.T @ F
    return np.linalg.solve(G + lam * np.eye(N), g)


# ---------------------------------------------------------------------------
# Iterate a tier with a simple Armijo-ish backtracking on ||F||
# ---------------------------------------------------------------------------
def run_tier(name, u0, Zc, dZ, v0, vN, P, fd, n_iter, tier,
             lr=1e-3, lam=1e-6, verbose=False):
    n = len(Zc)
    u = u0.copy()
    F, mu = residuals(u, Zc, dZ, v0, vN, P)
    res0 = np.sqrt(np.mean(F**2))
    for it in range(n_iter):
        J = jacobian(u, Zc, dZ, v0, vN, P)
        F, mu = residuals(u, Zc, dZ, v0, vN, P)
        if tier == 0:
            s = step_gradient(J, F, lr)
        elif tier == 1:
            s = step_diagonal(J, F, mu, lr)
        elif tier == 2:
            s = step_block_gram(J, F, n, lam)
        elif tier == 3:
            s = step_full_gram(J, F, lam)
        # backtracking line search on ||F||_2 to keep iterations stable
        f0 = np.linalg.norm(F)
        a = 1.0
        for _ in range(30):
            ut = u - a * s
            Ft, _ = residuals(ut, Zc, dZ, v0, vN, P)
            if np.linalg.norm(Ft) < f0 and np.all(np.isfinite(Ft)):
                break
            a *= 0.5
        else:
            a = 0.0
        u = u - a * s
    F, mu = residuals(u, Zc, dZ, v0, vN, P)
    resf = np.sqrt(np.mean(F**2))
    err = true_errors(u, Zc, fd)
    return u, res0, resf, err


def seed_scratch(Zc, P):
    """A bad (perturbation-slope) initial guess: linear-blend boundary v, FOC controls."""
    n = len(Zc)
    v0, vN = M.boundary_values(P)
    v = v0 + (vN - v0) * Zc       # crude linear guess
    p = np.gradient(v, Zc)
    i_d, i_g, _ = M.controls(Zc, p, P)
    return np.concatenate([v, i_d, i_g])


def seed_fd(Zc, P, fd, noise=0.0, rng=None):
    """FD-seeded start (optionally perturbed) -- tests local convergence floor."""
    n = len(Zc)
    v = fd["v"].copy(); i_d = fd["i_d"].copy(); i_g = fd["i_g"].copy()
    if noise > 0 and rng is not None:
        v = v + noise * np.std(v) * rng.standard_normal(n)
        i_d = i_d + noise * (np.std(i_d) + 1e-3) * rng.standard_normal(n)
        i_g = i_g + noise * (np.std(i_g) + 1e-3) * rng.standard_normal(n)
    return np.concatenate([v, i_d, i_g])


def study(half, n=60, n_iter=80):
    tag = "half" if half else "base"
    P = make_P(half)
    Zc = np.linspace(0.08, 0.92, n)
    dZ = Zc[1] - Zc[0]
    v0, vN = M.boundary_values(P)
    fd = fd_truth(P, Zc)

    print(f"\n############### REGIME = {tag}  (n={n}, n_iter={n_iter}) ###############")
    print(f"FD truth: i_d in [{fd['i_d'].min():+.4f},{fd['i_d'].max():+.4f}], "
          f"i_g in [{fd['i_g'].min():+.4f},{fd['i_g'].max():+.4f}], "
          f"FD-on-grid residual floor reported below")
    # FD residual of the FD solution on THIS coarse grid (discretization floor)
    u_fd = np.concatenate([fd["v"], fd["i_d"], fd["i_g"]])
    Ffd, _ = residuals(u_fd, Zc, dZ, v0, vN, P)
    print(f"  ||F(u_fd)||_RMS on coarse grid = {np.sqrt(np.mean(Ffd**2)):.3e}  "
          f"(coarse-grid discretization floor; truth errors below are vs fine FD)")

    tiers = [(0, "TIER0 gradient"), (1, "TIER1 diagonal 1/|mu| (#1)"),
             (2, "TIER2 block-Gram"), (3, "TIER3 full Gram (Newton)")]

    results = {}
    for label, mkseed in [("from-scratch", lambda: seed_scratch(Zc, P)),
                          ("FD-seeded+noise", lambda: seed_fd(Zc, P, fd, noise=0.05,
                                                              rng=np.random.default_rng(0)))]:
        print(f"\n----- start = {label} -----")
        u0 = mkseed()
        e0 = true_errors(u0, Zc, fd)
        print(f"  initial true error: max|i_d-FD|={e0['id_inf']:.4e}  "
              f"||v-FD||_inf={e0['v_inf']:.4e}")
        for tier, tname in tiers:
            u, r0, rf, err = run_tier(tname, u0, Zc, dZ, v0, vN, P, fd,
                                      n_iter, tier, lr=2e-3, lam=1e-7)
            results[(label, tier)] = err
            print(f"  {tname:<28} max|i_d-FD|={err['id_inf']:.4e}  "
                  f"max|i_g-FD|={err['ig_inf']:.4e}  ||v-FD||_inf={err['v_inf']:.4e}  "
                  f"(RMSres {r0:.2e}->{rf:.2e})")

    return tag, results


def main():
    summary = {}
    for half in (False, True):
        tag, res = study(half)
        summary[tag] = res

    print("\n\n================ MONOTONICITY CHECK (max|i_d - FD|) ================")
    print("Expect TIER0 >= TIER1 >= TIER2 >= TIER3 if richer metric => lower floor.")
    for tag, res in summary.items():
        for label in ["from-scratch", "FD-seeded+noise"]:
            seq = [res[(label, t)]["id_inf"] for t in range(4)]
            mono = all(seq[i] >= seq[i + 1] - 1e-9 for i in range(3))
            print(f"  [{tag:>4} | {label:<16}] id_inf by tier = "
                  f"{['%.3e' % x for x in seq]}  monotone_decreasing={mono}")
    print("\n(NOTE: residual RMS values are L2(mu) of the strong residual and are NOT "
          "comparable across tiers/norms -- we compare ONLY true error vs FD.)")


if __name__ == "__main__":
    main()
