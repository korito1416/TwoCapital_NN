"""
ADVERSARIAL VERIFICATION of the 'full-gram-metric-richness-bridge' result.

Reported verdict: the full-Gram (TIER3) damped-Newton step on the STRONG-form
residual BACKFIRES -- it drives the RMS residual to ~1e-4..1e-5 while the TRUE
error vs FD (max|i_d-FD|, ||v-FD||_inf) is FLAT or EXPLODES by 1-4 orders of
magnitude. The richer-metric hypothesis predicted TIER0>=T1>=T2>=T3 on true error;
reported FALSE in every cell.

This script re-runs the A/B with MULTIPLE SEEDS to check the pattern reproduces
and is not seed noise. We add genuine seed variation to BOTH starts:
  - FD-seeded+noise: 5% noise with seeds {0,1,2,3,4}
  - from-scratch + noise: linear-blend guess + small noise, seeds {0,1,2,3,4}
Same budget (n_iter, lr, lam) and same init across tiers within a seed -> fair A/B.

We judge ONLY true error vs FD, never the (non-comparable) residual norm.
"""

import os
import numpy as np

import two_capital_model as M
from theta_sensitivity import solve_fd
from precond_conditioning_study import make_P, residuals, jacobian
from metric_richness_bridge import (
    fd_truth, true_errors, run_tier, seed_scratch, seed_fd,
)

np.set_printoptions(precision=4, suppress=False)


def seed_scratch_noisy(Zc, P, noise, rng):
    u = seed_scratch(Zc, P)
    n = len(Zc)
    v = u[:n]; i_d = u[n:2*n]; i_g = u[2*n:]
    v = v + noise * (np.std(v) + 1e-6) * rng.standard_normal(n)
    i_d = i_d + noise * (np.std(i_d) + 1e-3) * rng.standard_normal(n)
    i_g = i_g + noise * (np.std(i_g) + 1e-3) * rng.standard_normal(n)
    return np.concatenate([v, i_d, i_g])


def run_one_seed(half, seed, n=36, n_iter=40, lr=2e-3, lam=1e-7):
    P = make_P(half)
    Zc = np.linspace(0.08, 0.92, n)
    dZ = Zc[1] - Zc[0]
    v0, vN = M.boundary_values(P)
    fd = fd_truth(P, Zc)

    tiers = [(0, "TIER0 gradient"), (1, "TIER1 diag 1/|mu|"),
             (2, "TIER2 block-Gram"), (3, "TIER3 FULL-Gram")]

    out = {}
    starts = {
        "FDseed+noise": seed_fd(Zc, P, fd, noise=0.05, rng=np.random.default_rng(seed)),
        "scratch+noise": seed_scratch_noisy(Zc, P, noise=0.05, rng=np.random.default_rng(1000+seed)),
    }
    for label, u0 in starts.items():
        e0 = true_errors(u0, Zc, fd)
        rec = {"init": e0}
        for tier, tname in tiers:
            u, r0, rf, err = run_tier(tname, u0, Zc, dZ, v0, vN, P, fd,
                                      n_iter, tier, lr=lr, lam=lam)
            rec[tier] = {"err": err, "res0": r0, "resf": rf}
        out[label] = rec
    return out, fd


def main():
    SEEDS = [0, 1, 2, 3, 4]
    for half in (False, True):
        tag = "half" if half else "base"
        print(f"\n############ REGIME {tag}  (n=36, n_iter=40, seeds={SEEDS}) ############")
        # aggregate id_inf and v_inf per (start, tier) across seeds
        agg = {}
        fd_ref = None
        for s in SEEDS:
            out, fd = run_one_seed(half, s)
            fd_ref = fd
            for label, rec in out.items():
                for tier in range(4):
                    agg.setdefault((label, tier), {"id": [], "v": [], "res": []})
                    agg[(label, tier)]["id"].append(rec[tier]["err"]["id_inf"])
                    agg[(label, tier)]["v"].append(rec[tier]["err"]["v_inf"])
                    agg[(label, tier)]["res"].append(rec[tier]["resf"])
                agg.setdefault((label, "init"), {"id": [], "v": []})
                agg[(label, "init")]["id"].append(rec["init"]["id_inf"])
                agg[(label, "init")]["v"].append(rec["init"]["v_inf"])
        print(f"  FD truth i_d in [{fd_ref['i_d'].min():+.4f},{fd_ref['i_d'].max():+.4f}]")
        tnames = {0: "TIER0 gradient", 1: "TIER1 diag1/|mu|",
                  2: "TIER2 block-Gram", 3: "TIER3 FULL-Gram"}
        for label in ["FDseed+noise", "scratch+noise"]:
            ini = agg[(label, "init")]
            print(f"\n  --- start={label} ---")
            print(f"    init                 id_inf={np.mean(ini['id']):.3e}  "
                  f"v_inf={np.mean(ini['v']):.3e}")
            for tier in range(4):
                a = agg[(label, tier)]
                idm, ids = np.mean(a["id"]), np.std(a["id"])
                vm, vs = np.mean(a["v"]), np.std(a["v"])
                rm = np.mean(a["res"])
                print(f"    {tnames[tier]:<18} "
                      f"id_inf={idm:.3e}+-{ids:.1e}  v_inf={vm:.3e}+-{vs:.1e}  "
                      f"RMSres_final={rm:.2e}")
            # monotonicity per seed: does richer metric lower true error?
            mono_count = 0
            t3_worse_than_t0 = 0
            for k in range(len(SEEDS)):
                seq = [agg[(label, t)]["id"][k] for t in range(4)]
                if all(seq[i] >= seq[i+1] - 1e-12 for i in range(3)):
                    mono_count += 1
                if seq[3] > seq[0] + 1e-12:
                    t3_worse_than_t0 += 1
            print(f"    -> monotone-decreasing(T0>=T1>=T2>=T3) in {mono_count}/{len(SEEDS)} seeds; "
                  f"TIER3 WORSE than TIER0 in {t3_worse_than_t0}/{len(SEEDS)} seeds")


if __name__ == "__main__":
    main()
