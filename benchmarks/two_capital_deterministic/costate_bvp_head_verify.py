"""
VERIFICATION re-run of boundary-anchored-costate-bvp-head with DIFFERENT seeds.

Reported (seed=1, N_ADAM=4000, w_end=10):
  A_d=0.05 (weak-id pocket): ctrl_id=1.3551e-1, treat_id=5.0634e-2 (id_ratio 0.374)
  A_d=0.13 (well-cond):      ctrl_id=1.8348e-4, treat_id=7.6429e-2 (id_ratio 416 -> BACKFIRES)

This script re-runs with seeds [7,11,23] (none == reported seed 1), exact same
config (N_ADAM=4000, tight L-BFGS, w_end=10, N_COLLO=256, fd n=4000), matched
seed/init/budget between CONTROL and TREATED. Reuses make_control / make_treated
verbatim from costate_bvp_head_ab.py so the method is identical.
"""
import json
import numpy as np
import tensorflow as tf
import costate_bvp_head_ab as AB

tf.keras.backend.set_floatx("float32")

SEEDS = [7, 11, 23]
A_DS = [0.05, 0.13]
N_ADAM = 4000
ADAM_LR = AB.ADAM_LR
W_END = 10.0
Znp = AB.Znp


def run_one(make_fn, seed, P, v0, vN, extra):
    tf.random.set_seed(seed); np.random.seed(seed)
    net = AB.make_net()
    if extra is None:
        lg, ev = make_fn(net, P, v0, vN)
    else:
        lg, ev = make_fn(net, P, v0, vN, *extra)
    AB.adam(net, lg, N_ADAM, ADAM_LR)
    AB.lbfgs(net, lg)
    return ev


def main():
    out = []
    for A_d in A_DS:
        P, fd, v_fd, sl_fd, id_fd, v0, vN, qd0, qd1 = AB.build_case(A_d)
        mI = (Znp >= 0.1) & (Znp <= 0.9)
        sl_half_fd = float(np.interp(0.5, Znp, sl_fd))
        print(f"\n##### A_d={A_d} FD i_d interior [{id_fd[mI].min():+.4f},{id_fd[mI].max():+.4f}] "
              f"deinvest={bool((id_fd[mI]<0).any())} v'(0.5)_FD={sl_half_fd:.4f}", flush=True)
        ctrl, treat = [], []
        for seed in SEEDS:
            evc = run_one(AB.make_control, seed, P, v0, vN, None)
            cid, csl, cv, cvph, cR = AB.metrics(evc, v_fd, sl_fd, id_fd)
            evt = run_one(AB.make_treated, seed, P, v0, vN, (qd0, qd1, W_END))
            tid, tsl, tv, tvph, tR = AB.metrics(evt, v_fd, sl_fd, id_fd)
            ctrl.append((cid, csl, cv, cvph)); treat.append((tid, tsl, tv, tvph))
            print(f"  seed={seed}: CTRL id={cid:.4e} sl={csl:.4e} v'(.5)={cvph:.3f} | "
                  f"TREAT id={tid:.4e} sl={tsl:.4e} v'(.5)={tvph:.3f} | "
                  f"id_ratio={tid/max(cid,1e-12):.3f}", flush=True)
        cm = np.mean(np.array(ctrl), axis=0); tm = np.mean(np.array(treat), axis=0)
        cmed = np.median(np.array(ctrl), axis=0); tmed = np.median(np.array(treat), axis=0)
        print(f"  MEAN: CTRL id={cm[0]:.4e} sl={cm[1]:.4e} v'(.5)={cm[3]:.3f} | "
              f"TREAT id={tm[0]:.4e} sl={tm[1]:.4e} v'(.5)={tm[3]:.3f} | "
              f"id_ratio(mean)={tm[0]/max(cm[0],1e-12):.3f}", flush=True)
        print(f"  MEDIAN: CTRL id={cmed[0]:.4e} | TREAT id={tmed[0]:.4e} | "
              f"id_ratio(median)={tmed[0]/max(cmed[0],1e-12):.3f}", flush=True)
        out.append({"A_d": A_d, "seeds": SEEDS, "fd_vph": sl_half_fd,
                    "ctrl_id_mean": float(cm[0]), "treat_id_mean": float(tm[0]),
                    "ctrl_id_med": float(cmed[0]), "treat_id_med": float(tmed[0]),
                    "ctrl_sl_mean": float(cm[1]), "treat_sl_mean": float(tm[1]),
                    "treat_vph_mean": float(tm[3]),
                    "ctrl_id_each": [r[0] for r in ctrl],
                    "treat_id_each": [r[0] for r in treat],
                    "id_ratio_mean": float(tm[0]/max(cm[0],1e-12))})
    print("\nJSON " + json.dumps(out), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
