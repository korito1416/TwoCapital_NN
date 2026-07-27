"""Gate metrics for the redesign arms — CORRECTED level metric.

BUG IN v1 (found 2026-07-26): the checkpoint stores the RAW subnet phi, but for anchored arms the
value function is  v(x) = phi(x) - phi(x0) + v0.  v1 loaded phi and reported phi(x0) as "the level".
For an anchored arm phi(x0) is an ARBITRARY OFFSET that cancels out of v entirely, so v1's
"18.9x level pin" measured a quantity with no bearing on the solution.  (It also cannot be 0 by
construction, which is why it looked like a plausible number.)

CORRECT METRIC.  Reconstruct the actual value function per arm:
    anchored  :  v(x) = phi(x) - phi(x0) + v0
    unanchored:  v(x) = phi(x)
and measure the CROSS-SEED SPREAD of v at PROBE STATES x != x0.  For anchored arms v(x0) = v0 by
construction (spread exactly 0 and uninformative), so the informative question is whether the
DIFFERENCES v(x_probe) - v(x0) agree across seeds — which is exactly what re-centering claims to pin.

Reported per arm:
  level_spread_probe : max-min over seeds of v(x_probe), averaged over probe states
  xi_response        : v(X0, xi=0.05) - v(X0, neutral)   <- the welfare-cost-of-robustness object we
                       report to Lars; it must SURVIVE anchoring (the anchor is a level gauge, not a
                       constraint on xi-dependence)
  loss_v             : true unscaled HJB residual (eval mode), mean and MIN over seeds
"""
import os, sys, json, glob
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tensorflow as tf

from params import PARAMS
from feedforward_subnet import FeedForwardSubNet

SUB = "PostDamagePostTech"
LOGK0, Z0, LAM3M = 6.7799, 0.70, 1.0 / 6.0
YHAT = PARAMS["y_upper"]
AG2 = PARAMS["A_g_prime_prime"]
V0 = float(os.environ.get("REDESIGN_ANCHOR_V0", 3.64))
ANCHORED = ("anchor", "separable")          # arm-name substrings that used re-centering
THETA_ARMS = ("theta",)


def load_phi(run_dir):
    net = FeedForwardSubNet({"num_hiddens": [32] * 4, "use_bias": True, "activation": "swish",
                             "dim": 1, "nn_name": "v_nn", "final_activation": "softplus"})
    net.build((None, 7))
    ck = os.path.join(run_dir, SUB, "v_nn_checkpoint_" + SUB)
    if not glob.glob(ck + "*"):
        return None
    net.load_weights(ck).expect_partial()
    return net


def xi_col(arm, xi):
    """theta-mode arms carry theta = 1/xi in the uncertainty column; others carry log(xi)."""
    return (1.0 / xi) if any(t in arm for t in THETA_ARMS) else float(np.log(xi))


def state(logK, Z, Y, l3, col):
    return tf.constant([[logK, Z, Y, l3, AG2, col, col]], tf.float32)


ANCHOR_XI_MODE = os.environ.get("REDESIGN_ANCHOR_XI", "neutral")
ANCHOR_LOGXI_NEUTRAL = float(os.environ.get("REDESIGN_ANCHOR_LOGXI_NEUTRAL", 5.0))


def value(net, arm, logK, Z, Y, l3, xi):
    """Reconstruct the ACTUAL value function, undoing the re-centering the trainer applied.

    !! THE RECONSTRUCTION MUST MIRROR THE TRAINING CONVENTION EXACTLY !!
    The trainer anchors at a FIXED NEUTRAL xi, i.e. it subtracts phi(x0, xi_neutral).  An earlier
    version subtracted phi(x0, xi) at the EVALUATION xi instead, which re-imposed the very bug the
    training fix removed and made the xi-response read exactly 0.00000 for every anchored arm --
    reporting a destroyed quantity as if it were a measurement.  Keep the conventions in lockstep.
    """
    col = xi_col(arm, xi)
    v = float(net(state(logK, Z, Y, l3, col), training=False)[0, 0])
    if any(a in arm for a in ANCHORED):
        if ANCHOR_XI_MODE == "batch":
            col_anchor = col                                     # legacy (buggy) convention
        else:
            col_anchor = (0.0 if any(t in arm for t in THETA_ARMS)  # theta-mode: theta=0 == neutral
                          else ANCHOR_LOGXI_NEUTRAL)
        v0_net = float(net(state(LOGK0, Z0, YHAT, LAM3M, col_anchor), training=False)[0, 0])
        v = v - v0_net + V0
    return v


# probe states, all DIFFERENT from the anchor state x0
PROBES = [
    ("lowK",   6.0, 0.70, YHAT,       LAM3M),
    ("highK",  7.0, 0.70, YHAT,       LAM3M),
    ("lowZ",   LOGK0, 0.35, YHAT,     LAM3M),
    ("highY",  LOGK0, 0.70, 3.2,      LAM3M),
    ("lam3hi", LOGK0, 0.70, YHAT,     1.0 / 3.0),
]


def main(outroot):
    arms = sorted({os.path.basename(d).rsplit("_seed", 1)[0]
                   for d in glob.glob(os.path.join(outroot, "*_seed*"))})
    rep = {}
    print(f"{'arm':20} {'seeds':>5} {'loss_v mean':>12} {'loss_v min':>11} "
          f"{'lvl spread':>11} {'xi response':>12}")
    for arm in arms:
        dirs = sorted(glob.glob(os.path.join(outroot, arm + "_seed*")))
        nets, losses = [], []
        for d in dirs:
            n = load_phi(d)
            if n is None:
                continue
            nets.append((d, n))
            h = os.path.join(d, SUB, "training_history.csv")
            if os.path.exists(h):
                rows = [r.strip().split(",") for r in open(h) if r.strip()][1:]
                if rows:
                    losses.append(float(rows[-1][1]))
        if not nets:
            continue
        # cross-seed spread of the RECONSTRUCTED value at probe states
        spreads = []
        for pname, lk, z, y, l3 in PROBES:
            vals = [value(n, arm, lk, z, y, l3, 0.05) for _, n in nets]
            spreads.append(max(vals) - min(vals))
        # xi-response at X0 (must survive anchoring)
        xir = [value(n, arm, LOGK0, Z0, YHAT, LAM3M, 0.05)
               - value(n, arm, LOGK0, Z0, YHAT, LAM3M, 148.6) for _, n in nets]
        rep[arm] = {"n": len(nets), "loss_v_mean": float(np.mean(losses)) if losses else None,
                    "loss_v_min": float(np.min(losses)) if losses else None,
                    "level_spread_probe": float(np.mean(spreads)),
                    "spread_by_probe": {p[0]: float(s) for p, s in zip(PROBES, spreads)},
                    "xi_response_mean": float(np.mean(xir)), "xi_response": xir}
        print(f"{arm:20} {len(nets):>5} {np.mean(losses):>12.4e} {np.min(losses):>11.4e} "
              f"{np.mean(spreads):>11.5f} {np.mean(xir):>12.5f}")

    base = rep.get("A0_baseline")
    if base:
        print("\n=== GATES (vs A0 baseline) ===")
        for arm, r in rep.items():
            if arm == "A0_baseline":
                continue
            pin = base["level_spread_probe"] / max(r["level_spread_probe"], 1e-12)
            res = r["loss_v_mean"] / base["loss_v_mean"]
            note = ""
            if arm.startswith("A1"):
                note = ("level UNCHANGED as predicted (GN argument holds)" if 0.5 < pin < 2.0
                        else f"level moved {pin:.1f}x — CONTRADICTS the GN argument")
            elif "anchor" in arm or "separable" in arm:
                note = "PASS level pin (>=10x)" if pin >= 10 else f"level pin {pin:.1f}x — BELOW the 10x gate"
            print(f"  {arm:20} level {base['level_spread_probe']:.5f} -> {r['level_spread_probe']:.5f} "
                  f"({pin:5.1f}x)  residual {res:.2f}x  xi-resp {r['xi_response_mean']:+.5f}   {note}")
    json.dump(rep, open(os.path.join(outroot, "arm_report_v2.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(outroot, 'arm_report_v2.json')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output_redesign_20260726")
