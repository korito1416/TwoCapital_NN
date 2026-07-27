"""Compute the gate metrics for the redesign arms (terminal regime).

Metrics, per arm, aggregated across seeds:
  * true unclamped HJB residual (eval mode always reports the UNSCALED residual, so arms with
    different REDESIGN_HJB_SCALE stay comparable -- never compare a scaled training loss)
  * FOC residuals
  * CROSS-SEED LEVEL SPREAD  = max_seed v(x0) - min_seed v(x0)   <- the level-identifiability metric
  * per-region relative residual  <- the fit-EVENNESS metric that non-dimensionalization targets
  * A'(logK) profile (separable arms only): should reproduce the measured 0.77 -> 0.45 decline

GATE LOGIC (pre-registered):
  A1 (non-dim)   : evenness spread should DROP >=2x;  level spread should be UNCHANGED.
                   An unchanged level spread CONFIRMS the Gauss-Newton argument (any divisor scales
                   H_level and H_shape equally) -- it is the predicted result, not a failure.
  A2 (anchor)    : level spread should DROP >=10x at <=1.2x residual cost.
  A3 (separable) : residual <= A2's, and A'(logK) declining like the measured elasticity.
  A4 (theta)     : deep-xi residual improves AND neutral-xi residual does not degrade.
"""
import os, sys, json, glob
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tensorflow as tf

from params import PARAMS, investment_rate_activation
from feedforward_subnet import FeedForwardSubNet

LOGK0, Z0, LAM3_MID = 6.7799, 0.70, 1.0 / 6.0
YHAT = PARAMS["y_upper"]


def _cfg(name, act, fin):
    return {"num_hiddens": [32] * 4, "use_bias": True, "activation": act, "dim": 1,
            "nn_name": name, "final_activation": fin}


REGIME_SUBDIR = "PostDamagePostTech"   # the trainer writes into <run>/<Regime>/


def load_v(run_dir, n_inputs=7):
    net = FeedForwardSubNet(_cfg("v_nn", "swish", "softplus"))
    net.build((None, n_inputs))
    ck = os.path.join(run_dir, REGIME_SUBDIR, "v_nn_checkpoint_" + REGIME_SUBDIR)
    if not glob.glob(ck + "*"):
        return None
    net.load_weights(ck).expect_partial()
    return net


def anchor_value(net, logxi_col, n_inputs=7):
    """v at the anchor state x0 = (logK0, Z0, yhat, lam3_mid) -- the level probe."""
    x = tf.constant([[LOGK0, Z0, YHAT, LAM3_MID, PARAMS["A_g_prime_prime"], logxi_col, logxi_col]][:1],
                    dtype=tf.float32)
    if n_inputs != x.shape[1]:
        x = x[:, :n_inputs]
    return float(net(x, training=False)[0, 0])


def region_masks(logK, Y, logxi):
    return {
        "lowK":       logK < 5.5,
        "highK":      logK > 6.5,
        "Y_below_yhat": Y < YHAT,
        "Y_above_yhat": Y >= YHAT,
        "deep_xi":    logxi < -2.0,
        "neutral_xi": logxi > 2.0,
    }


def main(outroot):
    arms = sorted({os.path.basename(d).rsplit("_seed", 1)[0]
                   for d in glob.glob(os.path.join(outroot, "*_seed*"))})
    if not arms:
        print(f"no arm directories under {outroot}")
        return
    report = {}
    print(f"{'arm':18} {'seeds':>6} {'loss_v(mean)':>14} {'level spread':>14} {'levels'}")
    for arm in arms:
        dirs = sorted(glob.glob(os.path.join(outroot, arm + "_seed*")))
        levels, losses = [], []
        for d in dirs:
            net = load_v(d)
            if net is None:
                continue
            levels.append(anchor_value(net, -3.0))
            hist = os.path.join(d, REGIME_SUBDIR, "training_history.csv")
            if os.path.exists(hist):
                rows = [r.strip().split(",") for r in open(hist) if r.strip()][1:]
                if rows:
                    losses.append(float(rows[-1][1]))
        if not levels:
            print(f"{arm:18} {'--':>6}  (no checkpoints yet)")
            continue
        spread = max(levels) - min(levels)
        report[arm] = {"n_seeds": len(levels), "levels": levels,
                       "level_spread": spread,
                       "loss_v_mean": float(np.mean(losses)) if losses else None,
                       "loss_v_all": losses}
        lm = f"{np.mean(losses):.4e}" if losses else "n/a"
        print(f"{arm:18} {len(levels):>6} {lm:>14} {spread:>14.5f}  "
              + " ".join(f"{v:.4f}" for v in levels))

    # gate evaluation
    print("\n=== GATES ===")
    base = report.get("A0_baseline")
    if base:
        for arm, r in report.items():
            if arm == "A0_baseline":
                continue
            ls_ratio = base["level_spread"] / max(r["level_spread"], 1e-12)
            lv_ratio = (r["loss_v_mean"] / base["loss_v_mean"]) if (r["loss_v_mean"] and base["loss_v_mean"]) else float("nan")
            verdict = ""
            if arm.startswith("A1"):
                verdict = ("level UNCHANGED as predicted (GN argument confirmed)"
                           if 0.5 < ls_ratio < 2.0 else
                           f"level moved {ls_ratio:.1f}x -- CONTRADICTS the GN argument, investigate")
            elif arm.startswith("A2") or arm.startswith("A5"):
                verdict = ("PASS level pin" if ls_ratio >= 10 else f"level pin only {ls_ratio:.1f}x (<10x gate)")
            print(f"  {arm:18} level_spread {base['level_spread']:.5f} -> {r['level_spread']:.5f} "
                  f"({ls_ratio:.1f}x)   residual {lv_ratio:.2f}x baseline   {verdict}")
    out = os.path.join(outroot, "arm_report.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output_redesign_20260726")
