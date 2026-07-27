"""
compare_hardened_vs_incumbent.py -- DECISIVE grade of an exported HARDENED checkpoint vs the
existing INCUMBENT, on the SAME collocation sample and the SAME stabilized-FD ground truth.

This is the comparison the prior hardened run (job 51248225) never did: it loads two checkpoint
folders (incumbent + hardened-best), evaluates BOTH with the identical PostDamagePostTech_hardened
physics (pde_rhs is byte-identical to production), and reports:

  (i)   HJB residual L2 over the interior BOX and at the small-xi slice logxi=-3 (shared sample);
  (ii)  de-invest controls on the FD grid (lam3=1/6, logxi=log(148.4)) graded by
        benchmarks/post_damage_post_tech/stable_fd_eval.grade  -> di_min_i_d / di_frac_neg /
        di_err_vZ / box_err_i_d, etc.;
  (iii) basic stability (finite fraction).

A CLEAR WIN = hardened beats incumbent on >=1 economically-relevant metric (small-xi HJB residual
OR de-invest i_d sign/depth OR di_err_vZ) WITHOUT regressing the overall box HJB residual.

USAGE:
    python compare_hardened_vs_incumbent.py --incumbent <inc_PostDamagePostTech_dir> \
        --hardened  <hardened_export_dir/best> [--out report.npz] [--seed 0]

Each checkpoint dir must contain
    v_nn_checkpoint_PostDamagePostTech.{index,data-*}
    i_g_nn_checkpoint_PostDamagePostTech.{...}
    i_d_nn_checkpoint_PostDamagePostTech.{...}
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "models"))
sys.path.insert(0, os.path.join(HERE, "benchmarks", "post_damage_post_tech"))

from params import PARAMS, investment_rate_activation  # noqa: E402
import PostDamagePostTech_hardened as H  # noqa: E402
import stable_fd_eval as SFE  # noqa: E402


# FD ground-truth slice the de-invest grading lives on.
LAM3_SLICE = 1.0 / 6.0
LOGXI_SLICE = float(np.log(148.4))


def build_model():
    """Build a hardened model with the production NN configs, supervision OFF (we only evaluate).
    Checkpoints are loaded directly into the nets afterward."""
    num_neurons, num_hidden_layers = 32, 4
    v_cfg = {"num_hiddens": [num_neurons] * num_hidden_layers, "use_bias": True,
             "activation": "swish", "dim": 1, "nn_name": "v_nn", "final_activation": "softplus"}
    ig_cfg = {"num_hiddens": [num_neurons] * num_hidden_layers, "use_bias": True,
              "activation": "tanh", "dim": 1, "nn_name": "i_g_nn",
              "final_activation": investment_rate_activation(PARAMS["θ_g"])}
    id_cfg = {"num_hiddens": [num_neurons] * num_hidden_layers, "use_bias": True,
              "activation": "tanh", "dim": 1, "nn_name": "i_d_nn",
              "final_activation": investment_rate_activation(PARAMS["θ_d"])}
    params = {
        "batch_size": 128, "learning_rates": [1e-4, 1e-4],
        "v_nn_config": v_cfg, "i_g_nn_config": ig_cfg, "i_d_nn_config": id_cfg,
        "num_iterations": 1, "logging_frequency": 1, "verbose": False,
        "pretrained_path": None, "learning_rate_schedule_type": "warmup_cosine",
        "logξ_min": -3.0, "logξ_max": 5.0,
        # evaluation only -> no oracle build, no curriculum, no tensorboard, no export
        "costate_supervision_weight": 0.0, "relative_residual": False,
        "xi_curriculum": False, "tensorboard": False, "export_folder": None,
    }
    m = H.PostDamagePostTechModel(params)
    n_inputs = 7
    m.v_nn.build((None, n_inputs))
    m.i_g_nn.build((None, n_inputs))
    m.i_d_nn.build((None, n_inputs))
    return m


def load_checkpoint(m, ckpt_dir):
    m.v_nn.load_weights(os.path.join(ckpt_dir, "v_nn_checkpoint_PostDamagePostTech"))
    m.i_g_nn.load_weights(os.path.join(ckpt_dir, "i_g_nn_checkpoint_PostDamagePostTech"))
    m.i_d_nn.load_weights(os.path.join(ckpt_dir, "i_d_nn_checkpoint_PostDamagePostTech"))


def _col(x):
    return tf.constant(np.asarray(x, np.float32).reshape(-1, 1))


def eval_on_fd_grid(m):
    """Evaluate i_d, i_g, vZ on the FD grid (lam3=1/6, logxi=log(148.4)) -> dict for stable_fd_eval."""
    d = SFE.load_stable_fd()
    logK, Z, Y = d["logK"], d["Z"], d["Y"]
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    shp = LK.shape
    lk = _col(LK); z = _col(ZZ); y = _col(YY)
    lam3 = _col(np.full(LK.size, LAM3_SLICE))
    logxi = _col(np.full(LK.size, LOGXI_SLICE))
    logR = _col(np.full(LK.size, 3.0))  # logR is inert in pde_rhs (not used); value arbitrary
    # pde_rhs returns dv_dZ as the last element; controls are i_g_nn/i_d_nn on the same X.
    rhs, pv, dv_dY, c, ig_in, id_in, FOC_d, FOC_g, qd, qg, dv_dZ = m.pde_rhs(lk, z, y, logR, lam3, logxi)
    # recover i_d, i_g from inside_log = 1+theta*i
    i_g = (ig_in.numpy().reshape(shp) - 1.0) / m.params["θ_g"]
    i_d = (id_in.numpy().reshape(shp) - 1.0) / m.params["θ_d"]
    vZ = dv_dZ.numpy().reshape(shp)
    return {"logK": logK, "Z": Z, "Y": Y, "i_d": i_d, "i_g": i_g, "vZ": vZ}


def hjb_residual_on_sample(m, sample):
    """RMS HJB residual = rms(rhs - pv) on a fixed collocation sample (shared across both models)."""
    lk, z, y, logR, lam3, logxi = sample
    rhs, pv, *_ = m.pde_rhs(lk, z, y, logR, lam3, logxi)
    res = (rhs - pv).numpy().reshape(-1)
    return res


def make_shared_sample(seed, n=20000):
    """Box-uniform collocation sample (same for both models). Returns the full-box sample plus a
    pure small-xi slice at logxi=-3 (lam3, others box-uniform)."""
    rng = np.random.default_rng(seed)
    lk = rng.uniform(4.0, 7.0, n)
    z = rng.uniform(0.01, 0.99, n)
    y = rng.uniform(0.0, 4.0, n)
    logR = rng.uniform(1.0, 6.0, n)
    lam3 = rng.uniform(0.0, 1.0 / 3.0, n)
    logxi = rng.uniform(-3.0, 5.0, n)
    box = (_col(lk), _col(z), _col(y), _col(logR), _col(lam3), _col(logxi))
    # small-xi slice: logxi pinned at -3, everything else box-uniform
    logxi_sm = np.full(n, -3.0)
    small = (_col(lk), _col(z), _col(y), _col(logR), _col(lam3), _col(logxi_sm))
    return box, small


def grade_one(m, ckpt_dir, box, small, label):
    load_checkpoint(m, ckpt_dir)
    res_box = hjb_residual_on_sample(m, box)
    res_sm = hjb_residual_on_sample(m, small)
    fd_out = eval_on_fd_grid(m)
    sfe = SFE.grade(fd_out)
    rep = {
        "label": label,
        "hjb_rms_box": float(np.sqrt(np.mean(res_box ** 2))),
        "hjb_rms_smallxi": float(np.sqrt(np.mean(res_sm ** 2))),
        "frac_finite_box": float(np.mean(np.isfinite(res_box))),
        "di_min_i_d": sfe["di_min_i_d"],
        "di_frac_neg": sfe["di_frac_neg"],
        "di_mean_i_d": sfe["di_mean_i_d"],
        "di_err_vZ": sfe["di_err_vZ"],
        "di_err_i_d": sfe["di_err_i_d"],
        "box_err_i_d": sfe["box_err_i_d"],
        "box_err_i_g": sfe["box_err_i_g"],
        "di_FD_min_i_d": sfe["di_FD_min_i_d"],
        "di_FD_frac_neg": sfe["di_FD_frac_neg"],
    }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--hardened", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()

    box, small = make_shared_sample(args.seed, args.n)
    m = build_model()

    inc = grade_one(m, args.incumbent, box, small, "INCUMBENT")
    har = grade_one(m, args.hardened, box, small, "HARDENED")

    def show(rep):
        print(f"\n=== {rep['label']} ===", flush=True)
        for k in ("hjb_rms_box", "hjb_rms_smallxi", "frac_finite_box",
                  "di_min_i_d", "di_frac_neg", "di_mean_i_d", "di_err_vZ",
                  "di_err_i_d", "box_err_i_d", "box_err_i_g"):
            print(f"    {k:18s} = {rep[k]:+.6e}", flush=True)

    show(inc)
    show(har)

    print("\n=== FD TARGETS (de-invest region) ===", flush=True)
    print(f"    di_FD_min_i_d   = {inc['di_FD_min_i_d']:+.5f}", flush=True)
    print(f"    di_FD_frac_neg  = {inc['di_FD_frac_neg']:.3f}", flush=True)

    print("\n=== HEAD-TO-HEAD (hardened - incumbent; negative=better for errors) ===", flush=True)
    wins = []
    for k, lower_better in (("hjb_rms_box", True), ("hjb_rms_smallxi", True),
                            ("di_err_vZ", True), ("di_err_i_d", True)):
        delta = har[k] - inc[k]
        better = (delta < 0) if lower_better else (delta > 0)
        print(f"    {k:18s}: inc={inc[k]:+.4e}  har={har[k]:+.4e}  d={delta:+.4e}  "
              f"{'BETTER' if better else 'worse'}", flush=True)
        if k != "hjb_rms_box":  # box residual is the no-regress guard, not a win on its own
            wins.append((k, better))
    # de-invest sign/depth: closer to FD min (more negative, more frac_neg) is better
    print(f"    di_min_i_d        : inc={inc['di_min_i_d']:+.4e}  har={har['di_min_i_d']:+.4e}  "
          f"(FD={inc['di_FD_min_i_d']:+.4e}; more-negative=better)", flush=True)
    print(f"    di_frac_neg       : inc={inc['di_frac_neg']:.3f}  har={har['di_frac_neg']:.3f}  "
          f"(FD={inc['di_FD_frac_neg']:.3f}; higher=better)", flush=True)
    deinvest_better = (har["di_min_i_d"] < inc["di_min_i_d"]) or (har["di_frac_neg"] > inc["di_frac_neg"])
    wins.append(("de_invest_i_d", deinvest_better))

    no_regress = har["hjb_rms_box"] <= inc["hjb_rms_box"] * 1.05  # <=5% box-residual slack
    any_win = any(b for _, b in wins)
    verdict = "CLEAR WIN" if (any_win and no_regress) else ("NO-REGRESS-FAIL" if any_win else "NO WIN")
    print(f"\n=== VERDICT: {verdict} ===", flush=True)
    print(f"    economically-relevant wins: {[k for k, b in wins if b]}", flush=True)
    print(f"    box-residual no-regress (<=5%): {no_regress}", flush=True)

    if args.out:
        np.savez(args.out, incumbent=inc, hardened=har, verdict=verdict)
        print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
