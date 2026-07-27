"""COMMON-YARDSTICK evaluation: score every redesign arm on ONE identical benchmark.

WHY THIS EXISTS (the comparability trap, found 2026-07-26).
Each arm's `training_history.csv` reports a residual computed on THAT ARM'S OWN sampling
distribution, via large_sample_validation -> model.sample().  Two things therefore differ between
arms and make the logged numbers non-comparable:

  1. THE LOSS FUNCTION.  With REDESIGN_HJB_SCALE the training objective is divided by a scale.
     (Mitigated already: eval mode returns the TRUE UNSCALED residual — but only mitigates #1.)
  2. THE EVALUATION DISTRIBUTION.  The theta arms sample theta ~ mixture, putting ~80% of the batch
     at xi < 0.1, while the logxi arms put only ~8.6% there.  So the theta arm's reported residual
     is an average over a DIFFERENT, much harder region.  Comparing it to the baseline's number
     compares two different questions, not two answers to the same question.

THE FIX.  Draw ONE fixed state sample (seeded) and ONE fixed grid of ECONOMIC xi values.  Feed every
arm the identical economic state, converting only the uncertainty column into that arm's own input
convention (logxi vs theta = 1/xi).  Score with the same unscaled residual operator.  Report per-xi
so the deep-uncertainty and near-neutral ends can be read separately instead of being averaged into
one number that hides the trade-off.
"""
import os, sys, json, glob
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
VAR = f"{ROOT}/models_onejump_redesign"

# economic xi grid the comparison is defined on (identical for every arm)
XI_GRID = [148.6, 10.0, 1.0, 0.3, 0.1, 0.05]
N_STATES = 4096
SEED = 20260726
SUB = "PostDamagePostTech"

# arm -> the switches it was trained with (must match, or the value path is wrong)
ARM_SWITCHES = {
    "A0_baseline":     {},
    "A1_nondim":       {"REDESIGN_HJB_SCALE": "natural"},
    "A2_anchor":       {"REDESIGN_ANCHOR": "recenter", "REDESIGN_ANCHOR_XI": "neutral"},
    "A3_separable":    {"REDESIGN_DETREND": "on", "REDESIGN_ANCHOR": "recenter",
                        "REDESIGN_ANCHOR_XI": "neutral"},
    "A4_theta":        {"REDESIGN_XI_PARAM": "theta", "REDESIGN_XI_SAMPLING": "mixture"},
    "A5_theta_anchor": {"REDESIGN_XI_PARAM": "theta", "REDESIGN_XI_SAMPLING": "mixture",
                        "REDESIGN_ANCHOR": "recenter", "REDESIGN_ANCHOR_XI": "neutral"},
    # controls that decide whether "non-dimensionalisation" is anything more than a reweighting
    "A6_reweight":     {"REDESIGN_HJB_SCALE": "none", "REDESIGN_HJB_WEIGHT": "8.14"},
    "A7_nondim_norm":  {"REDESIGN_HJB_SCALE": "natural", "REDESIGN_HJB_WEIGHT": "0.1228"},
    "S0_baseline":     {},
    "S1_focxi":        {"REDESIGN_FOCXI_WEIGHT": "1.0"},
    "S2_sens":         {"REDESIGN_SENS_WEIGHT": "1.0"},
    "S3_both":         {"REDESIGN_SENS_WEIGHT": "1.0", "REDESIGN_FOCXI_WEIGHT": "1.0"},
    "S4_both_anchor":  {"REDESIGN_SENS_WEIGHT": "1.0", "REDESIGN_FOCXI_WEIGHT": "1.0",
                        "REDESIGN_ANCHOR": "recenter", "REDESIGN_ANCHOR_XI": "neutral"},
}


def common_states(n=N_STATES, seed=SEED, post_damage=True):
    """ONE fixed economic state sample, shared by every arm. Post-damage regimes are only
    economically visited at Y >= yhat, so Y is drawn there for the terminal regime."""
    r = np.random.default_rng(seed)
    return dict(
        logK=r.uniform(4.0, 7.0, (n, 1)),
        Z=r.uniform(0.01, 0.99, (n, 1)),
        Y=r.uniform(2.5, 4.0, (n, 1)) if post_damage else r.uniform(0.0, 2.5, (n, 1)),
        logR=r.uniform(1.0, 6.0, (n, 1)),
        lam3=r.choice([0.0, 1/12, 1/6, 1/4, 1/3], size=(n, 1)),
    )


def build_arm(arm, run_dir):
    """Instantiate the model under THIS ARM'S switches and restore its checkpoints."""
    for k in [k for k in os.environ if k.startswith("REDESIGN_")]:
        del os.environ[k]
    os.environ.update(ARM_SWITCHES.get(arm, {}))
    for m in ["config", "uncertainty", "state_layout", "hjb_scaling", "value_net",
              "params", "feedforward_subnet", "pretrained_paths", "PostDamagePostTech"]:
        sys.modules.pop(m, None)
    if VAR not in sys.path:
        sys.path.insert(0, VAR)
    import importlib
    import config; importlib.reload(config)
    from params import PARAMS, investment_rate_activation
    import PostDamagePostTech as M

    cfg = lambda n, a, f: {"num_hiddens": [32]*4, "use_bias": True, "activation": a,
                           "dim": 1, "nn_name": n, "final_activation": f}
    p = {"batch_size": 128, "learning_rates": [1e-5, 4e-4],
         "v_nn_config": cfg("v_nn", "swish", "softplus"),
         "i_g_nn_config": cfg("i_g_nn", "tanh", investment_rate_activation(PARAMS["θ_g"])),
         "i_d_nn_config": cfg("i_d_nn", "tanh", investment_rate_activation(PARAMS["θ_d"])),
         "i_r_nn_config": cfg("i_r_nn", "softplus", "softplus"),
         "num_iterations": 10, "logging_frequency": 100, "verbose": False,
         "pretrained_path": None, "learning_rate_schedule_type": "warmup_cosine",
         "tech_jump_intensity_scale": 1.0, "π": 1.0, "tensorboard": False, "export_folder": None}
    m = M.PostDamagePostTechModel(p)
    n_in = 7
    for net, nm in [(m.v_nn, "v_nn"), (m.i_g_nn, "i_g_nn"), (m.i_d_nn, "i_d_nn")]:
        net.build((None, n_in))
        ck = os.path.join(run_dir, SUB, f"{nm}_checkpoint_{SUB}")
        if glob.glob(ck + "*"):
            net.load_weights(ck).expect_partial()
    if getattr(m, "A_nn", None) is not None:
        ck = os.path.join(run_dir, SUB, "A_nn_checkpoint_" + SUB)
        if glob.glob(ck + "*"):
            m.A_nn.load_weights(ck).expect_partial()
    return m, config


def xi_column(cfgmod, xi, n):
    """Convert ONE economic xi into this arm's input convention."""
    if cfgmod.USE_THETA:
        return np.full((n, 1), 1.0 / xi)     # theta-mode arms carry theta = 1/xi
    return np.full((n, 1), np.log(xi))       # legacy arms carry log(xi)


def score(outroot, arms=None, seeds=(1, 2, 3)):
    import tensorflow as tf
    st = common_states()
    n = len(st["logK"])
    results = {}
    arms = arms or sorted({os.path.basename(d).rsplit("_seed", 1)[0]
                           for d in glob.glob(os.path.join(outroot, "*_seed*"))})
    for arm in arms:
        per_seed = []
        for s in seeds:
            run = os.path.join(outroot, f"{arm}_seed{s}")
            if not glob.glob(os.path.join(run, SUB, "v_nn_checkpoint_*")):
                continue
            m, cfgmod = build_arm(arm, run)
            row = {}
            for xi in XI_GRID:
                cols = [tf.constant(st["logK"], tf.float32), tf.constant(st["Z"], tf.float32),
                        tf.constant(st["Y"], tf.float32), tf.constant(st["logR"], tf.float32),
                        tf.constant(st["lam3"], tf.float32),
                        tf.constant(xi_column(cfgmod, xi, n), tf.float32)]
                out = m.objective_fn(*cols, compute_control=False, training=False)
                row[str(xi)] = {"hjb": float(out[0]), "foc_d": float(out[1]), "foc_g": float(out[2])}
            per_seed.append(row)
        if per_seed:
            results[arm] = per_seed
    return results


def report(results):
    print(f"COMMON YARDSTICK — identical {N_STATES}-state sample (seed {SEED}), "
          f"true UNSCALED HJB residual, per economic xi\n")
    hdr = f"{'arm':18}" + "".join(f"{('ξ='+str(x)):>11}" for x in XI_GRID) + f"{'all-ξ mean':>12}"
    print(hdr)
    for arm, seeds in results.items():
        vals = []
        for xi in XI_GRID:
            v = np.mean([s[str(xi)]["hjb"] for s in seeds])
            vals.append(v)
        print(f"{arm:18}" + "".join(f"{v:>11.3e}" for v in vals) + f"{np.mean(vals):>12.3e}")
    print("\n(deep-ξ and neutral columns read separately — a single all-ξ mean hides the trade-off)")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/output_redesign_20260726"
    res = score(root)
    report(res)
    out = os.path.join(root, "common_yardstick.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
