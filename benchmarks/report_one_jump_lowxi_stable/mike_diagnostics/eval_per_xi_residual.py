"""
Per-ξ HJB residual (= training loss_v) + FOC error for the ONE-JUMP (π=1) float64 run.

READ/EVAL ONLY. Reuses the PRODUCTION models_float64 residual computation verbatim by
importing the model classes, restoring THIS run's own checkpoints, and evaluating
objective_fn(..., training=False) on a fresh batch with logξ FIXED to each target ξ.

Regimes:
  - PreDamagePreTech   (initial; carries ALL jump terms: tech-breakthrough + damage)
  - PostDamagePostTech (terminal; 3-state, NO jumps -> clean control at all ξ)

Also patches the jump-exp to record whether the production clip_by_value(-1/ξ ΔV, -700, 350)
clamp fires at each ξ (the numerical guard that matters at low ξ).
"""
import os, sys, json
import numpy as np
import tensorflow as tf

# CRITICAL: match the production float64 sandbox BEFORE building any net.
tf.keras.backend.set_floatx("float64")

REPO   = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
MODELS = os.path.join(REPO, "models_float64")
BASE   = os.path.join(
    REPO, "output_lowxi_float64",
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostFull_logximin_m5p30_"
    "LR_warmup_cosine_10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations50000",
)
OUT = os.path.join(REPO, "benchmarks/report_one_jump_lowxi_stable/mike_diagnostics")
sys.path.insert(0, MODELS)

from params import PARAMS, investment_rate_activation          # noqa: E402
import PreDamagePreTech as PDPT_mod                             # noqa: E402
import PostDamagePostTech as PoDPoT_mod                         # noqa: E402

# This run's OWN exported checkpoints are float64 (saved by the float64 training run),
# so we restore them with plain float64 load_weights.  (load_weights_cast is only for
# warm-starting a float64 net from a *float32* legacy base -- not applicable here.)
def restore(net, ckpt_path, n_inputs):
    net.build((None, n_inputs))
    net.load_weights(ckpt_path).expect_partial()
    return net

N_INPUTS = 7
BATCH    = 4096
SEED     = 20260701

# ξ grid requested. logξ_max in this run = 5.0 (the neutral cap == "148.600").
XI_GRID = [("inf(logxi=5)", np.exp(5.0)), ("0.1", 0.1), ("0.05", 0.05), ("0.04", 0.04),
           ("0.03", 0.03), ("0.025", 0.025), ("0.02", 0.02), ("0.01", 0.01), ("0.005", 0.005)]

# --- clamp instrumentation ---------------------------------------------------
# We wrap tf.clip_by_value so that, during a residual eval, we can tell whether the
# production jump-exp clamp (lower=-700, upper=350) actually bit any sample point.
_CLAMP_FLAGS = {"lo": 0, "hi": 0, "n": 0}
_orig_clip = tf.clip_by_value
def _instrumented_clip(t, clip_value_min, clip_value_max, name=None):
    try:
        if float(clip_value_min) == -700.0 and float(clip_value_max) == 350.0:
            _CLAMP_FLAGS["lo"] += int(tf.reduce_sum(tf.cast(t < clip_value_min, tf.int64)).numpy())
            _CLAMP_FLAGS["hi"] += int(tf.reduce_sum(tf.cast(t > clip_value_max, tf.int64)).numpy())
            _CLAMP_FLAGS["n"]  += int(tf.size(t).numpy())
    except Exception:
        pass
    return _orig_clip(t, clip_value_min, clip_value_max, name=name)


def make_params(regime):
    lr_act_g = investment_rate_activation(PARAMS["θ_g"])
    lr_act_d = investment_rate_activation(PARAMS["θ_d"])
    v_cfg  = {"num_hiddens":[32]*4, "use_bias":True, "activation":"swish",   "dim":1, "nn_name":"v_nn",   "final_activation":"softplus"}
    ig_cfg = {"num_hiddens":[32]*4, "use_bias":True, "activation":"tanh",    "dim":1, "nn_name":"i_g_nn", "final_activation":lr_act_g}
    id_cfg = {"num_hiddens":[32]*4, "use_bias":True, "activation":"tanh",    "dim":1, "nn_name":"i_d_nn", "final_activation":lr_act_d}
    ir_cfg = {"num_hiddens":[32]*4, "use_bias":True, "activation":"softplus","dim":1, "nn_name":"i_r_nn", "final_activation":"softplus"}
    p = {"batch_size":128, "learning_rates":[1e-5,4e-4],
         "v_nn_config":v_cfg, "i_g_nn_config":ig_cfg, "i_d_nn_config":id_cfg, "i_r_nn_config":ir_cfg,
         "num_iterations":50000, "logging_frequency":100, "verbose":False,
         "pretrained_path":None, "learning_rate_schedule_type":"warmup_cosine",
         "tech_jump_intensity_scale":1.0, "π":1.0, "logξ_min":-5.3, "logξ_max":5.0,
         "tensorboard":False}
    # partner value-net paths (this run's own outputs); needed by PreDamagePreTech __init__
    p["v_PostDamagePostTech_nn_path"] = os.path.join(BASE, "PostDamagePostTech/v_nn_checkpoint_PostDamagePostTech")
    p["v_PreDamagePostTech_nn_path"]  = os.path.join(BASE, "PreDamagePostTech/v_nn_checkpoint_PreDamagePostTech")
    p["v_PostDamagePreTech_nn_path"]  = os.path.join(BASE, "PostDamagePreTech/v_nn_checkpoint_PostDamagePreTech")
    p["v_PreDamageIntermTech_nn_path"]= os.path.join(BASE, "PreDamageIntermTech/v_nn_checkpoint_PreDamageIntermTech")
    p["v_PostDamageIntermTech_nn_path"]=os.path.join(BASE, "PostDamageIntermTech/v_nn_checkpoint_PostDamageIntermTech")
    p["export_folder"] = None
    return p


def build_model(regime):
    """Instantiate production model; restore THIS run's own v/i checkpoints (float32->float64)."""
    p = make_params(regime)
    if regime == "PreDamagePreTech":
        # __init__ loads the partner value nets from *_nn_path (float64 load_weights).
        # Those partner ckpts are float32; models_float64 PreDamagePreTech.__init__ uses
        # plain load_weights, which needs a float32 backend for those partner nets.
        # -> load partners under float32, then restore main nets under float64.
        # Easiest faithful path: temporarily flip backend to float32 for __init__ partner
        # loads is NOT what production does; production builds partner nets in float64 and
        # load_weights casts?  It does NOT cast (that raised the very error load_weights_cast
        # fixes). So partner nets must also be cast. We therefore monkeypatch the model's
        # partner loads to use load_weights_cast.
        m = _build_predamage_pretech(p)
    else:
        m = PoDPoT_mod.PostDamagePostTechModel(p)
        restore(m.v_nn,   os.path.join(BASE, "PostDamagePostTech/v_nn_checkpoint_PostDamagePostTech"), N_INPUTS)
        restore(m.i_g_nn, os.path.join(BASE, "PostDamagePostTech/i_g_nn_checkpoint_PostDamagePostTech"), N_INPUTS)
        restore(m.i_d_nn, os.path.join(BASE, "PostDamagePostTech/i_d_nn_checkpoint_PostDamagePostTech"), N_INPUTS)
    return m


def _build_predamage_pretech(p):
    """Faithful PreDamagePreTech build: partner nets restored via load_weights_cast (float32->float64)."""
    from feedforward_subnet import FeedForwardSubNet
    from params import PARAMS as _P
    # Manually replicate __init__ WITHOUT its plain-load-weights partner restore,
    # so we can cast the float32 partner checkpoints into float64 nets.
    m = PDPT_mod.PreDamagePreTechModel.__new__(PDPT_mod.PreDamagePreTechModel)
    m.params = _P.copy(); m.params.update(p)
    from feedforward_subnet import setup_optimizers
    setup_optimizers(m.params)
    m.params["tensorboard"] = False
    m.v_nn   = FeedForwardSubNet(m.params["v_nn_config"])
    m.i_g_nn = FeedForwardSubNet(m.params["i_g_nn_config"])
    m.i_d_nn = FeedForwardSubNet(m.params["i_d_nn_config"])
    m.i_r_nn = FeedForwardSubNet(m.params["i_r_nn_config"])
    m.use_intermediate_tech_jump = float(m.params.get("π", 0.04)) < 1.0 - 1e-12  # -> False for π=1
    # partner value nets (this run's float64 exports): 6-input PreDamagePostTech, 8-input PostDamagePreTech
    m.v_PreDamagePostTech_nn = FeedForwardSubNet(m.params["v_nn_config"])
    restore(m.v_PreDamagePostTech_nn, m.params["v_PreDamagePostTech_nn_path"], 6)
    m.v_PreDamageIntermTech_nn = None
    m.v_PostDamagePreTech_nn = FeedForwardSubNet(m.params["v_nn_config"])
    restore(m.v_PostDamagePreTech_nn, m.params["v_PostDamagePreTech_nn_path"], 8)
    # state_intervals not needed for eval (we sample directly). Provide empty dict.
    m.params["state_intervals"] = {}
    # main nets (this run's PreDamagePreTech checkpoints)
    restore(m.v_nn,   os.path.join(BASE, "PreDamagePreTech/v_nn_checkpoint_PreDamagePreTech"), N_INPUTS)
    restore(m.i_g_nn, os.path.join(BASE, "PreDamagePreTech/i_g_nn_checkpoint_PreDamagePreTech"), N_INPUTS)
    restore(m.i_d_nn, os.path.join(BASE, "PreDamagePreTech/i_d_nn_checkpoint_PreDamagePreTech"), N_INPUTS)
    restore(m.i_r_nn, os.path.join(BASE, "PreDamagePreTech/i_r_nn_checkpoint_PreDamagePreTech"), N_INPUTS)
    return m


def sample_box(rng, n):
    """Uniform draw over the training box (§ ranges). Returns the six state columns (float64)."""
    logK = rng.uniform(4.0, 7.0, (n,1))
    Z    = rng.uniform(0.01, 0.99, (n,1))
    Y    = rng.uniform(0.0, 4.0, (n,1))
    logR = rng.uniform(1.0, 6.0, (n,1))
    l3   = rng.choice([0.,1/12,1/6,1/4,1/3], size=(n,1))
    return [tf.constant(a, dtype=tf.float64) for a in (logK, Z, Y, logR, l3)]


def load_econ_region(xi_label_float):
    """Load the deterministic ξ-path (K,Z,Y,R time series) to define an economically-relevant box."""
    # dir names use ξ_<3dp>; map neutral to 148.600
    sim = os.path.join(BASE, "SimulationDeterministic")
    avail = sorted(os.listdir(sim))
    # pick closest available ξ dir
    def parse(d):
        try: return float(d.split("ξ_")[1])
        except Exception: return None
    cands = [(parse(d), d) for d in avail if d.startswith("SimulationOutputs_ξ_")]
    cands = [(v,d) for v,d in cands if v is not None]
    if not cands: return None
    target = xi_label_float
    best = min(cands, key=lambda vd: abs(vd[0]-target))
    d = os.path.join(sim, best[1])
    try:
        K = np.loadtxt(os.path.join(d,"K.txt")); Z = np.loadtxt(os.path.join(d,"Z.txt"))
        Y = np.loadtxt(os.path.join(d,"Y.txt")); R = np.loadtxt(os.path.join(d,"R.txt"))
    except Exception:
        return None
    return best[0], dict(logK=np.log(K), Z=Z, Y=Y, logR=np.log(R))


def sample_econ(rng, n, region):
    """Sample within [min,max] of the deterministic path envelope (clipped to training box)."""
    def rng_col(arr, lo, hi):
        a, b = float(np.nanmin(arr)), float(np.nanmax(arr))
        a = max(a, lo); b = min(b, hi)
        if b <= a: b = a + 1e-3
        return rng.uniform(a, b, (n,1))
    logK = rng_col(region["logK"], 4.0, 7.0)
    Z    = rng_col(region["Z"],    0.01, 0.99)
    Y    = rng_col(region["Y"],    0.0, 4.0)
    logR = rng_col(region["logR"], 1.0, 6.0)
    l3   = rng.choice([0.,1/12,1/6,1/4,1/3], size=(n,1))
    return [tf.constant(a, dtype=tf.float64) for a in (logK, Z, Y, logR, l3)]


def eval_residual(model, regime, cols, xi):
    logK, Z, Y, logR, l3 = cols
    n = logK.shape[0]
    logxi = tf.constant(np.full((n,1), np.log(xi)), dtype=tf.float64)
    _CLAMP_FLAGS.update({"lo":0,"hi":0,"n":0})
    # objective_fn(training=False) returns (loss_v_rms, FOC_d_rms, FOC_g_rms, [FOC_r_rms,] loss_dv_dY_rms)
    out = model.objective_fn(logK, Z, Y, logR, l3, logxi, compute_control=False, training=False)
    out = [float(t.numpy()) for t in out]
    res = {"loss_v_rms": out[0], "FOC_d_rms": out[1], "FOC_g_rms": out[2]}
    if regime == "PreDamagePreTech":
        res["FOC_r_rms"] = out[3]; res["loss_dv_dY_rms"] = out[4]
        res["FOC_max"] = max(out[1], out[2], out[3])
    else:
        res["loss_dv_dY_rms"] = out[3]
        res["FOC_max"] = max(out[1], out[2])
    res["clamp_lo_frac"] = (_CLAMP_FLAGS["lo"]/_CLAMP_FLAGS["n"]) if _CLAMP_FLAGS["n"] else 0.0
    res["clamp_hi_frac"] = (_CLAMP_FLAGS["hi"]/_CLAMP_FLAGS["n"]) if _CLAMP_FLAGS["n"] else 0.0
    return res


def main():
    # instrument the clamp
    tf.clip_by_value = _instrumented_clip
    PDPT_mod.tf.clip_by_value = _instrumented_clip
    results = {"meta": {"BASE": BASE, "batch": BATCH, "seed": SEED, "n_inputs": N_INPUTS,
                        "note": "float64 run; jump-exp clamp = clip(-1/xi*dV, -700, 350); "
                                "logxi_max=5 == neutral 148.6; raw un-rescaled RMS residual."}}
    for regime in ["PreDamagePreTech", "PostDamagePostTech"]:
        print(f"\n===== {regime} =====", flush=True)
        model = build_model(regime)
        rng = np.random.default_rng(SEED)
        box_cols = sample_box(rng, BATCH)  # SAME batch across all ξ (isolates ξ effect)
        reg = results.setdefault(regime, {})
        for label, xi in XI_GRID:
            r_box = eval_residual(model, regime, box_cols, xi)
            entry = {"xi": xi, "logxi": float(np.log(xi)), "full_box": r_box}
            # economically-relevant region (per-ξ deterministic path envelope)
            region = load_econ_region(xi)
            if region is not None:
                rng2 = np.random.default_rng(SEED+7)
                econ_cols = sample_econ(rng2, BATCH, region[1])
                r_econ = eval_residual(model, regime, econ_cols, xi)
                entry["econ_region"] = {"sim_xi_used": region[0], **r_econ}
            reg[label] = entry
            print(f"  xi={xi:<10.4g} loss_v_rms(box)={r_box['loss_v_rms']:.4e} "
                  f"FOC_max={r_box['FOC_max']:.4e} clamp_lo={r_box['clamp_lo_frac']:.3f} "
                  f"clamp_hi={r_box['clamp_hi_frac']:.3f}", flush=True)
    with open(os.path.join(OUT, "per_xi_residual.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nWROTE", os.path.join(OUT, "per_xi_residual.json"), flush=True)


if __name__ == "__main__":
    main()
