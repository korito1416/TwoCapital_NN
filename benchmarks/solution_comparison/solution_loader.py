"""Shared loader for comparing trained solutions across runs and jump states.

Consolidates the evaluation harness used for the July-2026 two-trainings reports
(solution_identification / solution_uniqueness). Every eval/plot script in this
directory imports from here instead of re-implementing the checkpoint loading.

Conventions (see README.md):
- Checkpoints may be float32 or float64; weights are always cast to float32 and
  evaluated with the float32 `models/` code path.
- Post-damage jump states are entered at Y = y_upper = 2.5 and live at Y >= 2.5;
  evaluate them at the entry slice (Y := 2.5), never at pre-damage path Y.
- lambda3 pseudo-state fixed at 1/6 (grid midpoint) unless a script says otherwise.
"""
import os, sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "models"))

import tensorflow as tf  # noqa: E402
tf.keras.backend.set_floatx("float32")
from feedforward_subnet import FeedForwardSubNet            # noqa: E402
from PreDamagePreTech import PreDamagePreTechModel          # noqa: E402
from PreDamagePostTech import PreDamagePostTechModel        # noqa: E402
from PostDamagePreTech import PostDamagePreTechModel        # noqa: E402
from PostDamagePostTech import PostDamagePostTechModel      # noqa: E402
from params import PARAMS, investment_rate_activation       # noqa: E402

# ---------------------------------------------------------------- constants
REGS = ["PreDamagePreTech", "PreDamagePostTech", "PostDamagePreTech", "PostDamagePostTech"]
DIMS = {"PreDamagePreTech": 7, "PreDamagePostTech": 6, "PostDamagePreTech": 8, "PostDamagePostTech": 7}
HASR = {"PreDamagePreTech": True, "PreDamagePostTech": False, "PostDamagePreTech": True, "PostDamagePostTech": False}
NETS = {"PreDamagePreTech": ["v", "i_g", "i_d", "i_r"], "PreDamagePostTech": ["v", "i_g", "i_d"],
        "PostDamagePreTech": ["v", "i_g", "i_d", "i_r"], "PostDamagePostTech": ["v", "i_g", "i_d"]}
RLAB = {"PreDamagePreTech": "pre-dmg / pre-tech", "PreDamagePostTech": "pre-dmg / post-tech",
        "PostDamagePreTech": "post-dmg / pre-tech", "PostDamagePostTech": "post-dmg / post-tech"}
CLS = {"PreDamagePreTech": PreDamagePreTechModel, "PreDamagePostTech": PreDamagePostTechModel,
       "PostDamagePreTech": PostDamagePreTechModel, "PostDamagePostTech": PostDamagePostTechModel}
Y_ENTRY = 2.5          # post-damage entry temperature (y_upper; code: PreDamagePreTech.py L282)
L3_DEFAULT = 1.0 / 6.0
AGPP = 0.1567          # breakthrough green productivity (post-tech pseudo-input)

# reference float32 run used only to instantiate downstream nets before overwrite
DEFAULT_F32_RUN = os.path.join(
    REPO_ROOT,
    "output_001/OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_"
    "10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000",
)

def _cfg(act, fin, nm):
    return {"num_hiddens": [32] * 4, "use_bias": True, "activation": act,
            "dim": 1, "nn_name": nm, "final_activation": fin}

CFG = {"v": _cfg("swish", "softplus", "v_nn"),
       "i_g": _cfg("tanh", investment_rate_activation(PARAMS["θ_g"]), "i_g_nn"),
       "i_d": _cfg("tanh", investment_rate_activation(PARAMS["θ_d"]), "i_d_nn"),
       "i_r": _cfg("softplus", "softplus", "i_r_nn")}

# ---------------------------------------------------------------- loading
def cast_w(ckpt, cfg, dim):
    """Load a checkpoint that may be float32 or float64; return float32 weights."""
    for fx in ["float32", "float64"]:
        try:
            tf.keras.backend.set_floatx(fx)
            n = FeedForwardSubNet(cfg)
            n(tf.zeros([1, dim], dtype=fx))
            n.load_weights(ckpt).expect_partial()
            w = [np.asarray(x, dtype=np.float32) for x in n.get_weights()]
            tf.keras.backend.set_floatx("float32")
            return w
        except Exception:
            tf.keras.backend.set_floatx("float32")
            continue
    raise RuntimeError("could not load (f32/f64): " + ckpt)

def _base_params():
    p = PARAMS.copy()
    p["π"] = 1.0; p["tensorboard"] = False; p["batch_size"] = 512
    p["learning_rates"] = [1e-5, 4e-4]; p["learning_rate_schedule_type"] = "warmup_cosine"
    p["num_iterations"] = 1; p["gradient_clip_norm"] = 1.0; p["export_folder"] = None
    for k in CFG:
        p[f"{k}_nn_config"] = CFG[k]
    return p

def make_model(reg, root, f32_instantiation_run=DEFAULT_F32_RUN):
    """Full regime model (pde_rhs / objective_fn usable) with `root`'s weights.

    Downstream value nets are first instantiated from a float32 run (constructor
    loads them), then overwritten with `root`'s own downstream weights.
    """
    tf.keras.backend.set_floatx("float32")
    p = _base_params()
    if reg == "PreDamagePreTech":
        p["v_PreDamagePostTech_nn_path"] = f"{f32_instantiation_run}/PreDamagePostTech/v_nn_checkpoint_PreDamagePostTech"
        p["v_PostDamagePreTech_nn_path"] = f"{f32_instantiation_run}/PostDamagePreTech/v_nn_checkpoint_PostDamagePreTech"
    elif reg in ("PreDamagePostTech", "PostDamagePreTech"):
        p["v_PostDamagePostTech_nn_path"] = f"{f32_instantiation_run}/PostDamagePostTech/v_nn_checkpoint_PostDamagePostTech"
    m = CLS[reg](p)
    for nm in NETS[reg]:
        nn = getattr(m, f"{nm}_nn")
        nn(tf.zeros([1, DIMS[reg]]))
        nn.set_weights(cast_w(f"{root}/{reg}/{nm}_nn_checkpoint_{reg}", CFG[nm], DIMS[reg]))
    if reg == "PreDamagePreTech":
        m.v_PreDamagePostTech_nn.set_weights(cast_w(f"{root}/PreDamagePostTech/v_nn_checkpoint_PreDamagePostTech", CFG["v"], 6))
        m.v_PostDamagePreTech_nn.set_weights(cast_w(f"{root}/PostDamagePreTech/v_nn_checkpoint_PostDamagePreTech", CFG["v"], 8))
    elif reg in ("PreDamagePostTech", "PostDamagePreTech"):
        m.v_PostDamagePostTech_nn.set_weights(cast_w(f"{root}/PostDamagePostTech/v_nn_checkpoint_PostDamagePostTech", CFG["v"], 7))
    return m

def load_v(root, reg):
    """Value network only (for value/marginal-value comparisons)."""
    ck = f"{root}/{reg}/v_nn_checkpoint_{reg}"
    w = cast_w(ck, CFG["v"], DIMS[reg])
    m = FeedForwardSubNet(CFG["v"])
    m(tf.zeros([1, DIMS[reg]]))
    m.set_weights(w)
    return m

def build_X(reg, lk, Z, Y, lr, lx):
    """v-net input layout per jump state (matches each models/<Regime>.py concat)."""
    ones = tf.ones_like(Y)
    if reg == "PreDamagePreTech":
        return tf.concat([lk, Z, Y, lr, lx, lx, lx], 1)
    if reg == "PreDamagePostTech":
        return tf.concat([lk, Z, Y, AGPP * ones, lx, lx], 1)
    if reg == "PostDamagePreTech":
        return tf.concat([lk, Z, Y, lr, L3_DEFAULT * ones, lx, lx, lx], 1)
    if reg == "PostDamagePostTech":
        return tf.concat([lk, Z, Y, L3_DEFAULT * ones, AGPP * ones, lx, lx], 1)
    raise ValueError(reg)

# ---------------------------------------------------------------- states
def path_states(sim_run_root, xi_dir, stride=2):
    """States along a simulated 60-year path: SimulationOutputs_ξ_<xi_dir> under sim_run_root."""
    d = os.path.join(sim_run_root, "SimulationDeterministic", f"SimulationOutputs_ξ_{xi_dir}")
    K = np.loadtxt(f"{d}/K.txt")[::stride]; Z = np.loadtxt(f"{d}/Z.txt")[::stride]
    Y = np.loadtxt(f"{d}/Y.txt")[::stride]; R = np.loadtxt(f"{d}/R.txt")[::stride]
    n = min(len(K), len(Z), len(Y), len(R))
    f = lambda a: a[:n].reshape(-1, 1).astype(np.float32)
    return np.log(f(K)), f(Z), f(Y), np.log(f(R))

def lhs(n, bounds, seed=0):
    """Stratified (Latin-hypercube style) sample; bounds = [(lo,hi), ...]."""
    rng = np.random.RandomState(seed)
    return [((lo + (hi - lo) * (rng.permutation(n) + rng.rand(n)) / n)
             .reshape(-1, 1).astype(np.float32)) for lo, hi in bounds]

# ---------------------------------------------------------------- evaluation
def eval_terms(model, reg, lk, Z, Y, lr, xi, l3=L3_DEFAULT):
    """RMS of every objective term at the given states (objective_fn training=False).

    Returns dict(res, FOC_d, FOC_g, FOC_r) — FOC_r None for post-tech states.
    NaN on numerical failure (e.g. float32 overflow of the jump term at deep xi).
    """
    n = len(lk)
    T = [tf.constant(a) for a in (lk, Z, Y, lr, np.full_like(lk, l3))]
    lx = tf.constant(np.full((n, 1), np.log(xi), dtype=np.float32))
    try:
        r = model.objective_fn(T[0], T[1], T[2], T[3], T[4], lx, training=False)
        vals = [float(x) for x in r]
    except Exception:
        vals = [np.nan] * (5 if HASR[reg] else 4)
    if HASR[reg]:
        res, fd, fg, fr, _ = vals
    else:
        (res, fd, fg, _), fr = vals, None
    return dict(res=res, FOC_d=fd, FOC_g=fg, FOC_r=fr)

def parse_runs(pairs):
    """CLI helper: ['label=/abs/run/root', ...] -> ordered dict-like list."""
    out = []
    for p in pairs:
        lab, root = p.split("=", 1)
        out.append((lab, os.path.abspath(root)))
    return out
