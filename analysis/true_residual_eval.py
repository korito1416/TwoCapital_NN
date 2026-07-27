"""DECISIVE check: is the hardened 'win' an artifact of the C>0 softplus distorting consumption?

Evaluate the SAME weights (incumbent vs base-arm) under BOTH pde_rhs:
  * ORIGINAL  models/PostDamagePostTech.py  -> inside_log = max(c, 1e-8)  (TRUE consumption)
  * HARDENED  PostDamagePostTech_hardened   -> c_pos = c_floor + softplus(c - c_floor)  (distorts c~0.1 -> 0.74)

RMS(rhs - pv) on the SAME box-uniform 20k sample.  A 2x2 table:
                       original/clamp(TRUE)   hardened/softplus
   incumbent weights        ?                      2.73e-2 (grader)
   base-arm weights         ?                       7.1e-3 (grader)

If incumbent-under-clamp ~ 1.3e-3 and base-under-clamp >> that, the 'win' is a pure softplus artifact
and the base arm is actually solving a DISTORTED HJB.
"""
import os, sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

from params import PARAMS, investment_rate_activation
import importlib

ORIG = importlib.import_module("PostDamagePostTech")          # models/ on path -> original (clamp)
H    = importlib.import_module("PostDamagePostTech_hardened")  # hardened (softplus)

INC = "output_001/OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000/PostDamagePostTech"
BASE = "output_hardened_scratch/abl_base_c0/PostDamagePostTech/best"


def cfgs():
    nn, nl = 32, 4
    v = {"num_hiddens": [nn]*nl, "use_bias": True, "activation": "swish", "dim": 1, "nn_name": "v_nn", "final_activation": "softplus"}
    ig = {"num_hiddens": [nn]*nl, "use_bias": True, "activation": "tanh", "dim": 1, "nn_name": "i_g_nn", "final_activation": investment_rate_activation(PARAMS["θ_g"])}
    idd = {"num_hiddens": [nn]*nl, "use_bias": True, "activation": "tanh", "dim": 1, "nn_name": "i_d_nn", "final_activation": investment_rate_activation(PARAMS["θ_d"])}
    return v, ig, idd


def build(modmod):
    v, ig, idd = cfgs()
    params = dict(PARAMS)
    params.update({"batch_size": 128, "learning_rates": [1e-4, 1e-4], "v_nn_config": v,
                   "i_g_nn_config": ig, "i_d_nn_config": idd, "num_iterations": 1,
                   "logging_frequency": 1, "verbose": False, "pretrained_path": None,
                   "learning_rate_schedule_type": "warmup_cosine", "logξ_min": -3.0, "logξ_max": 5.0,
                   "costate_supervision_weight": 0.0, "relative_residual": False,
                   "xi_curriculum": False, "tensorboard": False, "export_folder": None})
    m = modmod.PostDamagePostTechModel(params)
    for net in (m.v_nn, m.i_g_nn, m.i_d_nn):
        net.build((None, 7))
    return m


def load(m, ckpt):
    m.v_nn.load_weights(os.path.join(ckpt, "v_nn_checkpoint_PostDamagePostTech"))
    m.i_g_nn.load_weights(os.path.join(ckpt, "i_g_nn_checkpoint_PostDamagePostTech"))
    m.i_d_nn.load_weights(os.path.join(ckpt, "i_d_nn_checkpoint_PostDamagePostTech"))


def col(a): return tf.constant(a.reshape(-1, 1), dtype=tf.float32)

def sample(seed=0, n=20000):
    r = np.random.default_rng(seed)
    return (col(r.uniform(4.0, 7.0, n)), col(r.uniform(0.01, 0.99, n)), col(r.uniform(0.0, 4.0, n)),
            col(r.uniform(1.0, 6.0, n)), col(r.uniform(0.0, 1/3, n)), col(r.uniform(-3.0, 5.0, n)))

def rms_resid(m, s):
    lk, z, y, lr, l3, lx = s
    lk = tf.Variable(lk); z = tf.Variable(z); y = tf.Variable(y); lr = tf.Variable(lr)
    out = m.pde_rhs(lk, z, y, lr, l3, lx)
    rhs, pv = out[0], out[1]
    return float(np.sqrt(np.mean((rhs.numpy() - pv.numpy())**2)))

def mean_c(m, s):
    lk, z, y, lr, l3, lx = s
    lk = tf.Variable(lk); z = tf.Variable(z); y = tf.Variable(y); lr = tf.Variable(lr)
    out = m.pde_rhs(lk, z, y, lr, l3, lx)
    c = out[3]
    return float(np.mean(c.numpy())), float(np.min(c.numpy()))


s = sample(0, 20000)
m_orig = build(ORIG)
m_hard = build(H)

print("=== softplus distortion sanity: c_pos vs c ===")
for c in [0.03, 0.05, 0.10, 0.13]:
    cp = 1e-3 + np.log1p(np.exp(c - 1e-3))
    print(f"  c={c:.3f} -> c_pos={cp:.4f} ({cp/c:.2f}x)")

print("\n=== 2x2: RMS(rhs-pv) on the SAME uniform-box 20k ===")
for name, ckpt in [("incumbent", INC), ("base_arm", BASE)]:
    load(m_orig, ckpt); load(m_hard, ckpt)
    r_true = rms_resid(m_orig, s)
    r_hard = rms_resid(m_hard, s)
    cmean, cmin = mean_c(m_orig, s)
    print(f"  {name:10s}:  original/clamp(TRUE)={r_true:.4e}   hardened/softplus={r_hard:.4e}   (mean c={cmean:.4f}, min c={cmin:.4f})")
