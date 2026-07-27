"""Decisive: is the FD de-invest corner an FD boundary artifact or a real NN miss?
(1) Does FD agree with NN in the BULK (=> corner-specific) or differ by a constant (=> normalization)?
(2) Is the NN's HJB residual LOW at the FD de-invest corner points (=> NN locally HJB-consistent = right,
    FD boundary artifact) or HIGH (=> NN wrong there)?  NN global residual is 1.47e-3.
"""
import os, sys
import numpy as np
import tensorflow as tf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
from params import PARAMS, investment_rate_activation
import importlib
ORIG = importlib.import_module("PostDamagePostTech")
p = PARAMS
INC = "output_001/OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000/PostDamagePostTech"
LOGXI = np.log(148.4); LAM3 = 1.0/6.0

d = np.load("benchmarks/post_damage_post_tech/outputs/fd_pdpt_v5_stable_lam3_0167_xi148.npz")
gK, gZ, gY, v_fd = d["logK"], d["Z"], d["Y"], d["v"]
from scipy.interpolate import RegularGridInterpolator
fd_v = RegularGridInterpolator((gK, gZ, gY), v_fd, bounds_error=False, fill_value=None)

def build():
    nn, nl = 32, 4
    vc = {"num_hiddens": [nn]*nl, "use_bias": True, "activation": "swish", "dim": 1, "nn_name": "v_nn", "final_activation": "softplus"}
    ig = {"num_hiddens": [nn]*nl, "use_bias": True, "activation": "tanh", "dim": 1, "nn_name": "i_g_nn", "final_activation": investment_rate_activation(p["θ_g"])}
    idd = {"num_hiddens": [nn]*nl, "use_bias": True, "activation": "tanh", "dim": 1, "nn_name": "i_d_nn", "final_activation": investment_rate_activation(p["θ_d"])}
    params = dict(p); params.update({"batch_size": 128, "learning_rates": [1e-4, 1e-4], "v_nn_config": vc,
        "i_g_nn_config": ig, "i_d_nn_config": idd, "num_iterations": 1, "logging_frequency": 1, "verbose": False,
        "pretrained_path": None, "learning_rate_schedule_type": "warmup_cosine", "logξ_min": -3.0, "logξ_max": 5.0,
        "tensorboard": False, "export_folder": None})
    m = ORIG.PostDamagePostTechModel(params)
    for net in (m.v_nn, m.i_g_nn, m.i_d_nn): net.build((None, 7))
    m.v_nn.load_weights(os.path.join(INC, "v_nn_checkpoint_PostDamagePostTech"))
    m.i_g_nn.load_weights(os.path.join(INC, "i_g_nn_checkpoint_PostDamagePostTech"))
    m.i_d_nn.load_weights(os.path.join(INC, "i_d_nn_checkpoint_PostDamagePostTech"))
    return m

def nn_v_resid(m, lkv, zv, yv):
    lk = tf.Variable([[lkv]], dtype=tf.float32); z = tf.Variable([[zv]], dtype=tf.float32); y = tf.Variable([[yv]], dtype=tf.float32)
    l3 = tf.constant([[LAM3]], dtype=tf.float32); lx = tf.constant([[LOGXI]], dtype=tf.float32)
    out = m.pde_rhs(lk, z, y, tf.Variable([[3.5]], dtype=tf.float32), l3, lx)
    rhs, pv = out[0], out[1]
    X = tf.concat([lk, z, y, l3, tf.constant([[p['A_g_prime_prime']]], dtype=tf.float32), lx, lx], axis=1)
    return float(m.v_nn(X).numpy()), float((rhs - pv).numpy())

m = build()

print("=== (1) FD value vs NN value: BULK vs CORNER (logxi=5 slice) ===")
print(f"  {'logK':>5} {'Z':>5} {'Y':>5} | {'v_FD':>7} {'v_NN':>7} {'gap':>7} | {'NN |resid|':>10}")
pts = [(5.5,0.5,1.0),(5.5,0.7,2.0),(6.0,0.7,2.0),(5.0,0.5,0.5),     # bulk
       (6.5,0.9,3.5),(7.0,0.9,4.0),(7.0,0.98,4.0),(7.0,0.7,4.0),(6.5,0.9,4.0)]  # corner / de-invest
for (lkv,zv,yv) in pts:
    vN, res = nn_v_resid(m, lkv, zv, yv)
    vF = float(fd_v([[lkv,zv,yv]])[0])
    tag = "  <-- corner/de-invest" if (lkv>=6.5 and yv>=3.5) else ""
    print(f"  {lkv:5.2f} {zv:5.2f} {yv:5.2f} | {vF:7.3f} {vN:7.3f} {vF-vN:7.3f} | {abs(res):10.3e}{tag}")

print("\nNN global box residual reference = 1.47e-3.  If NN |resid| stays ~1e-3 at the corner,")
print("the NN is locally HJB-consistent there and the FD de-invest (all at logK=7 & Y=4 boundary) is the artifact.")
