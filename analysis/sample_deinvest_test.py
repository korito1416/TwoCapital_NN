"""SIMPLE SAMPLING TEST: is the absence of de-invest a MODEL-SETUP fact, or an NN miss?

Load the incumbent (1M-iter) PostDamagePostTech under the ORIGINAL (clamp, TRUE) pde_rhs.
Scan Y in [0,4] at representative (Z, logxi).  For each state compute:
  * i_d_direct  = the i_d network output (what the NN actually plays)
  * qd          = V_logK - Z V_Z   (marginal value of dirty capital, autodiff of the value net)
  * thr         = delta / (Gamma_d theta_d (C/K))   (FOC de-invest threshold; i_d<0 iff qd<thr)
  * i_d_FOC     = (Gamma_d theta_d (C/K) qd / delta - 1)/theta_d   (i_d implied by the NN's OWN qd)

Reading:
  - if BOTH i_d_direct and i_d_FOC stay POSITIVE for all Y  -> the model setup itself has NO de-invest
    (the NN is right; the FD's mild de-invest is the artifact).
  - if i_d_FOC goes NEGATIVE at high Y but i_d_direct stays positive -> de-invest IS economically present
    in the model (via qd) but the i_d network FAILS to capture it (the real v_Z/control problem).
"""
import os, sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
from params import PARAMS, investment_rate_activation
import importlib
ORIG = importlib.import_module("PostDamagePostTech")

INC = "output_001/OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000/PostDamagePostTech"

p = PARAMS
A_d, A_gpp = p['A_d'], p['A_g_prime_prime']
δ, Γ_d, θ_d = p['δ'], p['Γ_d'], p['θ_d']


def build_and_load():
    nn, nl = 32, 4
    v = {"num_hiddens": [nn]*nl, "use_bias": True, "activation": "swish", "dim": 1, "nn_name": "v_nn", "final_activation": "softplus"}
    ig = {"num_hiddens": [nn]*nl, "use_bias": True, "activation": "tanh", "dim": 1, "nn_name": "i_g_nn", "final_activation": investment_rate_activation(p["θ_g"])}
    idd = {"num_hiddens": [nn]*nl, "use_bias": True, "activation": "tanh", "dim": 1, "nn_name": "i_d_nn", "final_activation": investment_rate_activation(p["θ_d"])}
    params = dict(p); params.update({"batch_size": 128, "learning_rates": [1e-4, 1e-4], "v_nn_config": v,
        "i_g_nn_config": ig, "i_d_nn_config": idd, "num_iterations": 1, "logging_frequency": 1, "verbose": False,
        "pretrained_path": None, "learning_rate_schedule_type": "warmup_cosine", "logξ_min": -3.0, "logξ_max": 5.0,
        "tensorboard": False, "export_folder": None})
    m = ORIG.PostDamagePostTechModel(params)
    for net in (m.v_nn, m.i_g_nn, m.i_d_nn):
        net.build((None, 7))
    m.v_nn.load_weights(os.path.join(INC, "v_nn_checkpoint_PostDamagePostTech"))
    m.i_g_nn.load_weights(os.path.join(INC, "i_g_nn_checkpoint_PostDamagePostTech"))
    m.i_d_nn.load_weights(os.path.join(INC, "i_d_nn_checkpoint_PostDamagePostTech"))
    return m


def scan(m, Z, logxi, logK=5.5, lam3=1.0/6.0, nY=21):
    Ys = np.linspace(0.0, 4.0, nY)
    rows = []
    for Yv in Ys:
        lk = tf.Variable([[logK]], dtype=tf.float32)
        z  = tf.Variable([[Z]],    dtype=tf.float32)
        y  = tf.Variable([[Yv]],   dtype=tf.float32)
        l3 = tf.constant([[lam3]], dtype=tf.float32)
        lx = tf.constant([[logxi]], dtype=tf.float32)
        Ag = tf.constant([[A_gpp]], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tp:
            tp.watch([lk, z])
            X = tf.concat([lk, z, y, l3, Ag, lx, lx], axis=1)
            v = m.v_nn(X)
        dv_dlogK = tp.gradient(v, lk)
        dv_dZ = tp.gradient(v, z)
        del tp
        qd = float(dv_dlogK.numpy()) - Z * float(dv_dZ.numpy())
        X = tf.concat([lk, z, y, l3, Ag, lx, lx], axis=1)
        i_d = float(m.i_d_nn(X).numpy())
        i_g = float(m.i_g_nn(X).numpy())
        CK = (A_d - i_d) * (1 - Z) + (A_gpp - i_g) * Z   # C/K
        thr = δ / (Γ_d * θ_d * max(CK, 1e-6))
        i_d_FOC = (Γ_d * θ_d * max(CK, 1e-6) * qd / δ - 1.0) / θ_d
        rows.append((Yv, i_d, i_d_FOC, qd, thr, CK))
    return rows


m = build_and_load()
print(f"params: delta={δ}, Gamma_d={Γ_d}, theta_d={θ_d}, -1/theta_d (de-invest floor)={-1/θ_d:.4f}")
print(f"A_d={A_d}, A_g''={A_gpp}")
for logxi, tag in [(5.0, "logxi=5 (~neutral, FD slice)"), (-3.0, "logxi=-3 (averse)")]:
    for Z in [0.7, 0.9]:
        print(f"\n=== Z={Z}, {tag}, logK=5.5, lam3=1/6 ===")
        print(f"   {'Y':>5} {'i_d(NN)':>10} {'i_d(FOC)':>10} {'qd':>9} {'thr':>9} {'C/K':>8}   deinvest?")
        for (Yv, i_d, i_d_FOC, qd, thr, CK) in scan(m, Z, logxi):
            flag = "<-- i_d_FOC<0" if i_d_FOC < 0 else ("<- NN<0" if i_d < 0 else "")
            print(f"   {Yv:5.2f} {i_d:10.5f} {i_d_FOC:10.5f} {qd:9.4f} {thr:9.4f} {CK:8.4f}   {flag}")
