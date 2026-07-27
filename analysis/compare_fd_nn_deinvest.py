"""Where does the FD say de-invest (i_d<0), and is its driving qd=vlK-Z*vZ a real feature or a
boundary/numerical artifact?  Compare the FD grid vs the incumbent NN at the FD's de-invest points."""
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
LOGXI = np.log(148.4)  # FD slice
LAM3 = 1.0/6.0

d = np.load("benchmarks/post_damage_post_tech/outputs/fd_pdpt_v5_stable_lam3_0167_xi148.npz")
logK, Z, Y = d["logK"], d["Z"], d["Y"]
v_fd, id_fd, vlK_fd, vZ_fd = d["v"], d["i_d"], d["vlK"], d["vZ"]
qd_fd = vlK_fd - Z[None, :, None] * vZ_fd

def build_and_load():
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

def nn_at(m, lkv, zv, yv):
    lk = tf.Variable([[lkv]], dtype=tf.float32); z = tf.Variable([[zv]], dtype=tf.float32); y = tf.constant([[yv]], dtype=tf.float32)
    l3 = tf.constant([[LAM3]], dtype=tf.float32); lx = tf.constant([[LOGXI]], dtype=tf.float32); Ag = tf.constant([[p['A_g_prime_prime']]], dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tp:
        tp.watch([lk, z]); X = tf.concat([lk, z, y, l3, Ag, lx, lx], axis=1); v = m.v_nn(X)
    vlK = float(tp.gradient(v, lk).numpy()); vZ = float(tp.gradient(v, z).numpy()); del tp
    X = tf.concat([lk, z, y, l3, Ag, lx, lx], axis=1)
    return float(v.numpy()), vlK, vZ, vlK - zv*vZ, float(m.i_d_nn(X).numpy())

m = build_and_load()

# locate FD de-invest points
ii, jj, kk = np.where(id_fd < 0)
print(f"FD de-invest points: {len(ii)} of {id_fd.size} ({100*len(ii)/id_fd.size:.1f}%)")
if len(ii):
    print(f"  Z range of de-invest: [{Z[jj].min():.3f}, {Z[jj].max():.3f}]  (grid Z max={Z.max()})")
    print(f"  Y range of de-invest: [{Y[kk].min():.3f}, {Y[kk].max():.3f}]  (grid Y max={Y.max()})")
    print(f"  logK range: [{logK[ii].min():.3f}, {logK[ii].max():.3f}]  (grid logK: {logK.min()}-{logK.max()})")
    # are they all at the Z boundary?
    print(f"  fraction of de-invest at Z>=0.95: {np.mean(Z[jj]>=0.95):.2f};  at Y top-2 rows: {np.mean(kk>=len(Y)-2):.2f}")

print("\n=== FD vs NN at the most-negative FD de-invest points ===")
order = np.argsort(id_fd[ii, jj, kk])  # most negative first
print(f"  {'logK':>5} {'Z':>5} {'Y':>5} | {'id_FD':>8} {'qd_FD':>8} {'vZ_FD':>8} | {'id_NN':>8} {'qd_NN':>8} {'vZ_NN':>8} | {'v_FD':>7} {'v_NN':>7}")
for n in order[:12]:
    i, j, k = ii[n], jj[n], kk[n]
    vN, vlKN, vZN, qdN, idN = nn_at(m, logK[i], Z[j], Y[k])
    print(f"  {logK[i]:5.2f} {Z[j]:5.2f} {Y[k]:5.2f} | {id_fd[i,j,k]:8.4f} {qd_fd[i,j,k]:8.4f} {vZ_fd[i,j,k]:8.3f} | {idN:8.4f} {qdN:8.4f} {vZN:8.3f} | {v_fd[i,j,k]:7.3f} {vN:7.3f}")
