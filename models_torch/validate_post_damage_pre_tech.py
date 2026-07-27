"""
VALIDATION: torch port of PostDamagePreTech vs the original TF model.

Loads the SAME trained TF checkpoints (v/i_g/i_d/i_r for PostDamagePreTech and
the frozen v for the PostDamagePostTech neighbour) into both the TF model and
the torch port, evaluates pde_rhs on the SAME random inputs over the regime
state box, and reports the max RELATIVE discrepancy per quantity.

Run:
    module load python/anaconda-2021.05
    cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal
    python models_torch/validate_post_damage_pre_tech.py
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")

RUN_DIR = os.path.join(
    ROOT,
    "output_001",
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000",
)
PRE_DIR = os.path.join(RUN_DIR, "PostDamagePreTech")
POST_DIR = os.path.join(RUN_DIR, "PostDamagePostTech")

INPUT_DIM = 8        # the PreTech networks take an 8-dim input
POST_INPUT_DIM = 7   # the frozen PostDamagePostTech neighbour takes a 7-dim input
N = 4096
SEED = 0
PI = 1.0             # the trained run config (folder name: Pi_1p0)


def make_configs():
    """Configs matching the trained run.

    v   : swish hidden / softplus final
    i_g : tanh  hidden / custom  (bounded investment-rate) final
    i_d : tanh  hidden / custom  final
    i_r : softplus hidden / softplus final  (i_r = exp(-i_r_nn(X)))
    """
    nh = [32, 32, 32, 32]
    v_cfg = {"num_hiddens": nh, "use_bias": True, "activation": "swish",
             "dim": 1, "nn_name": "v_nn", "final_activation": "softplus"}
    ig_cfg = {"num_hiddens": nh, "use_bias": True, "activation": "tanh",
              "dim": 1, "nn_name": "i_g_nn", "final_activation": "custom"}
    id_cfg = {"num_hiddens": nh, "use_bias": True, "activation": "tanh",
              "dim": 1, "nn_name": "i_d_nn", "final_activation": "custom"}
    ir_cfg = {"num_hiddens": nh, "use_bias": True, "activation": "softplus",
              "dim": 1, "nn_name": "i_r_nn", "final_activation": "softplus"}
    return v_cfg, ig_cfg, id_cfg, ir_cfg


def max_rel(a, b, eps=1e-8):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), eps)
    return float(np.max(np.abs(a - b) / denom))


def max_abs(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float(np.max(np.abs(a - b)))


def sample_state_box(params, n, seed):
    rng = np.random.default_rng(seed)

    def col(lo, hi):
        return rng.uniform(lo, hi, size=(n, 1)).astype(np.float32)

    logK = col(params["logK_min"], params["logK_max"])
    Z = col(params["Z_min"], params["Z_max"])
    Y = col(params["Y_min"], params["Y_max"])
    logR = col(params["logR_min"], params["logR_max"])
    λ3 = col(params["λ3_min"], params["λ3_max"])
    logξ = col(params["logξ_min"], params["logξ_max"])
    return logK, Z, Y, logR, λ3, logξ


def main():
    sys.path.insert(0, HERE)      # models_torch
    from params import PARAMS as TORCH_PARAMS
    from PostDamagePreTech import PostDamagePreTechModel as TorchModel
    from tf_torch_harness import load_tf_weights_into_torch

    import torch

    v_cfg, ig_cfg, id_cfg, ir_cfg = make_configs()

    logK, Z, Y, logR, λ3, logξ = sample_state_box(TORCH_PARAMS, N, SEED)

    # ============================================================
    #  TORCH side
    # ============================================================
    torch_model = TorchModel({
        "v_nn_config": v_cfg, "i_g_nn_config": ig_cfg,
        "i_d_nn_config": id_cfg, "i_r_nn_config": ir_cfg,
        "π": PI,
    })
    load_tf_weights_into_torch(torch_model.v_nn, os.path.join(PRE_DIR, "v_nn_checkpoint_PostDamagePreTech"))
    load_tf_weights_into_torch(torch_model.i_g_nn, os.path.join(PRE_DIR, "i_g_nn_checkpoint_PostDamagePreTech"))
    load_tf_weights_into_torch(torch_model.i_d_nn, os.path.join(PRE_DIR, "i_d_nn_checkpoint_PostDamagePreTech"))
    load_tf_weights_into_torch(torch_model.i_r_nn, os.path.join(PRE_DIR, "i_r_nn_checkpoint_PostDamagePreTech"))
    load_tf_weights_into_torch(torch_model.v_PostDamagePostTech_nn,
                               os.path.join(POST_DIR, "v_nn_checkpoint_PostDamagePostTech"))

    def t(a, rg=False):
        x = torch.from_numpy(a.astype(np.float32))
        x.requires_grad_(rg)
        return x

    tlogK = t(logK, True); tZ = t(Z, True); tY = t(Y, True)
    tlogR = t(logR, True); tλ3 = t(λ3, False); tlogξ = t(logξ, False)

    (rhs_t, pv_t, dvdY_t, c_t, gig_t, gid_t, FOC_d_t, FOC_g_t, FOC_r_t, dvdlogR_t) = \
        torch_model.pde_rhs(tlogK, tZ, tY, tlogR, tλ3, tlogξ)

    def nd(x):
        return x.detach().cpu().numpy()

    torch_out = {
        "rhs": nd(rhs_t), "pv": nd(pv_t), "dv_dY": nd(dvdY_t), "c": nd(c_t),
        "1+theta_g*i_g": nd(gig_t), "1+theta_d*i_d": nd(gid_t),
        "FOC_d": nd(FOC_d_t), "FOC_g": nd(FOC_g_t), "FOC_r": nd(FOC_r_t),
        "dv_dlogR": nd(dvdlogR_t),
    }

    # raw NN forwards on the assembled 8-dim X
    Xnp = np.concatenate([logK, Z, Y, logR, λ3, logξ, logξ, logξ], axis=1).astype(np.float32)
    with torch.no_grad():
        v_t = nd(torch_model.v_nn(torch.from_numpy(Xnp)))
        ig_t = nd(torch_model.i_g_nn(torch.from_numpy(Xnp)))
        id_t = nd(torch_model.i_d_nn(torch.from_numpy(Xnp)))
        ir_t = nd(torch.exp(-torch_model.i_r_nn(torch.from_numpy(Xnp))))

    # ============================================================
    #  TF side
    # ============================================================
    if MODELS not in sys.path:
        sys.path.insert(0, MODELS)
    import importlib
    for m in ["feedforward_subnet", "params", "pretrained_paths", "PostDamagePreTech"]:
        if m in sys.modules:
            del sys.modules[m]
    sys.path.remove(HERE)  # prefer models/ versions for the TF build
    import tensorflow as tf
    import PostDamagePreTech as TFmod

    tf_v_cfg = dict(v_cfg)
    tf_ig_cfg = dict(ig_cfg); tf_id_cfg = dict(id_cfg); tf_ir_cfg = dict(ir_cfg)
    from params import investment_rate_activation as tf_inv_act, PARAMS as TFP
    tf_ig_cfg["final_activation"] = tf_inv_act(TFP["θ_g"])
    tf_id_cfg["final_activation"] = tf_inv_act(TFP["θ_d"])

    tf_params = {
        "batch_size": 128,
        "learning_rates": [1e-4, 1e-4],
        "v_nn_config": tf_v_cfg, "i_g_nn_config": tf_ig_cfg,
        "i_d_nn_config": tf_id_cfg, "i_r_nn_config": tf_ir_cfg,
        "num_iterations": 10, "logging_frequency": 10,
        "pretrained_path": None, "learning_rate_schedule_type": "warmup_cosine",
        "tensorboard": False,
        "π": PI,
        "export_folder": os.path.join(HERE, "_tf_tmp_pretech"),
        # frozen-neighbour checkpoint paths used by __init__
        "v_PostDamagePostTech_nn_path": os.path.join(POST_DIR, "v_nn_checkpoint_PostDamagePostTech"),
        "v_PostDamageIntermTech_nn_path": os.path.join(RUN_DIR, "PostDamageIntermTech", "v_nn_checkpoint_PostDamageIntermTech"),
    }
    tf_model = TFmod.PostDamagePreTechModel(tf_params)
    tf_model.v_nn.build((None, INPUT_DIM)); tf_model.v_nn.load_weights(os.path.join(PRE_DIR, "v_nn_checkpoint_PostDamagePreTech")).expect_partial()
    tf_model.i_g_nn.build((None, INPUT_DIM)); tf_model.i_g_nn.load_weights(os.path.join(PRE_DIR, "i_g_nn_checkpoint_PostDamagePreTech")).expect_partial()
    tf_model.i_d_nn.build((None, INPUT_DIM)); tf_model.i_d_nn.load_weights(os.path.join(PRE_DIR, "i_d_nn_checkpoint_PostDamagePreTech")).expect_partial()
    tf_model.i_r_nn.build((None, INPUT_DIM)); tf_model.i_r_nn.load_weights(os.path.join(PRE_DIR, "i_r_nn_checkpoint_PostDamagePreTech")).expect_partial()

    # raw NN forwards
    v_tf = tf_model.v_nn(tf.convert_to_tensor(Xnp), training=False).numpy()
    ig_tf = tf_model.i_g_nn(tf.convert_to_tensor(Xnp), training=False).numpy()
    id_tf = tf_model.i_d_nn(tf.convert_to_tensor(Xnp), training=False).numpy()
    ir_tf = np.exp(-tf_model.i_r_nn(tf.convert_to_tensor(Xnp), training=False).numpy())

    flogK = tf.Variable(logK); fZ = tf.Variable(Z); fY = tf.Variable(Y)
    flogR = tf.Variable(logR); fλ3 = tf.Variable(λ3); flogξ = tf.Variable(logξ)
    (rhs_f, pv_f, dvdY_f, c_f, gig_f, gid_f, FOC_d_f, FOC_g_f, FOC_r_f, dvdlogR_f) = \
        tf_model.pde_rhs(flogK, fZ, fY, flogR, fλ3, flogξ)

    tf_out = {
        "rhs": rhs_f.numpy(), "pv": pv_f.numpy(), "dv_dY": dvdY_f.numpy(),
        "c": c_f.numpy(), "1+theta_g*i_g": gig_f.numpy(), "1+theta_d*i_d": gid_f.numpy(),
        "FOC_d": FOC_d_f.numpy(), "FOC_g": FOC_g_f.numpy(), "FOC_r": FOC_r_f.numpy(),
        "dv_dlogR": dvdlogR_f.numpy(),
    }

    # ============================================================
    #  REPORT
    # ============================================================
    print("\n==== NN forward (torch vs TF) max rel error ====")
    nn_errs = {
        "v_nn": max_rel(v_t, v_tf),
        "i_g_nn": max_rel(ig_t, ig_tf),
        "i_d_nn": max_rel(id_t, id_tf),
        "i_r": max_rel(ir_t, ir_tf),
    }
    for k, e in nn_errs.items():
        print(f"  {k:8s}: {e:.3e}")

    print("\n==== pde_rhs per-quantity (torch vs TF) ====")
    pde_errs = {}
    for k in torch_out:
        e = max_rel(torch_out[k], tf_out[k])
        ea = max_abs(torch_out[k], tf_out[k])
        pde_errs[k] = e
        print(f"  {k:16s}: max_rel={e:.3e}   max_abs={ea:.3e}   tf_maxabs={np.max(np.abs(tf_out[k])):.3e}")

    print("\nRESULT_JSON " + repr({"nn": nn_errs, "pde": pde_errs}))


if __name__ == "__main__":
    main()
