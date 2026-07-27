"""
VALIDATION: torch port of PostDamagePostTech vs the original TF model.

Loads the SAME trained TF checkpoint into both the TF model and the torch port,
evaluates the NN forward and pde_rhs on the SAME random inputs over the regime
state box, and reports the max RELATIVE discrepancy per quantity.

Run:
    module load python/anaconda-2021.05
    cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal
    python models_torch/validate_post_damage_post_tech.py
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")

CKPT_DIR = os.path.join(
    ROOT,
    "output_001",
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000",
    "PostDamagePostTech",
)

INPUT_DIM = 7
N = 4096
SEED = 0


def make_configs():
    """Configs matching the trained run (v: swish/softplus, i_*: tanh/custom)."""
    nh = [32, 32, 32, 32]
    v_cfg = {"num_hiddens": nh, "use_bias": True, "activation": "swish",
             "dim": 1, "nn_name": "v_nn", "final_activation": "softplus"}
    ig_cfg = {"num_hiddens": nh, "use_bias": True, "activation": "tanh",
              "dim": 1, "nn_name": "i_g_nn", "final_activation": "custom"}
    id_cfg = {"num_hiddens": nh, "use_bias": True, "activation": "tanh",
              "dim": 1, "nn_name": "i_d_nn", "final_activation": "custom"}
    return v_cfg, ig_cfg, id_cfg


def max_rel(a, b, eps=1e-8):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), eps)
    return float(np.max(np.abs(a - b) / denom))


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
    from feedforward_subnet import FeedForwardSubNet as TorchNet
    from PostDamagePostTech import PostDamagePostTechModel as TorchModel
    from tf_torch_harness import load_tf_weights_into_torch

    import torch

    v_cfg, ig_cfg, id_cfg = make_configs()

    # ---------------- random inputs over the state box ----------------
    logK, Z, Y, logR, λ3, logξ = sample_state_box(TORCH_PARAMS, N, SEED)

    # ============================================================
    #  TORCH side
    # ============================================================
    torch_model = TorchModel({
        "v_nn_config": v_cfg, "i_g_nn_config": ig_cfg, "i_d_nn_config": id_cfg,
    })
    load_tf_weights_into_torch(torch_model.v_nn, os.path.join(CKPT_DIR, "v_nn_checkpoint_PostDamagePostTech"))
    load_tf_weights_into_torch(torch_model.i_g_nn, os.path.join(CKPT_DIR, "i_g_nn_checkpoint_PostDamagePostTech"))
    load_tf_weights_into_torch(torch_model.i_d_nn, os.path.join(CKPT_DIR, "i_d_nn_checkpoint_PostDamagePostTech"))

    def t(a, rg=False):
        x = torch.from_numpy(a.astype(np.float32))
        x.requires_grad_(rg)
        return x

    tlogK = t(logK, True); tZ = t(Z, True); tY = t(Y, True)
    tlogR = t(logR, False); tλ3 = t(λ3, False); tlogξ = t(logξ, False)

    (rhs_t, pv_t, dvdY_t, c_t, gig_t, gid_t, FOC_d_t, FOC_g_t) = \
        torch_model.pde_rhs(tlogK, tZ, tY, tlogR, tλ3, tlogξ)

    def nd(x):
        return x.detach().cpu().numpy()

    torch_out = {
        "rhs": nd(rhs_t), "pv": nd(pv_t), "dv_dY": nd(dvdY_t), "c": nd(c_t),
        "1+theta_g*i_g": nd(gig_t), "1+theta_d*i_d": nd(gid_t),
        "FOC_d": nd(FOC_d_t), "FOC_g": nd(FOC_g_t),
    }

    # also raw NN forwards on the assembled X
    Xnp = np.concatenate(
        [logK, Z, Y, λ3, TORCH_PARAMS["A_g_prime_prime"] * np.ones_like(Y), logξ, logξ],
        axis=1,
    ).astype(np.float32)
    with torch.no_grad():
        v_t = nd(torch_model.v_nn(torch.from_numpy(Xnp)))
        ig_t = nd(torch_model.i_g_nn(torch.from_numpy(Xnp)))
        id_t = nd(torch_model.i_d_nn(torch.from_numpy(Xnp)))

    # ============================================================
    #  TF side
    # ============================================================
    if MODELS not in sys.path:
        sys.path.insert(0, MODELS)
    # Ensure TF model picks up the right modules from models/
    import importlib
    for m in ["feedforward_subnet", "params", "pretrained_paths", "PostDamagePostTech"]:
        if m in sys.modules:
            del sys.modules[m]
    sys.path.remove(HERE)  # prefer models/ versions for the TF build
    import tensorflow as tf
    import PostDamagePostTech as TFmod

    tf_v_cfg = dict(v_cfg)
    tf_ig_cfg = dict(ig_cfg); tf_id_cfg = dict(id_cfg)
    # The TF model swaps "custom" -> investment_rate_activation in __main__; do it here.
    from params import investment_rate_activation as tf_inv_act, PARAMS as TFP
    tf_ig_cfg["final_activation"] = tf_inv_act(TFP["θ_g"])
    tf_id_cfg["final_activation"] = tf_inv_act(TFP["θ_d"])

    tf_params = {
        "batch_size": 128,
        "learning_rates": [1e-4, 1e-4],
        "v_nn_config": tf_v_cfg, "i_g_nn_config": tf_ig_cfg, "i_d_nn_config": tf_id_cfg,
        "num_iterations": 10, "logging_frequency": 10,
        "pretrained_path": None, "learning_rate_schedule_type": "warmup_cosine",
        "tensorboard": False,
        "export_folder": os.path.join(HERE, "_tf_tmp"),
    }
    tf_model = TFmod.PostDamagePostTechModel(tf_params)
    tf_model.v_nn.build((None, INPUT_DIM)); tf_model.v_nn.load_weights(os.path.join(CKPT_DIR, "v_nn_checkpoint_PostDamagePostTech")).expect_partial()
    tf_model.i_g_nn.build((None, INPUT_DIM)); tf_model.i_g_nn.load_weights(os.path.join(CKPT_DIR, "i_g_nn_checkpoint_PostDamagePostTech")).expect_partial()
    tf_model.i_d_nn.build((None, INPUT_DIM)); tf_model.i_d_nn.load_weights(os.path.join(CKPT_DIR, "i_d_nn_checkpoint_PostDamagePostTech")).expect_partial()

    # raw NN forwards
    v_tf = tf_model.v_nn(tf.convert_to_tensor(Xnp), training=False).numpy()
    ig_tf = tf_model.i_g_nn(tf.convert_to_tensor(Xnp), training=False).numpy()
    id_tf = tf_model.i_d_nn(tf.convert_to_tensor(Xnp), training=False).numpy()

    # pde_rhs needs differentiable inputs
    flogK = tf.Variable(logK); fZ = tf.Variable(Z); fY = tf.Variable(Y)
    flogR = tf.Variable(logR); fλ3 = tf.Variable(λ3); flogξ = tf.Variable(logξ)
    (rhs_f, pv_f, dvdY_f, c_f, gig_f, gid_f, FOC_d_f, FOC_g_f) = \
        tf_model.pde_rhs(flogK, fZ, fY, flogR, fλ3, flogξ)

    tf_out = {
        "rhs": rhs_f.numpy(), "pv": pv_f.numpy(), "dv_dY": dvdY_f.numpy(),
        "c": c_f.numpy(), "1+theta_g*i_g": gig_f.numpy(), "1+theta_d*i_d": gid_f.numpy(),
        "FOC_d": FOC_d_f.numpy(), "FOC_g": FOC_g_f.numpy(),
    }

    # ============================================================
    #  REPORT
    # ============================================================
    print("\n==== NN forward (torch vs TF) max rel error ====")
    nn_errs = {
        "v_nn": max_rel(v_t, v_tf),
        "i_g_nn": max_rel(ig_t, ig_tf),
        "i_d_nn": max_rel(id_t, id_tf),
    }
    for k, e in nn_errs.items():
        print(f"  {k:8s}: {e:.3e}")

    print("\n==== pde_rhs per-quantity (torch vs TF) max rel error ====")
    pde_errs = {}
    for k in torch_out:
        e = max_rel(torch_out[k], tf_out[k])
        pde_errs[k] = e
        print(f"  {k:16s}: {e:.3e}")

    print("\nRESULT_JSON " + repr({"nn": nn_errs, "pde": pde_errs}))


if __name__ == "__main__":
    main()
