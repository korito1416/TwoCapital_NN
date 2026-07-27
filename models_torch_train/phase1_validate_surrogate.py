"""
PHASE 1 validation: confirm the TF -> torch surrogate load is faithful.

Loads the SAME TF checkpoints (40e-5 root run) into BOTH the TF model and the
torch port, then compares pde_rhs on a SHARED seed-0 box sample (n=4096) over
the box:
    logK U(4,7), Z U(0.01,0.99), Y U(0,4), lam3 U(0,1/3), logxi U(-3,5).

Reports MAX ABS diff (torch vs TF) for the HJB residual (rhs - pv) and the
controls (i_d, i_g via 1+theta*i), plus the value-loss scale of the surrogate.
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")
MODELS_TORCH = os.path.join(ROOT, "models_torch")

CKPT_DIR = os.path.join(
    ROOT, "output_001",
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_"
    "10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations1000000",
    "PostDamagePostTech",
)
INPUT_DIM = 7
N = 4096
SEED = 0


def max_abs(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float(np.max(np.abs(a - b)))


def max_rel(a, b, eps=1e-8):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), eps)
    return float(np.max(np.abs(a - b) / denom))


def sample_box(n, seed):
    rng = np.random.default_rng(seed)

    def col(lo, hi):
        return rng.uniform(lo, hi, size=(n, 1)).astype(np.float32)

    logK = col(4.0, 7.0)
    Z = col(0.01, 0.99)
    Y = col(0.0, 4.0)
    logR = col(1.0, 6.0)
    lam3 = col(0.0, 1.0 / 3.0)
    logxi = col(-3.0, 5.0)
    return logK, Z, Y, logR, lam3, logxi


def main():
    sys.path.insert(0, HERE)
    sys.path.insert(0, MODELS_TORCH)
    from tf_to_torch_loader import build_root_torch_model
    import torch

    logK, Z, Y, logR, lam3, logxi = sample_box(N, SEED)

    # ----------------- TORCH side -----------------
    model = build_root_torch_model(load_surrogate=True, ckpt_dir=CKPT_DIR)

    def t(a, rg=False):
        x = torch.from_numpy(a.astype(np.float32))
        x.requires_grad_(rg)
        return x

    tlogK = t(logK, True); tZ = t(Z, True); tY = t(Y, True)
    tlogR = t(logR, False); tlam3 = t(lam3, False); tlogxi = t(logxi, False)

    (rhs_t, pv_t, dvdY_t, c_t, gig_t, gid_t, FOC_d_t, FOC_g_t) = \
        model.pde_rhs(tlogK, tZ, tY, tlogR, tlam3, tlogxi)

    def nd(x):
        return x.detach().cpu().numpy()

    resid_t = nd(rhs_t) - nd(pv_t)
    torch_out = {
        "rhs": nd(rhs_t), "pv": nd(pv_t), "residual(rhs-pv)": resid_t,
        "dv_dY": nd(dvdY_t), "c": nd(c_t),
        "1+theta_g*i_g": nd(gig_t), "1+theta_d*i_d": nd(gid_t),
        "FOC_d": nd(FOC_d_t), "FOC_g": nd(FOC_g_t),
    }
    torch_value_loss = float(np.sqrt(np.mean(resid_t ** 2)))
    torch_focd_loss = float(np.sqrt(np.mean(nd(FOC_d_t) ** 2)))
    torch_focg_loss = float(np.sqrt(np.mean(nd(FOC_g_t) ** 2)))

    # ----------------- TF side -----------------
    if MODELS not in sys.path:
        sys.path.insert(0, MODELS)
    import importlib
    for m in ["feedforward_subnet", "params", "pretrained_paths", "PostDamagePostTech"]:
        if m in sys.modules:
            del sys.modules[m]
    if MODELS_TORCH in sys.path:
        sys.path.remove(MODELS_TORCH)
    if HERE in sys.path:
        sys.path.remove(HERE)
    import tensorflow as tf
    import PostDamagePostTech as TFmod
    from params import investment_rate_activation as tf_inv_act, PARAMS as TFP

    nh = [32, 32, 32, 32]
    tf_v_cfg = {"num_hiddens": nh, "use_bias": True, "activation": "swish",
                "dim": 1, "nn_name": "v_nn", "final_activation": "softplus"}
    tf_ig_cfg = {"num_hiddens": nh, "use_bias": True, "activation": "tanh",
                 "dim": 1, "nn_name": "i_g_nn",
                 "final_activation": tf_inv_act(TFP["θ_g"])}
    tf_id_cfg = {"num_hiddens": nh, "use_bias": True, "activation": "tanh",
                 "dim": 1, "nn_name": "i_d_nn",
                 "final_activation": tf_inv_act(TFP["θ_d"])}

    tf_params = {
        "batch_size": 128, "learning_rates": [1e-4, 1e-4],
        "v_nn_config": tf_v_cfg, "i_g_nn_config": tf_ig_cfg, "i_d_nn_config": tf_id_cfg,
        "num_iterations": 10, "logging_frequency": 10,
        "pretrained_path": None, "learning_rate_schedule_type": "warmup_cosine",
        "tensorboard": False,
        "export_folder": os.path.join(MODELS_TORCH, "_tf_tmp"),
    }
    tf_model = TFmod.PostDamagePostTechModel(tf_params)
    tf_model.v_nn.build((None, INPUT_DIM))
    tf_model.v_nn.load_weights(os.path.join(CKPT_DIR, "v_nn_checkpoint_PostDamagePostTech")).expect_partial()
    tf_model.i_g_nn.build((None, INPUT_DIM))
    tf_model.i_g_nn.load_weights(os.path.join(CKPT_DIR, "i_g_nn_checkpoint_PostDamagePostTech")).expect_partial()
    tf_model.i_d_nn.build((None, INPUT_DIM))
    tf_model.i_d_nn.load_weights(os.path.join(CKPT_DIR, "i_d_nn_checkpoint_PostDamagePostTech")).expect_partial()

    flogK = tf.Variable(logK); fZ = tf.Variable(Z); fY = tf.Variable(Y)
    flogR = tf.Variable(logR); flam3 = tf.Variable(lam3); flogxi = tf.Variable(logxi)
    (rhs_f, pv_f, dvdY_f, c_f, gig_f, gid_f, FOC_d_f, FOC_g_f) = \
        tf_model.pde_rhs(flogK, fZ, fY, flogR, flam3, flogxi)

    resid_f = rhs_f.numpy() - pv_f.numpy()
    tf_out = {
        "rhs": rhs_f.numpy(), "pv": pv_f.numpy(), "residual(rhs-pv)": resid_f,
        "dv_dY": dvdY_f.numpy(), "c": c_f.numpy(),
        "1+theta_g*i_g": gig_f.numpy(), "1+theta_d*i_d": gid_f.numpy(),
        "FOC_d": FOC_d_f.numpy(), "FOC_g": FOC_g_f.numpy(),
    }
    tf_value_loss = float(np.sqrt(np.mean(resid_f ** 2)))
    tf_focd_loss = float(np.sqrt(np.mean(FOC_d_f.numpy() ** 2)))
    tf_focg_loss = float(np.sqrt(np.mean(FOC_g_f.numpy() ** 2)))

    # ----------------- REPORT -----------------
    print("\n==== pde_rhs torch vs TF (max ABS diff / max REL diff) ====")
    diffs = {}
    for k in torch_out:
        ma = max_abs(torch_out[k], tf_out[k])
        mr = max_rel(torch_out[k], tf_out[k])
        diffs[k] = {"abs": ma, "rel": mr}
        print(f"  {k:18s}: abs={ma:.3e}  rel={mr:.3e}")

    print("\n==== surrogate loss scale (this seed-0 sample) ====")
    print(f"  torch value loss (sqrt-mean (rhs-pv)^2): {torch_value_loss:.5e}")
    print(f"  TF    value loss                       : {tf_value_loss:.5e}")
    print(f"  torch FOC_d / FOC_g loss               : {torch_focd_loss:.5e} / {torch_focg_loss:.5e}")
    print(f"  TF    FOC_d / FOC_g loss               : {tf_focd_loss:.5e} / {tf_focg_loss:.5e}")

    result = {
        "residual_abs": diffs["residual(rhs-pv)"]["abs"],
        "residual_rel": diffs["residual(rhs-pv)"]["rel"],
        "i_d_abs": diffs["1+theta_d*i_d"]["abs"],
        "i_g_abs": diffs["1+theta_g*i_g"]["abs"],
        "torch_value_loss": torch_value_loss,
        "tf_value_loss": tf_value_loss,
    }
    print("\nRESULT_JSON " + repr(result))


if __name__ == "__main__":
    main()
