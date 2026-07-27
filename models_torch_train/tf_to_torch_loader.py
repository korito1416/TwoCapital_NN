"""
TF -> torch weight loader for the ROOT regime (PostDamagePostTech).

This is the canonical loader for treating the TF-trained DGM-PIA solution as a
SURROGATE that we load into the torch nets.  It REUSES the validated loader in
``models_torch/tf_torch_harness.py`` (which reads the tf.keras object-graph
checkpoint, transposes Dense kernels [in,out] -> Linear [out,in], and copies
BatchNorm moments).  This file just wraps it with:

  * a build_root_torch_model() helper that constructs the torch
    PostDamagePostTechModel and (optionally) loads the three TF checkpoints
    (v_nn / i_g_nn / i_d_nn) into it, and
  * a build_trainable_net() helper that constructs a FeedForwardSubNet with the
    CORRECT first-layer input width (7) so it can be trained from scratch (the
    bare FeedForwardSubNet.__init__ leaves layer-0 in_features as a placeholder
    that is only fixed when TF weights are loaded).

The TF Dense kernel is [in, out]; torch Linear weight is [out, in]  -> TRANSPOSE.
Bias copies directly.  (Handled inside load_tf_weights_into_torch.)
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_TORCH = os.path.join(ROOT, "models_torch")

# Make the validated torch inference port importable.
if MODELS_TORCH not in sys.path:
    sys.path.insert(0, MODELS_TORCH)

from feedforward_subnet import FeedForwardSubNet, _BatchNorm1dInference  # noqa: E402
from tf_torch_harness import load_tf_weights_into_torch  # noqa: E402  (REUSED)

INPUT_DIM = 7
REGIME = "PostDamagePostTech"

DEFAULT_CKPT_DIR = os.path.join(
    ROOT,
    "output_001",
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_"
    "10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations1000000",
    "PostDamagePostTech",
)


def make_root_configs():
    """NN configs matching the trained TF run (v: swish/softplus, i_*: tanh/custom)."""
    nh = [32, 32, 32, 32]
    v_cfg = {"num_hiddens": nh, "use_bias": True, "activation": "swish",
             "dim": 1, "nn_name": "v_nn", "final_activation": "softplus"}
    ig_cfg = {"num_hiddens": nh, "use_bias": True, "activation": "tanh",
              "dim": 1, "nn_name": "i_g_nn", "final_activation": "custom"}
    id_cfg = {"num_hiddens": nh, "use_bias": True, "activation": "tanh",
              "dim": 1, "nn_name": "i_d_nn", "final_activation": "custom"}
    return v_cfg, ig_cfg, id_cfg


def _fix_input_layer(net: FeedForwardSubNet, input_dim=INPUT_DIM):
    """Rebuild layer-0 Linear and the input BatchNorm with the right widths.

    The bare FeedForwardSubNet.__init__ uses placeholder in_features=1 for the
    first Dense and num_features=1 for the input BN (they are only fixed when TF
    weights are loaded).  For from-scratch training we set them to ``input_dim``
    and re-randomise.  BN[0] starts as identity (gamma=1, beta=0, mean=0, var=1).
    """
    h0 = net.dense_layers[0].out_features
    use_bias = net.dense_layers[0].bias is not None
    net.dense_layers[0] = nn.Linear(input_dim, h0, bias=use_bias)
    net.bn_layers[0] = _BatchNorm1dInference(input_dim)
    return net


def build_trainable_net(config, input_dim=INPUT_DIM, seed=None):
    """A FeedForwardSubNet with correct input width, ready for training."""
    if seed is not None:
        torch.manual_seed(seed)
    net = FeedForwardSubNet(config)
    _fix_input_layer(net, input_dim)
    return net


def build_root_torch_model(load_surrogate=True, ckpt_dir=DEFAULT_CKPT_DIR,
                           configs=None, dtype=torch.float32):
    """Construct the torch root model; optionally load the TF surrogate weights.

    Returns the torch PostDamagePostTechModel (from models_torch/).  When
    ``load_surrogate`` is True the three TF checkpoints are loaded in place.
    """
    from PostDamagePostTech import PostDamagePostTechModel as TorchModel

    v_cfg, ig_cfg, id_cfg = configs or make_root_configs()
    model = TorchModel({
        "v_nn_config": v_cfg, "i_g_nn_config": ig_cfg, "i_d_nn_config": id_cfg,
    })

    if load_surrogate:
        load_root_surrogate(model, ckpt_dir)
    else:
        # Fresh-init nets with correct input width.
        model.v_nn = _fix_input_layer(model.v_nn)
        model.i_g_nn = _fix_input_layer(model.i_g_nn)
        model.i_d_nn = _fix_input_layer(model.i_d_nn)

    if dtype != torch.float32:
        for net in (model.v_nn, model.i_g_nn, model.i_d_nn):
            net.to(dtype)
    return model


def load_root_surrogate(model, ckpt_dir=DEFAULT_CKPT_DIR):
    """Load the TF v_nn/i_g_nn/i_d_nn checkpoints into a torch root model (in place)."""
    load_tf_weights_into_torch(
        model.v_nn, os.path.join(ckpt_dir, f"v_nn_checkpoint_{REGIME}"))
    load_tf_weights_into_torch(
        model.i_g_nn, os.path.join(ckpt_dir, f"i_g_nn_checkpoint_{REGIME}"))
    load_tf_weights_into_torch(
        model.i_d_nn, os.path.join(ckpt_dir, f"i_d_nn_checkpoint_{REGIME}"))
    return model


def assemble_X(logK, Z, Y, lam3, logxi, A_g_prime_prime):
    """Assemble the 7-col network input exactly as the TF/torch pde_rhs does."""
    return np.concatenate(
        [logK, Z, Y, lam3, A_g_prime_prime * np.ones_like(Y), logxi, logxi],
        axis=1,
    ).astype(np.float32)
