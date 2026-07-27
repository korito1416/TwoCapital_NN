"""
Reusable TF <-> torch validation harness.

Given a TF checkpoint prefix (e.g.
   .../PostDamagePostTech/v_nn_checkpoint_PostDamagePostTech)
this module:

  * reads the Dense kernels/biases and BatchNorm moments via
    tf.train.list_variables / tf.train.load_variable,
  * loads them into a torch ``FeedForwardSubNet`` (TRANSPOSING Dense kernels for
    torch's ``Linear`` which stores weight as [out, in]),
  * optionally builds the *reference* TF Keras model and compares the two
    forward passes on random inputs, returning the max relative error.

The checkpoint variable layout (object-graph format) is:
  dense_layers/<i>/kernel/.ATTRIBUTES/VARIABLE_VALUE   [in, out]
  dense_layers/<i>/bias/.ATTRIBUTES/VARIABLE_VALUE     [out]
  bn_layers/<j>/{gamma,beta,moving_mean,moving_variance}/.ATTRIBUTES/VARIABLE_VALUE
"""

import os
import numpy as np
import torch
import torch.nn as nn

from feedforward_subnet import FeedForwardSubNet, _BatchNorm1dInference


def _read_tf_checkpoint(ckpt_prefix):
    """Return dict name->np.ndarray for all variables in a TF checkpoint."""
    import tensorflow as tf

    out = {}
    for name, _shape in tf.train.list_variables(ckpt_prefix):
        if name == "_CHECKPOINTABLE_OBJECT_GRAPH":
            continue
        out[name] = tf.train.load_variable(ckpt_prefix, name)
    return out


def _attr(d, base):
    return d[base + "/.ATTRIBUTES/VARIABLE_VALUE"]


def load_tf_weights_into_torch(net: FeedForwardSubNet, ckpt_prefix: str):
    """Load a TF FeedForwardSubNet checkpoint into the torch net (in place)."""
    d = _read_tf_checkpoint(ckpt_prefix)

    # ---- Dense layers ----
    n_dense = len(net.dense_layers)
    for i in range(n_dense):
        kernel = np.asarray(_attr(d, f"dense_layers/{i}/kernel"))  # [in, out]
        in_f, out_f = kernel.shape
        has_bias = (f"dense_layers/{i}/bias/.ATTRIBUTES/VARIABLE_VALUE") in d
        # Rebuild the Linear with correct in/out (placeholders were used at init).
        lin = nn.Linear(in_f, out_f, bias=has_bias)
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(kernel.T.copy().astype(np.float32)))
            if has_bias:
                bias = np.asarray(_attr(d, f"dense_layers/{i}/bias")).astype(np.float32)
                lin.bias.copy_(torch.from_numpy(bias))
        net.dense_layers[i] = lin

    # ---- BatchNorm layers ----
    n_bn = len(net.bn_layers)
    for j in range(n_bn):
        gamma = np.asarray(_attr(d, f"bn_layers/{j}/gamma")).astype(np.float32)
        beta = np.asarray(_attr(d, f"bn_layers/{j}/beta")).astype(np.float32)
        mean = np.asarray(_attr(d, f"bn_layers/{j}/moving_mean")).astype(np.float32)
        var = np.asarray(_attr(d, f"bn_layers/{j}/moving_variance")).astype(np.float32)
        bn = _BatchNorm1dInference(gamma.shape[0])
        with torch.no_grad():
            bn.gamma.copy_(torch.from_numpy(gamma))
            bn.beta.copy_(torch.from_numpy(beta))
            bn.moving_mean.copy_(torch.from_numpy(mean))
            bn.moving_variance.copy_(torch.from_numpy(var))
        net.bn_layers[j] = bn

    net.eval()
    return net


def build_torch_net_from_checkpoint(config, ckpt_prefix):
    net = FeedForwardSubNet(config)
    load_tf_weights_into_torch(net, ckpt_prefix)
    return net


# ---------------------------------------------------------------------------
# TF reference model (only used for cross-checking)
# ---------------------------------------------------------------------------
def build_tf_reference(config, ckpt_prefix, input_dim):
    """Build the original TF FeedForwardSubNet and restore the checkpoint."""
    import importlib.util
    import tensorflow as tf

    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )
    # Load the TF feedforward_subnet under a private name to avoid clashing with
    # the torch module of the same basename that lives in models_torch/.
    spec = importlib.util.spec_from_file_location(
        "_tf_feedforward_subnet", os.path.join(models_dir, "feedforward_subnet.py")
    )
    tf_ffn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tf_ffn)
    TFNet = tf_ffn.FeedForwardSubNet

    tf_net = TFNet(config)
    tf_net.build((None, input_dim))
    tf_net.load_weights(ckpt_prefix).expect_partial()
    return tf_net


def max_rel_error(a, b, eps=1e-8):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.maximum(denom, eps)
    return float(np.max(np.abs(a - b) / denom))


def compare_nn_forward(config, ckpt_prefix, input_dim, n=4096, seed=0):
    """Compare torch vs TF forward on random inputs. Returns max rel error."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 6.0, size=(n, input_dim)).astype(np.float32)

    torch_net = build_torch_net_from_checkpoint(config, ckpt_prefix)
    with torch.no_grad():
        y_torch = torch_net(torch.from_numpy(x)).cpu().numpy()

    import tensorflow as tf
    tf_net = build_tf_reference(config, ckpt_prefix, input_dim)
    y_tf = tf_net(tf.convert_to_tensor(x), training=False).numpy()

    return max_rel_error(y_torch, y_tf), y_torch, y_tf
