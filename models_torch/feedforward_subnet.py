"""
PyTorch port of models/feedforward_subnet.py :: FeedForwardSubNet (FORWARD only).

This reproduces the *inference* forward pass of the TensorFlow Keras model
exactly, so that loading the trained TF weights gives bit-faithful outputs.

TF Keras structure (models/feedforward_subnet.py, .call):

    x = bn_layers[0](x)                         # batchnorm on the raw input
    x_inputs = []
    for i in range(len(dense_layers) - 1):      # the hidden layers
        x = dense_layers[i](x)                  # Dense INCLUDES its activation
        x = bn_layers[i+1](x)                   # batchnorm AFTER activation
        x_inputs.append(x)
    x = add_n(x_inputs)                         # SUM of all hidden post-bn acts
    x = dense_layers[-1](x)                     # output Dense (+ final activation)
    return x

Two subtleties that MUST be honoured for fidelity:

1. In Keras a ``Dense(activation=act)`` layer applies the activation INSIDE the
   layer, i.e. ``act(x @ W + b)``.  The BatchNormalization is then applied to the
   *activated* output.  So the per-hidden-layer op is  bn( act( linear(x) ) ).

2. BatchNormalization at inference (training=False) is a pure affine map using
   the stored moving statistics:
       y = gamma * (x - moving_mean) / sqrt(moving_variance + eps) + beta
   with epsilon = 1e-6 (matching the Keras config in the TF model).

The output Dense always uses a bias and its ``final_activation`` (which may be a
Python callable such as the bounded investment-rate activation).
"""

import torch
import torch.nn as nn


BN_EPS = 1e-6  # matches tf.keras.layers.BatchNormalization(epsilon=1e-6)


def _swish(x):
    # tf.keras 'swish' == x * sigmoid(x)  (beta = 1)
    return x * torch.sigmoid(x)


def _resolve_activation(act):
    """Map a TF activation spec (string / None / callable) to a torch callable."""
    if act is None:
        return None
    if callable(act):
        # e.g. the bounded investment_rate_activation lambda (already torch)
        return act
    name = str(act).lower()
    if name == "swish" or name == "silu":
        return _swish
    if name == "tanh":
        return torch.tanh
    if name == "softplus":
        return nn.functional.softplus
    if name == "relu":
        return torch.relu
    if name == "sigmoid":
        return torch.sigmoid
    if name == "elu":
        return nn.functional.elu
    if name == "linear" or name == "none":
        return None
    raise ValueError(f"Unsupported activation: {act!r}")


class _BatchNorm1dInference(nn.Module):
    """Affine inference-only batchnorm matching Keras BN with stored moments."""

    def __init__(self, num_features, eps=BN_EPS):
        super().__init__()
        self.eps = eps
        self.register_buffer("gamma", torch.ones(num_features))
        self.register_buffer("beta", torch.zeros(num_features))
        self.register_buffer("moving_mean", torch.zeros(num_features))
        self.register_buffer("moving_variance", torch.ones(num_features))

    def forward(self, x):
        return self.gamma * (x - self.moving_mean) / torch.sqrt(
            self.moving_variance + self.eps
        ) + self.beta


class FeedForwardSubNet(nn.Module):
    """Torch port of the TF FeedForwardSubNet (forward / inference only)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        num_hiddens = list(config["num_hiddens"])
        dim = config["dim"]
        self.use_bias = config["use_bias"]

        n_dense = len(num_hiddens) + 1            # hidden layers + 1 output
        n_bn = len(num_hiddens) + 1               # bn[0] on input + bn after each hidden

        # Activations.  Hidden layers share config['activation']; the output uses
        # config['final_activation'].
        self.hidden_activation = _resolve_activation(config.get("activation"))
        self.final_activation = _resolve_activation(config.get("final_activation"))

        # Dense layers.  Input width is unknown until first forward, so we build
        # lazily using LazyLinear-style deferral via a placeholder, but since the
        # TF input dim is fixed (7) and the checkpoint encodes shapes, we build
        # them eagerly from the kernel shapes when weights are loaded.  To keep
        # __init__ self-contained we create Linear modules with placeholder
        # in_features that get overwritten in load_tf_weights; here we use the
        # known structure: first hidden in_features is set on weight load.
        self.dense_layers = nn.ModuleList()
        # hidden layers
        prev = None  # set on weight load for layer 0
        for i, h in enumerate(num_hiddens):
            in_f = prev if prev is not None else 1  # placeholder, fixed on load
            self.dense_layers.append(nn.Linear(in_f, h, bias=self.use_bias))
            prev = h
        # output layer (always has bias)
        self.dense_layers.append(nn.Linear(prev, dim, bias=True))

        # BatchNorm layers: bn[0] on input, bn[1..n] after each hidden dense.
        # bn[0] feature count = input dim (unknown until load); placeholder=1.
        self.bn_layers = nn.ModuleList()
        self.bn_layers.append(_BatchNorm1dInference(1))  # input bn, fixed on load
        for h in num_hiddens:
            self.bn_layers.append(_BatchNorm1dInference(h))

    def forward(self, x):
        x = self.bn_layers[0](x)
        x_inputs = []
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            if self.hidden_activation is not None:
                x = self.hidden_activation(x)
            x = self.bn_layers[i + 1](x)
            x_inputs.append(x)
        x = torch.stack(x_inputs, dim=0).sum(dim=0)
        x = self.dense_layers[-1](x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x
