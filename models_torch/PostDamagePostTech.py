"""
PyTorch port of models/PostDamagePostTech.py  ::  pde_rhs + controls.

ROOT regime (terminal, post-damage & post-technology): NO jump operators.
This is a bit-faithful translation of the TF ``pde_rhs`` economics.

Only the forward / inference economics is ported here (NOT the training loop).
Derivatives of the value function are taken with torch autograd, mirroring the
TF ``tf.gradients(..., unconnected_gradients='zero')`` calls (allow_unused +
zero-fill, create_graph for the second derivatives).

The state input assembled for the networks is (exactly as in TF):
    X = concat([logK, Z, Y, λ3, A_g_prime_prime * ones_like(Y), logξ, logξ], dim=1)
"""

import torch

from feedforward_subnet import FeedForwardSubNet
from params import PARAMS, investment_rate_activation


def _grad(y, x, create_graph=False):
    """torch analogue of tf.gradients(y, x, unconnected_gradients='zero').

    Returns d y / d x as a [-1, 1] tensor; zero-filled if unconnected.
    """
    g = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if g is None:
        g = torch.zeros_like(x)
    return g.reshape(-1, 1)


class PostDamagePostTechModel:
    """Post-damage & post-technology HJB (torch inference port)."""

    def __init__(self, params=None):
        self.params = PARAMS.copy()
        self.params.update(params or {})

        # Resolve the bounded custom control activation if requested (mirrors the
        # TF __main__ block that swaps in investment_rate_activation for "custom").
        for cfg_key, theta_key in (("i_g_nn_config", "θ_g"), ("i_d_nn_config", "θ_d")):
            cfg = self.params.get(cfg_key)
            if cfg is not None and cfg.get("final_activation") == "custom":
                cfg["final_activation"] = investment_rate_activation(self.params[theta_key])

        self.v_nn = FeedForwardSubNet(self.params["v_nn_config"])
        self.i_g_nn = FeedForwardSubNet(self.params["i_g_nn_config"])
        self.i_d_nn = FeedForwardSubNet(self.params["i_d_nn_config"])

    # ------------------------------------------------------------------
    def pde_rhs(self, logK, Z, Y, logR, λ3, logξ):
        """Returns (rhs, pv, dv_dY, c, 1+θ_g*i_g, 1+θ_d*i_d, FOC_d, FOC_g).

        Bit-faithful port of models/PostDamagePostTech.py::pde_rhs.
        Each state argument must be a [-1, 1] tensor with requires_grad=True for
        the ones we differentiate (logK, Z, Y).
        """
        p = self.params
        A_d = p['A_d']
        A_g = p['A_g']
        A_g_prime = p['A_g_prime']
        A_g_prime_prime = p['A_g_prime_prime']
        π = p['π']

        δ = p['δ']

        α_d = p['α_d']; Γ_d = p['Γ_d']; θ_d = p['θ_d']; σ_d = p['σ_d']
        α_g = p['α_g']; Γ_g = p['Γ_g']; θ_g = p['θ_g']; σ_g = p['σ_g']

        ζ = p['ζ']; ψ0 = p['ψ0']; ψ1 = p['ψ1']; σ_κ = p['σ_κ']; varrho = p['varrho']

        θ_bar = p['θ_bar']; η = p['η']; ϛ = p['ϛ']

        λ1 = p['λ1']; λ2 = p['λ2']; L = p['L']
        λ3_values = p['λ3_values']; r1 = p['r1']; r2 = p['r2']
        y_lower = p['y_lower']; y_upper = p['y_upper']

        # ---- Compute value functions and derivatives ----
        X = torch.cat(
            [logK, Z, Y, λ3, A_g_prime_prime * torch.ones_like(Y), logξ, logξ],
            dim=1,
        )

        v = self.v_nn(X)
        i_g = self.i_g_nn(X)
        i_d = self.i_d_nn(X)

        ξ = torch.exp(logξ)
        K = torch.exp(logK)

        # ---- derivatives ----
        dv_dlogK = _grad(v, logK, create_graph=True)
        d2v_dlogK2 = _grad(dv_dlogK, logK, create_graph=False)
        d2v_dlogKdZ = _grad(dv_dlogK, Z, create_graph=False)

        dv_dZ = _grad(v, Z, create_graph=True)
        d2v_dZ2 = _grad(dv_dZ, Z, create_graph=False)

        dv_dY = _grad(v, Y, create_graph=True)
        d2v_dY2 = _grad(dv_dY, Y, create_graph=False)

        # ---- drift distortions ----
        h_d = - 1.0 / ξ * ((dv_dlogK - Z * dv_dZ) * (1 - Z) * σ_d)
        h_g = - 1.0 / ξ * ((dv_dlogK + (1 - Z) * dv_dZ) * Z * σ_g)

        h_y = - 1.0 / ξ * (dv_dY - (λ1 + λ2 * Y + λ3 * (Y - y_upper))) * η * A_d * (1 - Z) * K * ϛ

        # ---- consumption and flow ----
        pv = δ * v

        c = (A_d - i_d) * (1 - Z) + (A_g_prime_prime - i_g) * Z
        inside_log = torch.clamp(c, min=1e-8).reshape(-1, 1)
        flow = δ * (torch.log(inside_log) + logK)

        v_logKlogK_term = (σ_d ** 2 * (1 - Z) ** 2 + σ_g ** 2 * Z ** 2) / 2.0

        inside_log_i_d = torch.clamp(1.0 + θ_d * i_d, min=1e-8).reshape(-1, 1)
        inside_log_i_g = torch.clamp(1.0 + θ_g * i_g, min=1e-8).reshape(-1, 1)

        v_logK_term = (α_d + Γ_d * torch.log(inside_log_i_d)) * (1 - Z) \
            + (α_g + Γ_g * torch.log(inside_log_i_g)) * Z \
            - v_logKlogK_term

        v_Z_term = (α_g + Γ_g * torch.log(inside_log_i_g)
                    - (α_d + Γ_d * torch.log(inside_log_i_d))
                    - Z * σ_g ** 2
                    + (1 - Z) * σ_d ** 2) * Z * (1 - Z)

        v_ZZ_term = 0.5 * Z ** 2 * (1 - Z) ** 2 * (σ_g ** 2 + σ_d ** 2)

        v_logK_Z_term = - Z * (1 - Z) ** 2 * σ_d ** 2 + Z ** 2 * (1.0 - Z) * σ_g ** 2

        v_y_term = (θ_bar + h_y * ϛ) * η * A_d * (1 - Z) * K

        v_yy_term = 0.5 * ϛ ** 2 * (η * A_d * (1 - Z) * K) ** 2

        v_logN_term = (λ1 + λ2 * Y + λ3 * (Y - y_upper)) * v_y_term + (λ2 + λ3) * v_yy_term

        rhs = flow \
            + v_logK_term * dv_dlogK + v_logKlogK_term * d2v_dlogK2 \
            + v_Z_term * dv_dZ + v_ZZ_term * d2v_dZ2 \
            + h_d * (dv_dlogK - Z * dv_dZ) * (1 - Z) * σ_d + h_g * (dv_dlogK + (1 - Z) * dv_dZ) * Z * σ_g \
            + v_logK_Z_term * d2v_dlogKdZ \
            + dv_dY * v_y_term + v_yy_term * d2v_dY2 + 0.5 * ξ * (torch.pow(h_d, 2) + torch.pow(h_g, 2) + torch.pow(h_y, 2)) \
            + (-1.0) * v_logN_term

        # ---- FOCs ----
        marginal_util_c = δ / inside_log

        FOC_d = -marginal_util_c + Γ_d * θ_d / (inside_log_i_d) * (dv_dlogK - Z * dv_dZ)
        FOC_g = -marginal_util_c + Γ_g * θ_g / (inside_log_i_g) * (dv_dlogK + (1.0 - Z) * dv_dZ)

        return rhs, pv, dv_dY, c, 1.0 + θ_g * i_g, 1.0 + θ_d * i_d, FOC_d, FOC_g
