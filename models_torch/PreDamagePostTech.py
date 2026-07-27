"""
PyTorch port of models/PreDamagePostTech.py  ::  pde_rhs + controls.

PRE-DAMAGE & POST-TECHNOLOGY regime.  Relative to the ROOT regime
(PostDamagePostTech) this regime ADDS the TECH-JUMP operator: a frozen
PostDamagePostTech neighbor value net is loaded, and the HJB rhs gains a
``Jump_term`` summed over the L damage realizations:

    g_l       = exp(-1/ξ * (v_PostDamagePostTech - v))
    Jump_term += J_d/L * g_l * (v_PostDamagePostTech - v)
               + ξ * J_d/L * (1 - g_l + g_l * log(g_l))

with J_d the damage-jump intensity (active only for Y > y_lower).

Two further differences vs the root regime are faithfully reproduced:

  * The MAIN nets (v_nn / i_g_nn / i_d_nn) take a **6-dim** input here, with X
    assembled WITHOUT the λ3 column:
        X = concat([logK, Z, Y, A_g_prime_prime*ones, logξ, logξ], dim=1)
    (The root regime uses a 7-dim X with λ3 in slot 4.)  The FROZEN neighbor
    PostDamagePostTech net is still a 7-dim net and is fed a 7-dim X.

  * v_logN_term here is  (λ1 + λ2*Y) * v_y_term + (λ2) * v_yy_term  -- it does
    NOT carry the λ3*(Y - y_upper) / λ3 terms that the post-damage root regime
    has (there is no realized damage curvature pre-damage).

Only the forward / inference economics is ported (NOT the training loop).
Derivatives are taken with torch autograd, mirroring TF
``tf.gradients(..., unconnected_gradients='zero')`` (allow_unused + zero-fill,
create_graph for second derivatives).
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


class PreDamagePostTechModel:
    """Pre-damage & post-technology HJB (torch inference port)."""

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

        # Frozen neighbor: PostDamagePostTech value net (the tech-jump target).
        # This is a 7-dim-input net; its weights are loaded by the validator via
        # the harness (load_tf_weights_into_torch).
        self.v_PostDamagePostTech_nn = FeedForwardSubNet(self.params["v_nn_config"])

    # ------------------------------------------------------------------
    def pde_rhs(self, logK, Z, Y, logR, λ3, logξ):
        """Returns (rhs, pv, dv_dY, c, 1+θ_g*i_g, 1+θ_d*i_d, FOC_d, FOC_g).

        Bit-faithful port of models/PreDamagePostTech.py::pde_rhs.
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
        # MAIN nets take a 6-dim input (NO λ3 column), matching the TF source.
        X = torch.cat(
            [logK, Z, Y, A_g_prime_prime * torch.ones_like(Y), logξ, logξ],
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

        # v = V - log N, so dV/dY = dv_dY - d(log N)/dY, d(log N)/dY = λ1 + λ2*Y
        h_y = - 1.0 / ξ * (dv_dY - (λ1 + λ2 * Y)) * η * A_d * (1 - Z) * K * ϛ

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

        # Damage function is from the 2024 SITE Paper.  (Pre-damage: no λ3 terms.)
        v_logN_term = (λ1 + λ2 * Y) * v_y_term + (λ2) * v_yy_term

        # ---- Jump terms (tech jump to PostDamagePostTech) ----
        J_d = r1 * (torch.exp(r2 / 2 * torch.pow(Y - y_lower, 2)) - 1) \
            * (Y > y_lower).to(Y.dtype)

        Jump_term = torch.zeros_like(dv_dZ)

        for l in range(L):  # L is the number of damage realizations
            λ3_l = λ3_values[l]

            X_jump = torch.cat(
                [logK, Z,
                 torch.ones_like(Y) * y_upper,
                 λ3_l * torch.ones_like(Y),
                 A_g_prime_prime * torch.ones_like(Y),
                 logξ, logξ],
                dim=1,
            )
            v_PostDamagePostTech = self.v_PostDamagePostTech_nn(X_jump)
            g_l = torch.exp(-1 / ξ * (v_PostDamagePostTech - v))
            Jump_term = Jump_term \
                + J_d / L * g_l * (v_PostDamagePostTech - v) \
                + ξ * J_d / L * (1 - g_l + g_l * torch.log(g_l))

        # ---- HJB Equation ----
        rhs = flow \
            + v_logK_term * dv_dlogK + v_logKlogK_term * d2v_dlogK2 \
            + v_Z_term * dv_dZ + v_ZZ_term * d2v_dZ2 \
            + h_d * (dv_dlogK - Z * dv_dZ) * (1 - Z) * σ_d + h_g * (dv_dlogK + (1 - Z) * dv_dZ) * Z * σ_g \
            + v_logK_Z_term * d2v_dlogKdZ \
            + dv_dY * v_y_term + v_yy_term * d2v_dY2 + 0.5 * ξ * (torch.pow(h_d, 2) + torch.pow(h_g, 2) + torch.pow(h_y, 2)) \
            + (-1.0) * v_logN_term \
            + Jump_term

        # ---- FOCs ----
        marginal_util_c = δ / inside_log

        FOC_d = -marginal_util_c + Γ_d * θ_d / (inside_log_i_d) * (dv_dlogK - Z * dv_dZ)
        FOC_g = -marginal_util_c + Γ_g * θ_g / (inside_log_i_g) * (dv_dlogK + (1.0 - Z) * dv_dZ)

        return rhs, pv, dv_dY, c, 1.0 + θ_g * i_g, 1.0 + θ_d * i_d, FOC_d, FOC_g
