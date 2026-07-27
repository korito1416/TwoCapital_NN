"""
PyTorch port of models/PreDamagePreTech.py  ::  pde_rhs + controls.

PRE-damage & PRE-technology regime.  Relative to the ROOT regime
(PostDamagePostTech) this adds, EXACTLY mirroring the TF source:

  * the i_r R&D control:  i_r = exp(-i_r_nn(X))   (I_r / K)
  * the logR state dynamics (drift v_logR_term, diffusion, robustness h_r)
  * the TECH-jump operator to PreDamagePostTech (and, only when π<1, the
    intermediate tech jump to PreDamageIntermTech)
  * the DAMAGE-jump operator (sum over L damage realizations) to
    PostDamagePreTech
  * the robustness distortions h_d, h_g, h_y, h_r and FOC_d, FOC_g, FOC_r

Only the forward / inference economics is ported (NOT the training loop).
The neighbour value networks (frozen) are loaded by the validation harness and
assigned onto the model before calling pde_rhs.

The state input assembled for THIS regime's own networks (exactly as in TF):
    X = concat([logK, Z, Y, logR, logξ, logξ, logξ], dim=1)   # 7 columns
"""

import torch

from feedforward_subnet import FeedForwardSubNet
from params import PARAMS, investment_rate_activation


def _grad(y, x, create_graph=False):
    """torch analogue of tf.gradients(y, x, unconnected_gradients='zero')."""
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


class PreDamagePreTechModel:
    """Pre-damage & pre-technology HJB (torch inference port)."""

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
        self.i_r_nn = FeedForwardSubNet(self.params["i_r_nn_config"])

        # Frozen neighbour value networks (assigned/loaded externally by the
        # harness).  Mirrors the TF self.v_*_nn members.
        self.v_PreDamagePostTech_nn = None
        self.v_PreDamageIntermTech_nn = None
        self.v_PostDamagePreTech_nn = None

        # Whether the intermediate tech jump is active (π < 1).
        self.use_intermediate_tech_jump = (
            float(self.params.get("π", PARAMS.get("π", 0.04))) < 1.0 - 1e-12
        )

    # ------------------------------------------------------------------
    def pde_rhs(self, logK, Z, Y, logR, λ3, logξ):
        """Returns (rhs, pv, dv_dY, c, 1+θ_g*i_g, 1+θ_d*i_d, FOC_d, FOC_g, FOC_r, dv_dlogR).

        Bit-faithful port of models/PreDamagePreTech.py::pde_rhs.
        logK, Z, Y, logR must be [-1,1] tensors with requires_grad=True.
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
        tech_jump_intensity_scale = p.get('tech_jump_intensity_scale', 1.0)

        θ_bar = p['θ_bar']; η = p['η']; ϛ = p['ϛ']

        λ1 = p['λ1']; λ2 = p['λ2']; L = p['L']
        λ3_values = p['λ3_values']; r1 = p['r1']; r2 = p['r2']
        y_lower = p['y_lower']; y_upper = p['y_upper']

        # ---- Compute value functions and derivatives ----
        X = torch.cat([logK, Z, Y, logR, logξ, logξ, logξ], dim=1)

        v = self.v_nn(X)
        i_g = self.i_g_nn(X)
        i_d = self.i_d_nn(X)
        i_r = torch.exp(- self.i_r_nn(X))  # I_r / K

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

        dv_dlogR = _grad(v, logR, create_graph=True)
        d2v_dlogR2 = _grad(dv_dlogR, logR, create_graph=False)

        # ---- drift distortions ----
        h_d = - 1.0 / ξ * ((dv_dlogK - Z * dv_dZ) * (1 - Z) * σ_d)
        h_g = - 1.0 / ξ * ((dv_dlogK + (1 - Z) * dv_dZ) * Z * σ_g)

        # v = V - log N, so dV/dY = dv_dY - d(logN)/dY = dv_dY - (λ1 + λ2*Y)
        h_y = - 1.0 / ξ * (dv_dY - (λ1 + λ2 * Y)) * η * A_d * (1 - Z) * K * ϛ

        h_r = - 1.0 / ξ * dv_dlogR * σ_κ

        # ---- consumption and flow ----
        pv = δ * v

        c = (A_d - i_d) * (1 - Z) + (A_g - i_g) * Z - i_r
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

        # Damage function is from the 2024 SITE Paper.
        v_logN_term = (λ1 + λ2 * Y) * v_y_term + (λ2) * v_yy_term

        v_logR_term = - ζ + ψ0 * torch.exp(ψ1 * (torch.log(i_r) + logK - logR)) - 0.5 * σ_κ ** 2 + σ_κ * h_r
        v_logRlogR_term = 0.5 * σ_κ ** 2

        # ---- Jump terms ----
        J_g_prime = tech_jump_intensity_scale * (1 - π) * torch.exp(logR) / varrho
        J_g_prime_prime = tech_jump_intensity_scale * π * torch.exp(logR) / varrho

        v_PreDamagePostTech = self.v_PreDamagePostTech_nn(
            torch.cat([logK, Z, Y, A_g_prime_prime * torch.ones_like(Y), logξ, logξ], dim=1)
        )

        g_l_prime_prime = torch.exp(-1 / ξ * (v_PreDamagePostTech - v))

        Jump_term = J_g_prime_prime * g_l_prime_prime * (v_PreDamagePostTech - v) \
            + ξ * J_g_prime_prime * (1 - g_l_prime_prime + g_l_prime_prime * torch.log(g_l_prime_prime))

        if self.use_intermediate_tech_jump:
            v_PreDamageIntermTech = self.v_PreDamageIntermTech_nn(
                torch.cat([logK, Z, Y, logR, logξ, logξ, logξ], dim=1)
            )
            g_l_prime = torch.exp(-1 / ξ * (v_PreDamageIntermTech - v))
            Jump_term = Jump_term + J_g_prime * g_l_prime * (v_PreDamageIntermTech - v) \
                + ξ * J_g_prime * (1 - g_l_prime + g_l_prime * torch.log(g_l_prime))

        # ---- Damage Jump ----
        J_d = r1 * (torch.exp(r2 / 2 * torch.pow(Y - y_lower, 2)) - 1) * (Y > y_lower).to(Y.dtype)

        for l in range(L):  # L is the number of damage realizations
            λ3_l = λ3_values[l]

            v_PostDamagePreTech = self.v_PostDamagePreTech_nn(
                torch.cat(
                    [logK, Z, torch.ones_like(Y) * y_upper, logR,
                     λ3_l * torch.ones_like(Y), logξ, logξ, logξ],
                    dim=1,
                )
            )
            g_l = torch.exp(-1 / ξ * (v_PostDamagePreTech - v))
            Jump_term = Jump_term + J_d / L * g_l * (v_PostDamagePreTech - v) \
                + ξ * J_d / L * (1 - g_l + g_l * torch.log(g_l))

        # ---- HJB Equation ----
        rhs = flow \
            + v_logK_term * dv_dlogK + v_logKlogK_term * d2v_dlogK2 \
            + v_Z_term * dv_dZ + v_ZZ_term * d2v_dZ2 \
            + h_d * (dv_dlogK - Z * dv_dZ) * (1 - Z) * σ_d + h_g * (dv_dlogK + (1 - Z) * dv_dZ) * Z * σ_g \
            + v_logK_Z_term * d2v_dlogKdZ \
            + dv_dY * v_y_term + v_yy_term * d2v_dY2 + 0.5 * ξ * (torch.pow(h_d, 2) + torch.pow(h_g, 2) + torch.pow(h_y, 2)) \
            + (-1.0) * v_logN_term \
            + v_logR_term * dv_dlogR + v_logRlogR_term * d2v_dlogR2 + 0.5 * ξ * torch.pow(h_r, 2) \
            + Jump_term

        # ---- FOCs ----
        marginal_util_c = δ / inside_log

        FOC_d = -marginal_util_c + Γ_d * θ_d / (inside_log_i_d) * (dv_dlogK - Z * dv_dZ)
        FOC_g = -marginal_util_c + Γ_g * θ_g / (inside_log_i_g) * (dv_dlogK + (1.0 - Z) * dv_dZ)
        FOC_r = -marginal_util_c + ψ0 * ψ1 * torch.exp(ψ1 * (torch.log(i_r) + logK - logR)) * dv_dlogR / i_r

        return rhs, pv, dv_dY, c, 1.0 + θ_g * i_g, 1.0 + θ_d * i_d, FOC_d, FOC_g, FOC_r, dv_dlogR
