"""Root-regime costate solver with xi-conditioned RobustMinimizerNet.

This is the first executable prototype for the redesigned architecture:

  CostateFieldNet + EGM line-integral value recovery
  + RobustMinimizerNet(logxi-conditioned)
  + costate-aware controls.

Scope: terminal PostDamagePostTech only.  There are no jump distortions in this
regime, so the learned inner minimizer covers Brownian drift distortions
``h_d,h_g,h_y``.  The next stage extends the same robust net to jump ``log_g`` in
the two-regime problem.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS_TORCH = ROOT / "models_torch"
MODELS_TORCH_TRAIN = ROOT / "models_torch_train"
for _path in (str(HERE), str(MODELS_TORCH_TRAIN), str(MODELS_TORCH)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from anchored_nets import (  # noqa: E402
    ROOT_BOUNDS_7,
    CleanMLP,
    InputNorm,
    WarmupCosine,
    build_control_net,
    rms,
    sample_box,
    trainable_parameters,
)
from params import PARAMS  # noqa: E402
from tf_to_torch_loader import DEFAULT_CKPT_DIR, build_root_torch_model  # noqa: E402


def dtype_from_name(name):
    return torch.float64 if name == "float64" else torch.float32


def set_module_grad(module, flag):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad_(flag)


class RootAnchor:
    """Surrogate root value at the line-integral anchor, same pseudo-state."""

    def __init__(self, dtype, device, ckpt_dir=DEFAULT_CKPT_DIR):
        self.model = build_root_torch_model(load_surrogate=True, ckpt_dir=ckpt_dir, dtype=dtype)
        self.model.v_nn.to(device=device, dtype=dtype)
        self.model.v_nn.eval()
        self.device = device
        self.dtype = dtype

    def __call__(self, lam3, logxi):
        n = lam3.shape[0]
        y = torch.full((n, 1), 0.5 * (PARAMS["Y_min"] + PARAMS["Y_max"]),
                       dtype=self.dtype, device=self.device)
        x0 = torch.cat(
            [
                torch.full_like(y, 0.5 * (PARAMS["logK_min"] + PARAMS["logK_max"])),
                torch.full_like(y, 0.5 * (PARAMS["Z_min"] + PARAMS["Z_max"])),
                y,
                lam3,
                torch.full_like(y, PARAMS["A_g_prime_prime"]),
                logxi,
                logxi,
            ],
            dim=1,
        )
        with torch.no_grad():
            return self.model.v_nn(x0).detach()


class XiFiLMRobustNet(nn.Module):
    """Amortized inner minimizer conditioned on xi with FiLM modulation."""

    def __init__(self, width=64, depth=3, seed=None, dtype=torch.float32, device=torch.device("cpu")):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.x_norm = InputNorm(ROOT_BOUNDS_7)
        self.in_dim = 7 + 3 + 3 + 3  # normed X + p + q + xi features
        self.layers = nn.ModuleList()
        self.films = nn.ModuleList()
        prev = self.in_dim
        for _ in range(depth):
            self.layers.append(nn.Linear(prev, width))
            self.films.append(nn.Linear(3, 2 * width))
            prev = width
        self.out = nn.Linear(prev, 3)
        self.to(device=device, dtype=dtype)

    def forward(self, x, p_vec, q_vec, logxi):
        invxi = torch.exp(-logxi).clamp(max=50.0)
        xi_features = torch.cat([logxi, invxi, torch.log1p(invxi)], dim=1)
        h = torch.cat([self.x_norm(x), p_vec, q_vec, xi_features], dim=1)
        for layer, film in zip(self.layers, self.films):
            h = layer(h)
            gamma, beta = torch.chunk(film(xi_features), 2, dim=1)
            h = h * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
            h = torch.tanh(h)
        return self.out(h)


class CostateRobustRoot:
    def __init__(self, args, dtype, device):
        self.args = args
        self.params = PARAMS.copy()
        self.dtype = dtype
        self.device = device

        hidden = tuple([args.width] * args.layers)
        self.costate_nn = CleanMLP(
            input_dim=7,
            bounds=ROOT_BOUNDS_7,
            hidden=hidden,
            activation=args.activation,
            final_activation=None,
            seed=args.seed + 1000,
            residual=not args.no_residual,
            output_dim=3,
        ).to(device=device, dtype=dtype)

        self.i_g_nn = build_control_net(
            7, ROOT_BOUNDS_7, self.params["θ_g"], args.seed + 2000,
            activation="tanh", init_rate=args.control_init_rate,
        ).to(device=device, dtype=dtype)
        self.i_d_nn = build_control_net(
            7, ROOT_BOUNDS_7, self.params["θ_d"], args.seed + 3000,
            activation="tanh", init_rate=args.control_init_rate,
        ).to(device=device, dtype=dtype)

        self.robust_nn = None
        if args.robust_mode != "closed":
            self.robust_nn = XiFiLMRobustNet(
                width=args.robust_width,
                depth=args.robust_depth,
                seed=args.seed + 4000,
                dtype=dtype,
                device=device,
            )

        self.anchor = RootAnchor(dtype, device, ckpt_dir=args.ckpt_dir)

        nodes, weights = np.polynomial.legendre.leggauss(args.quad_points)
        self.gl_t = torch.tensor(0.5 * (nodes + 1.0), dtype=dtype, device=device)
        self.gl_w = torch.tensor(0.5 * weights, dtype=dtype, device=device)

    def assemble_x(self, logK, Z, Y, lam3, logxi):
        return torch.cat(
            [
                logK,
                Z,
                Y,
                lam3,
                torch.full_like(Y, self.params["A_g_prime_prime"]),
                logxi,
                logxi,
            ],
            dim=1,
        )

    def costates(self, logK, Z, Y, lam3, logxi):
        x = self.assemble_x(logK, Z, Y, lam3, logxi)
        p = self.costate_nn(x)
        pK = 1.0 + self.args.p_logk_scale * torch.tanh(p[:, 0:1])
        pZ = self.args.p_other_scale * p[:, 1:2]
        pY = self.args.p_other_scale * p[:, 2:3]
        return pK, pZ, pY

    def recover_v(self, logK, Z, Y, lam3, logxi):
        n = logK.shape[0]
        x0K = torch.full((n, 1), 0.5 * (self.params["logK_min"] + self.params["logK_max"]),
                         dtype=self.dtype, device=self.device)
        x0Z = torch.full((n, 1), 0.5 * (self.params["Z_min"] + self.params["Z_max"]),
                         dtype=self.dtype, device=self.device)
        x0Y = torch.full((n, 1), 0.5 * (self.params["Y_min"] + self.params["Y_max"]),
                         dtype=self.dtype, device=self.device)
        dK = logK - x0K
        dZ = Z - x0Z
        dY = Y - x0Y
        acc = torch.zeros((n, 1), dtype=self.dtype, device=self.device)
        for t, w in zip(self.gl_t, self.gl_w):
            kt = x0K + t * dK
            zt = x0Z + t * dZ
            yt = x0Y + t * dY
            pK, pZ, pY = self.costates(kt, zt, yt, lam3, logxi)
            acc = acc + w * (pK * dK + pZ * dZ + pY * dY)
        return self.anchor(lam3, logxi) + acc

    def _grad(self, y, x):
        g = torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        return torch.zeros_like(x) if g is None else g.reshape(-1, 1)

    def robust_terms(self, logK, Z, Y, lam3, logxi, pK, pZ, pY, robust_input_detach=False):
        p = self.params
        xi = torch.exp(logxi)
        K = torch.exp(logK)
        dlogN = p["λ1"] + p["λ2"] * Y + lam3 * (Y - p["y_upper"])
        emissions = p["η"] * p["A_d"] * (1.0 - Z) * K

        qd = (pK - Z * pZ) * (1.0 - Z) * p["σ_d"]
        qg = (pK + (1.0 - Z) * pZ) * Z * p["σ_g"]
        qy = (pY - dlogN) * emissions * p["ϛ"]
        q_vec = torch.cat([qd, qg, qy], dim=1)
        p_vec = torch.cat([pK, pZ, pY], dim=1)
        h_closed = -q_vec / xi

        if self.args.robust_mode == "closed":
            h_hat = h_closed.detach() if robust_input_detach else h_closed
        else:
            x = self.assemble_x(logK, Z, Y, lam3, logxi)
            if robust_input_detach:
                x = x.detach()
                p_vec_in = p_vec.detach()
                q_vec_in = q_vec.detach()
                logxi_in = logxi.detach()
                h_closed_in = h_closed.detach()
            else:
                p_vec_in = p_vec
                q_vec_in = q_vec
                logxi_in = logxi
                h_closed_in = h_closed
            raw = self.robust_nn(x, p_vec_in, q_vec_in, logxi_in)
            if self.args.robust_mode == "residual":
                h_hat = h_closed_in + self.args.robust_residual_scale * torch.tanh(raw)
            elif self.args.robust_mode == "direct":
                h_hat = self.args.robust_direct_scale * torch.tanh(raw)
            else:
                raise ValueError("unknown robust_mode")

        inner = (q_vec * h_hat).sum(dim=1, keepdim=True) + 0.5 * xi * (h_hat * h_hat).sum(dim=1, keepdim=True)
        inner_closed = -0.5 / xi * (q_vec * q_vec).sum(dim=1, keepdim=True)
        gap = inner - inner_closed
        h_err = h_hat - h_closed
        return h_hat[:, 0:1], h_hat[:, 1:2], h_hat[:, 2:3], gap, h_err

    def robust_only_terms(self, sample, robust_input_detach=True):
        """Lightweight inner-minimizer loss path.

        The inner minimizer only needs first-order costates and the closed-form
        quadratic target.  Calling ``pde_terms`` here would also build Hessians
        and recover the value by quadrature, which is unnecessary and makes
        adversarial-style multi-step robust updates very expensive.
        """
        logK, Z, Y, logR, lam3, logxi = sample
        del logR
        logK = logK.detach()
        Z = Z.detach()
        Y = Y.detach()
        lam3 = lam3.detach()
        logxi = logxi.detach()
        pK, pZ, pY = self.costates(logK, Z, Y, lam3, logxi)
        hd, hg, hy, robust_gap, h_err = self.robust_terms(
            logK, Z, Y, lam3, logxi, pK, pZ, pY,
            robust_input_detach=robust_input_detach,
        )
        del hd, hg, hy
        return {
            "robust_gap": robust_gap,
            "h_err": h_err,
        }

    def pde_terms(self, sample, want_curl=False, robust_input_detach=False):
        logK, Z, Y, logR, lam3, logxi = sample
        logK = logK.detach().requires_grad_(True)
        Z = Z.detach().requires_grad_(True)
        Y = Y.detach().requires_grad_(True)
        lam3 = lam3.detach()
        logxi = logxi.detach()

        p = self.params
        A_d = p["A_d"]
        A_g_pp = p["A_g_prime_prime"]
        delta = p["δ"]

        pK, pZ, pY = self.costates(logK, Z, Y, lam3, logxi)
        pKK = self._grad(pK, logK)
        pKZ = self._grad(pK, Z)
        pZZ = self._grad(pZ, Z)
        pYY = self._grad(pY, Y)

        v = self.recover_v(logK, Z, Y, lam3, logxi)
        hd, hg, hy, robust_gap, h_err = self.robust_terms(
            logK, Z, Y, lam3, logxi, pK, pZ, pY,
            robust_input_detach=robust_input_detach,
        )

        x = self.assemble_x(logK, Z, Y, lam3, logxi)
        i_g = self.i_g_nn(x)
        i_d = self.i_d_nn(x)
        xi = torch.exp(logxi)
        K = torch.exp(logK)

        c = (A_d - i_d) * (1.0 - Z) + (A_g_pp - i_g) * Z
        inside_log = torch.clamp(c, min=1e-8).reshape(-1, 1)
        flow = delta * (torch.log(inside_log) + logK)
        pv = delta * v

        inside_log_i_d = torch.clamp(1.0 + p["θ_d"] * i_d, min=1e-8).reshape(-1, 1)
        inside_log_i_g = torch.clamp(1.0 + p["θ_g"] * i_g, min=1e-8).reshape(-1, 1)

        vKK_term = (p["σ_d"] ** 2 * (1 - Z) ** 2 + p["σ_g"] ** 2 * Z ** 2) / 2.0
        vK_term = (p["α_d"] + p["Γ_d"] * torch.log(inside_log_i_d)) * (1 - Z) \
            + (p["α_g"] + p["Γ_g"] * torch.log(inside_log_i_g)) * Z - vKK_term
        vZ_term = (p["α_g"] + p["Γ_g"] * torch.log(inside_log_i_g)
                   - (p["α_d"] + p["Γ_d"] * torch.log(inside_log_i_d))
                   - Z * p["σ_g"] ** 2 + (1 - Z) * p["σ_d"] ** 2) * Z * (1 - Z)
        vZZ_term = 0.5 * Z ** 2 * (1 - Z) ** 2 * (p["σ_g"] ** 2 + p["σ_d"] ** 2)
        vKZ_term = -Z * (1 - Z) ** 2 * p["σ_d"] ** 2 + Z ** 2 * (1 - Z) * p["σ_g"] ** 2

        emissions = p["η"] * p["A_d"] * (1.0 - Z) * K
        vY_term = (p["θ_bar"] + hy * p["ϛ"]) * emissions
        vYY_term = 0.5 * p["ϛ"] ** 2 * emissions ** 2
        dlogN = p["λ1"] + p["λ2"] * Y + lam3 * (Y - p["y_upper"])
        v_logN_term = dlogN * vY_term + (p["λ2"] + lam3) * vYY_term

        rhs = flow \
            + vK_term * pK + vKK_term * pKK \
            + vZ_term * pZ + vZZ_term * pZZ \
            + hd * (pK - Z * pZ) * (1.0 - Z) * p["σ_d"] \
            + hg * (pK + (1.0 - Z) * pZ) * Z * p["σ_g"] \
            + vKZ_term * pKZ \
            + pY * vY_term + vYY_term * pYY \
            + 0.5 * xi * (hd * hd + hg * hg + hy * hy) \
            - v_logN_term
        resid = rhs - pv

        mu_c = delta / inside_log
        FOC_d = -mu_c + p["Γ_d"] * p["θ_d"] / inside_log_i_d * (pK - Z * pZ)
        FOC_g = -mu_c + p["Γ_g"] * p["θ_g"] / inside_log_i_g * (pK + (1 - Z) * pZ)

        curl = torch.zeros_like(resid)
        if want_curl:
            pZK = self._grad(pZ, logK)
            pKY = self._grad(pK, Y)
            pYK = self._grad(pY, logK)
            pZY = self._grad(pZ, Y)
            pYZ = self._grad(pY, Z)
            curl = (pKZ - pZK) ** 2 + (pKY - pYK) ** 2 + (pZY - pYZ) ** 2

        return {
            "resid": resid,
            "FOC_d": FOC_d,
            "FOC_g": FOC_g,
            "dv_dY": pY,
            "v": v,
            "curl": curl,
            "robust_gap": robust_gap,
            "h_err": h_err,
            "c": c,
        }


def validation(model, n, batches, dtype, device, seed=12345, robust_input_detach=True):
    out = {
        "hjb": 0.0,
        "foc_d": 0.0,
        "foc_g": 0.0,
        "curl": 0.0,
        "robust_gap": 0.0,
        "h_err": 0.0,
        "level": 0.0,
        "c_min": 0.0,
    }
    for i in range(batches):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 1009 * i)
            sample = sample_box(model.params, n, dtype, device)
        terms = model.pde_terms(sample, want_curl=True, robust_input_detach=robust_input_detach)
        out["hjb"] += float(rms(terms["resid"]).detach().cpu())
        out["foc_d"] += float(rms(terms["FOC_d"]).detach().cpu())
        out["foc_g"] += float(rms(terms["FOC_g"]).detach().cpu())
        out["curl"] += float(rms(terms["curl"]).detach().cpu())
        out["robust_gap"] += float(torch.mean(terms["robust_gap"]).detach().cpu())
        out["h_err"] += float(rms(terms["h_err"]).detach().cpu())
        out["level"] += float(torch.mean(terms["v"]).detach().cpu())
        out["c_min"] += float(torch.min(terms["c"]).detach().cpu())
    for k in out:
        out[k] /= max(1, batches)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--valid_batch_size", type=int, default=512)
    ap.add_argument("--valid_batches", type=int, default=2)
    ap.add_argument("--valid_seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--lr_costate", type=float, default=4e-4)
    ap.add_argument("--lr_control", type=float, default=4e-4)
    ap.add_argument("--lr_robust", type=float, default=4e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.01)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--activation", default="tanh")
    ap.add_argument("--no_residual", action="store_true")
    ap.add_argument("--p_logk_scale", type=float, default=0.5)
    ap.add_argument("--p_other_scale", type=float, default=0.2)
    ap.add_argument("--quad_points", type=int, default=8)
    ap.add_argument("--robust_mode", choices=["closed", "residual", "direct"], default="residual")
    ap.add_argument("--robust_width", type=int, default=64)
    ap.add_argument("--robust_depth", type=int, default=3)
    ap.add_argument("--robust_residual_scale", type=float, default=0.05)
    ap.add_argument("--robust_direct_scale", type=float, default=2.0)
    ap.add_argument("--robust_steps", type=int, default=1)
    ap.add_argument("--control_steps", type=int, default=1)
    ap.add_argument("--lambda_curl", type=float, default=1.0)
    ap.add_argument("--duality_weight", type=float, default=1.0)
    ap.add_argument("--robust_gap_weight", type=float, default=0.0)
    ap.add_argument("--no_outer_envelope_detach", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--control_init_rate", type=float, default=0.02)
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--export", default=None)
    args = ap.parse_args()

    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = CostateRobustRoot(args, dtype, device)
    model.costate_nn.train()
    model.i_g_nn.train()
    model.i_d_nn.train()
    if model.robust_nn is not None:
        model.robust_nn.train()

    opt_cst = torch.optim.Adam(trainable_parameters([model.costate_nn]), lr=args.lr_costate)
    opt_ctl = torch.optim.Adam(trainable_parameters([model.i_g_nn, model.i_d_nn]), lr=args.lr_control)
    opt_rob = None
    if model.robust_nn is not None:
        opt_rob = torch.optim.Adam(trainable_parameters([model.robust_nn]), lr=args.lr_robust)

    warmup = max(1, int(args.steps * args.warmup_frac))
    sch_cst = WarmupCosine(args.lr_costate, args.steps, warmup, args.min_lr)
    sch_ctl = WarmupCosine(args.lr_control, args.steps, warmup, args.min_lr)
    sch_rob = WarmupCosine(args.lr_robust, args.steps, warmup, args.min_lr)

    print(
        f"[setup] robust_mode={args.robust_mode} dtype={args.dtype} device={device} "
        f"steps={args.steps} bs={args.batch_size} robust_steps={args.robust_steps}",
        flush=True,
    )

    history = []
    outer_envelope_detach = not args.no_outer_envelope_detach
    val0 = validation(
        model, args.valid_batch_size, args.valid_batches, dtype, device, args.valid_seed,
        robust_input_detach=outer_envelope_detach,
    )
    print(
        f"[step      0] hjb={val0['hjb']:.5e} foc=({val0['foc_d']:.2e},{val0['foc_g']:.2e}) "
        f"curl={val0['curl']:.2e} gap={val0['robust_gap']:.2e} h_err={val0['h_err']:.2e}",
        flush=True,
    )
    history.append({"step": 0, **val0})
    t0 = time.time()

    for step in range(1, args.steps + 1):
        for group in opt_cst.param_groups:
            group["lr"] = sch_cst(step)
        for group in opt_ctl.param_groups:
            group["lr"] = sch_ctl(step)
        if opt_rob is not None:
            for group in opt_rob.param_groups:
                group["lr"] = sch_rob(step)

        if opt_rob is not None:
            set_module_grad(model.costate_nn, False)
            set_module_grad(model.i_g_nn, False)
            set_module_grad(model.i_d_nn, False)
            set_module_grad(model.robust_nn, True)
            for _ in range(max(1, args.robust_steps)):
                sample = sample_box(model.params, args.batch_size, dtype, device)
                terms = model.robust_only_terms(sample, robust_input_detach=True)
                robust_loss = torch.mean(terms["robust_gap"]) + args.duality_weight * rms(terms["h_err"])
                opt_rob.zero_grad(set_to_none=True)
                robust_loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters([model.robust_nn]), args.grad_clip)
                opt_rob.step()

        set_module_grad(model.costate_nn, True)
        set_module_grad(model.i_g_nn, False)
        set_module_grad(model.i_d_nn, False)
        set_module_grad(model.robust_nn, False)
        sample = sample_box(model.params, args.batch_size, dtype, device)
        terms = model.pde_terms(sample, want_curl=True, robust_input_detach=outer_envelope_detach)
        costate_loss = rms(terms["resid"]) + args.lambda_curl * rms(terms["curl"])
        if args.robust_gap_weight > 0:
            costate_loss = costate_loss + args.robust_gap_weight * torch.mean(terms["robust_gap"])
        opt_cst.zero_grad(set_to_none=True)
        costate_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters([model.costate_nn]), args.grad_clip)
        opt_cst.step()

        set_module_grad(model.costate_nn, False)
        set_module_grad(model.i_g_nn, True)
        set_module_grad(model.i_d_nn, True)
        set_module_grad(model.robust_nn, False)
        for _ in range(max(1, args.control_steps)):
            sample = sample_box(model.params, args.batch_size, dtype, device)
            terms = model.pde_terms(sample, want_curl=False, robust_input_detach=outer_envelope_detach)
            Y = sample[2]
            mask = ((Y > model.params["y_upper"]).to(dtype)) * ((terms["dv_dY"] > 0).to(dtype))
            control_loss = rms(terms["FOC_d"]) + rms(terms["FOC_g"]) + rms(terms["dv_dY"] * mask)
            opt_ctl.zero_grad(set_to_none=True)
            control_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters([model.i_g_nn, model.i_d_nn]), args.grad_clip)
            opt_ctl.step()

        if step % args.log_every == 0 or step == args.steps:
            val = validation(
                model, args.valid_batch_size, args.valid_batches, dtype, device, args.valid_seed,
                robust_input_detach=outer_envelope_detach,
            )
            dt = time.time() - t0
            print(
                f"[step {step:6d}] hjb={val['hjb']:.5e} foc=({val['foc_d']:.2e},{val['foc_g']:.2e}) "
                f"curl={val['curl']:.2e} gap={val['robust_gap']:.2e} h_err={val['h_err']:.2e} ({dt:.0f}s)",
                flush=True,
            )
            history.append({"step": step, **val})

    final = history[-1]
    if args.export:
        os.makedirs(args.export, exist_ok=True)
        torch.save(model.costate_nn.state_dict(), os.path.join(args.export, "costate_nn.pt"))
        torch.save(model.i_g_nn.state_dict(), os.path.join(args.export, "i_g_nn.pt"))
        torch.save(model.i_d_nn.state_dict(), os.path.join(args.export, "i_d_nn.pt"))
        if model.robust_nn is not None:
            torch.save(model.robust_nn.state_dict(), os.path.join(args.export, "robust_nn.pt"))
        with open(os.path.join(args.export, "history.jsonl"), "w", encoding="utf-8") as f:
            for row in history:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"[export] wrote {args.export}", flush=True)

    result = {
        "mode": "costate_robust_root",
        "robust_mode": args.robust_mode,
        "steps": args.steps,
        "hjb_final": final["hjb"],
        "foc_d_final": final["foc_d"],
        "foc_g_final": final["foc_g"],
        "curl_final": final["curl"],
        "robust_gap_final": final["robust_gap"],
        "h_err_final": final["h_err"],
        "level_final": final["level"],
    }
    print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
