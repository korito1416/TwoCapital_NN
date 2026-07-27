"""
occam_trial.py -- Is the costate/EGM machinery necessary, or does a RE-CENTERED
value net pin the level just as hard, far simpler?

Analytical result: the level freedom is exactly ONE additive constant per slice.
=> The minimal way to remove it: take a plain value net phi and use

        v_rec(s) = phi(s) - phi(x0; lam3,logxi) + v0

  * v_rec(x0) = v0 BY CONSTRUCTION (for any phi)         -> level pinned, no drift
  * v_rec is INVARIANT to phi's additive constant        -> weak-id direction quotiented out
  * grad v_rec = grad phi (the subtracted term is const in logK,Z,Y) -> conservative
    BY CONSTRUCTION; NO curl penalty, NO 8-pt line integral.

Same anchor convention as costate_trial.py (v0 = surrogate transformed-v at x0,
phi(x0) uses the eval point's pseudo-states lam3,logxi) so the comparison is fair.

Variants (CLEAN arch: input width 5 [logK,Z,Y,lam3,logxi], LINEAR value output):
  recentered : v_rec = phi - phi(x0) + v0   (hard level pin by construction)
  plain      : v     = phi                  (no pin; clean-arch weak-id control)

Protocol matches costate_trial.py / method_trial.py: from scratch, 3 seeds, N steps,
batch 128, Adam, WarmupCosine 4e-4, grad-clip 1.0; 256-pt probe (gen seed 12345);
LEVEL = mean over probe of v_rec; level_spread = std over seeds. Prints RESULT {dict}.
Compare recentered's level_spread to costate_egm's 0.0077.
"""
import argparse, os, sys, time
import numpy as np, torch

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
HERE = os.path.join(ROOT, "models_torch_train"); MT = os.path.join(ROOT, "models_torch")
for _p in (HERE, MT):
    if _p not in sys.path: sys.path.insert(0, _p)
from params import PARAMS, investment_rate_activation  # noqa
from tf_to_torch_loader import build_root_torch_model, build_trainable_net, make_root_configs  # noqa
from train_root import sample_box, WarmupCosine  # noqa

DELTA = float(PARAMS["δ"]); NPROBE = 256; CLEAN_DIM = 5


def anchor(params, dtype):
    return (torch.tensor([[0.5 * (params["logK_min"] + params["logK_max"])]], dtype=dtype),
            torch.tensor([[0.5 * (params["Z_min"] + params["Z_max"])]], dtype=dtype),
            torch.tensor([[0.5 * (params["Y_min"] + params["Y_max"])]], dtype=dtype))


def surrogate_v0(params, dtype, x0):
    sur = build_root_torch_model(load_surrogate=True, dtype=dtype); sur.v_nn.eval()
    lam3 = torch.tensor([[0.5 * (params["λ3_min"] + params["λ3_max"])]], dtype=dtype)
    logxi = torch.tensor([[0.5 * (params["logξ_min"] + params["logξ_max"])]], dtype=dtype)
    X = torch.cat([x0[0], x0[1], x0[2], lam3,
                   params["A_g_prime_prime"] * torch.ones_like(x0[2]), logxi, logxi], 1)
    with torch.no_grad():
        return sur.v_nn(X).reshape(()).detach()


class ValueModel:
    def __init__(self, params, dtype, seed, recenter):
        self.params = params; self.dtype = dtype; self.recenter = recenter
        v_cfg, ig_cfg, id_cfg = make_root_configs()
        vc = dict(v_cfg); vc["final_activation"] = "linear"; vc["nn_name"] = "phi"
        self.phi = build_trainable_net(vc, input_dim=CLEAN_DIM, seed=seed)
        ig = dict(ig_cfg); ig["final_activation"] = investment_rate_activation(params["θ_g"])
        idd = dict(id_cfg); idd["final_activation"] = investment_rate_activation(params["θ_d"])
        self.i_g_nn = build_trainable_net(ig, CLEAN_DIM, None if seed is None else seed + 101)
        self.i_d_nn = build_trainable_net(idd, CLEAN_DIM, None if seed is None else seed + 202)
        for n in (self.phi, self.i_g_nn, self.i_d_nn):
            n.to(dtype=dtype); n.train()
        self.x0 = anchor(params, dtype)
        self.v0 = surrogate_v0(params, dtype, self.x0)

    def _X(self, logK, Z, Y, lam3, logxi):
        return torch.cat([logK, Z, Y, lam3, logxi], 1)

    def _phi_x0(self, lam3, logxi):
        """phi at the anchor true-state x0, with the eval point's pseudo-states."""
        N = lam3.shape[0]
        X0 = torch.cat([self.x0[0].expand(N, 1), self.x0[1].expand(N, 1),
                        self.x0[2].expand(N, 1), lam3, logxi], 1)
        return self.phi(X0)

    def value(self, phi_s, lam3, logxi):
        if not self.recenter:
            return phi_s
        return phi_s - self._phi_x0(lam3, logxi) + self.v0

    def pde_terms(self, logK, Z, Y, logR, lam3, logxi):
        """Root-regime HJB residual + FOCs (faithful to PostDamagePostTech.py),
        derivatives = autograd of phi, level term = v_rec."""
        p = self.params
        A_d = p['A_d']; A_g_pp = p['A_g_prime_prime']; δ = p['δ']
        α_d = p['α_d']; Γ_d = p['Γ_d']; θ_d = p['θ_d']; σ_d = p['σ_d']
        α_g = p['α_g']; Γ_g = p['Γ_g']; θ_g = p['θ_g']; σ_g = p['σ_g']
        θ_bar = p['θ_bar']; η = p['η']; ϛ = p['ϛ']
        λ1 = p['λ1']; λ2 = p['λ2']; y_upper = p['y_upper']

        logK = logK.detach().requires_grad_(True)
        Z = Z.detach().requires_grad_(True)
        Y = Y.detach().requires_grad_(True)

        X = self._X(logK, Z, Y, lam3, logxi)
        phi_s = self.phi(X)
        v = self.value(phi_s, lam3, logxi)  # level term uses v_rec

        def d(out, x):
            g = torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out),
                                    create_graph=True, retain_graph=True, allow_unused=True)[0]
            return torch.zeros_like(x) if g is None else g.reshape(-1, 1)

        # grad v_rec = grad phi_s  (subtracted phi(x0) is const in logK,Z,Y)
        dv_dlogK = d(phi_s, logK); dv_dZ = d(phi_s, Z); dv_dY = d(phi_s, Y)
        d2v_dlogK2 = d(dv_dlogK, logK); d2v_dlogKdZ = d(dv_dlogK, Z)
        d2v_dZ2 = d(dv_dZ, Z); d2v_dY2 = d(dv_dY, Y)

        i_g = self.i_g_nn(X); i_d = self.i_d_nn(X)
        ξ = torch.exp(logxi); K = torch.exp(logK)

        h_d = -1.0 / ξ * ((dv_dlogK - Z * dv_dZ) * (1 - Z) * σ_d)
        h_g = -1.0 / ξ * ((dv_dlogK + (1 - Z) * dv_dZ) * Z * σ_g)
        h_y = -1.0 / ξ * (dv_dY - (λ1 + λ2 * Y + lam3 * (Y - y_upper))) * η * A_d * (1 - Z) * K * ϛ

        pv = δ * v
        c = (A_d - i_d) * (1 - Z) + (A_g_pp - i_g) * Z
        inside_log = torch.clamp(c, min=1e-8).reshape(-1, 1)
        flow = δ * (torch.log(inside_log) + logK)

        v_logKlogK_term = (σ_d ** 2 * (1 - Z) ** 2 + σ_g ** 2 * Z ** 2) / 2.0
        ild = torch.clamp(1.0 + θ_d * i_d, min=1e-8).reshape(-1, 1)
        ilg = torch.clamp(1.0 + θ_g * i_g, min=1e-8).reshape(-1, 1)
        v_logK_term = (α_d + Γ_d * torch.log(ild)) * (1 - Z) + (α_g + Γ_g * torch.log(ilg)) * Z - v_logKlogK_term
        v_Z_term = (α_g + Γ_g * torch.log(ilg) - (α_d + Γ_d * torch.log(ild))
                    - Z * σ_g ** 2 + (1 - Z) * σ_d ** 2) * Z * (1 - Z)
        v_ZZ_term = 0.5 * Z ** 2 * (1 - Z) ** 2 * (σ_g ** 2 + σ_d ** 2)
        v_logK_Z_term = -Z * (1 - Z) ** 2 * σ_d ** 2 + Z ** 2 * (1.0 - Z) * σ_g ** 2
        v_y_term = (θ_bar + h_y * ϛ) * η * A_d * (1 - Z) * K
        v_yy_term = 0.5 * ϛ ** 2 * (η * A_d * (1 - Z) * K) ** 2
        v_logN_term = (λ1 + λ2 * Y + lam3 * (Y - y_upper)) * v_y_term + (λ2 + lam3) * v_yy_term

        rhs = flow + v_logK_term * dv_dlogK + v_logKlogK_term * d2v_dlogK2 \
            + v_Z_term * dv_dZ + v_ZZ_term * d2v_dZ2 \
            + h_d * (dv_dlogK - Z * dv_dZ) * (1 - Z) * σ_d \
            + h_g * (dv_dlogK + (1 - Z) * dv_dZ) * Z * σ_g \
            + v_logK_Z_term * d2v_dlogKdZ + dv_dY * v_y_term + v_yy_term * d2v_dY2 \
            + 0.5 * ξ * (h_d ** 2 + h_g ** 2 + h_y ** 2) - v_logN_term
        resid = rhs - pv

        mu_c = δ / inside_log
        FOC_d = -mu_c + Γ_d * θ_d / ild * (dv_dlogK - Z * dv_dZ)
        FOC_g = -mu_c + Γ_g * θ_g / ilg * (dv_dlogK + (1.0 - Z) * dv_dZ)
        return resid, FOC_d, FOC_g, dv_dY, v


def fixed_probe(dtype):
    g = torch.Generator().manual_seed(12345)
    def col(lo, hi): return (lo + (hi - lo) * torch.rand((NPROBE, 1), generator=g)).to(dtype)
    return [col(4, 7), col(0.01, 0.99), col(0, 4), col(1, 6), col(0, 1 / 3.), col(-3, 5)]


def _rms(x): return torch.sqrt(torch.mean(x ** 2))


def evaluate(model, probe):
    resid, FOC_d, FOC_g, dv_dY, v = model.pde_terms(*probe)
    return (float(v.mean().detach()), float(_rms(resid).detach()),
            float(_rms(FOC_d).detach()), float(_rms(FOC_g).detach()))


def train_one_seed(seed, steps, dtype, lr, recenter, device=torch.device("cpu")):
    torch.manual_seed(seed); np.random.seed(seed)
    m = ValueModel(PARAMS, dtype, seed, recenter); params = PARAMS
    opt_v = torch.optim.Adam(m.phi.parameters(), lr=lr)
    opt_c = torch.optim.Adam(list(m.i_g_nn.parameters()) + list(m.i_d_nn.parameters()), lr=lr)
    sch = WarmupCosine(lr, steps, max(1, steps // 100), 1e-5); CLIP = 1.0
    for step in range(1, steps + 1):
        for g in opt_v.param_groups: g["lr"] = sch(step)
        for g in opt_c.param_groups: g["lr"] = sch(step)
        s = sample_box(params, 128, dtype, device)
        resid, _, _, _, _ = m.pde_terms(*s)
        vloss = _rms(resid)
        opt_v.zero_grad(set_to_none=True); vloss.backward()
        torch.nn.utils.clip_grad_norm_(m.phi.parameters(), CLIP); opt_v.step()
        s2 = sample_box(params, 128, dtype, device)
        _, FOC_d, FOC_g, dv_dY, _ = m.pde_terms(*s2)
        mask = ((s2[2] > params["y_upper"]).to(dtype)) * ((dv_dY > 0).to(dtype))
        closs = _rms(FOC_d) + _rms(FOC_g) + _rms(dv_dY * mask)
        opt_c.zero_grad(set_to_none=True); closs.backward()
        torch.nn.utils.clip_grad_norm_(
            list(m.i_g_nn.parameters()) + list(m.i_d_nn.parameters()), CLIP); opt_c.step()
    for n in (m.phi, m.i_g_nn, m.i_d_nn): n.eval()
    return m


def run(variant, steps, seeds, dtype, lr):
    recenter = (variant == "recentered")
    probe = fixed_probe(dtype); levels, lvs, fds, fgs = [], [], [], []
    for sd in seeds:
        t0 = time.time()
        m = train_one_seed(sd, steps, dtype, lr, recenter)
        lvl, lv, fd, fg = evaluate(m, probe)
        levels.append(lvl); lvs.append(lv); fds.append(fd); fgs.append(fg)
        print(f"  [{variant}] seed={sd} level={lvl:+.4f} loss_v={lv:.3e} "
              f"FOC_d={fd:.3e} FOC_g={fg:.3e} ({time.time()-t0:.0f}s)", flush=True)
    out = {"method": "occam_" + variant, "steps": steps, "seeds": list(seeds),
           "dtype": str(dtype).replace("torch.", ""),
           "level_mean": float(np.mean(levels)), "level_spread": float(np.std(levels)),
           "loss_v": float(np.mean(lvs)), "FOC_d": float(np.mean(fds)),
           "FOC_g": float(np.mean(fgs)), "levels": [round(x, 4) for x in levels]}
    print("RESULT " + repr(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["recentered", "plain"], default="recentered")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = ap.parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    print(f"[occam] variant={args.variant} steps={args.steps} seeds={args.seeds}", flush=True)
    run(args.variant, args.steps, args.seeds, dtype, args.lr)


if __name__ == "__main__":
    main()
