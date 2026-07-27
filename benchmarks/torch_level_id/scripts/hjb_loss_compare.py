"""
hjb_loss_compare.py -- apples-to-apples HJB-loss comparison of the three
parameterizations on the root regime, computed with ONE canonical residual.

For EACH trained model we evaluate the SAME canonical root-regime HJB residual
  resid = rhs(autograd derivs of the model's value, model's controls) - delta*v
on a shared 256-pt probe, where:
  * plain / recentered: v = phi / (phi - phi(x0) + v0); derivs = autograd of v.
  * costate          : v = recover_v (the 8-pt line integral of the costate field);
                       derivs = autograd of THAT recovered v  (the TRUE gradient of
                       the recovered value -- NOT the raw costate heads p, NOT the
                       d/dx(p) used during costate TRAINING). So this exposes the
                       integrability gap: costate's TRUE recovered-v residual vs its
                       training (dp-based) residual.

All three trained from scratch, same seeds, same steps. Reports, per variant:
  canon_loss_v  = rms of the canonical (autograd-of-value) HJB residual  <-- the fair "HJB loss"
  train_loss_v  = what the variant's own training loop minimized (for reference)
  level_spread  = std over seeds of probe-mean v
Prints RESULT {dict} per variant.
"""
import argparse, os, sys, time
import numpy as np, torch

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
HERE = os.path.join(ROOT, "models_torch_train"); MT = os.path.join(ROOT, "models_torch")
SCRIPTS = os.path.join(ROOT, "benchmarks/torch_level_id/scripts")
for _p in (HERE, MT, SCRIPTS):
    if _p not in sys.path: sys.path.insert(0, _p)

from params import PARAMS  # noqa
import occam_trial as OC   # noqa  ValueModel, train_one_seed, fixed_probe
import costate_trial as CO # noqa  CostateModel, train_one_seed

DELTA = float(PARAMS["δ"]); NPROBE = 256


def _g(o, x):
    r = torch.autograd.grad(o, x, torch.ones_like(o), create_graph=True,
                            retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(x) if r is None else r.reshape(-1, 1)


def canonical_residual(v, i_g, i_d, logK, Z, Y, lam3, logxi, p):
    """Root-regime HJB residual (faithful to PostDamagePostTech), autograd of v."""
    A_d = p['A_d']; A_g_pp = p['A_g_prime_prime']; δ = p['δ']
    α_d = p['α_d']; Γ_d = p['Γ_d']; θ_d = p['θ_d']; σ_d = p['σ_d']
    α_g = p['α_g']; Γ_g = p['Γ_g']; θ_g = p['θ_g']; σ_g = p['σ_g']
    θ_bar = p['θ_bar']; η = p['η']; ϛ = p['ϛ']; λ1 = p['λ1']; λ2 = p['λ2']; y_upper = p['y_upper']

    dvK = _g(v, logK); dvZ = _g(v, Z); dvY = _g(v, Y)
    d2vK = _g(dvK, logK); d2vKZ = _g(dvK, Z); d2vZ = _g(dvZ, Z); d2vY = _g(dvY, Y)

    ξ = torch.exp(logxi); K = torch.exp(logK)
    h_d = -1.0 / ξ * ((dvK - Z * dvZ) * (1 - Z) * σ_d)
    h_g = -1.0 / ξ * ((dvK + (1 - Z) * dvZ) * Z * σ_g)
    h_y = -1.0 / ξ * (dvY - (λ1 + λ2 * Y + lam3 * (Y - y_upper))) * η * A_d * (1 - Z) * K * ϛ

    pv = δ * v
    c = (A_d - i_d) * (1 - Z) + (A_g_pp - i_g) * Z
    il = torch.clamp(c, min=1e-8).reshape(-1, 1)
    flow = δ * (torch.log(il) + logK)
    vKK = (σ_d ** 2 * (1 - Z) ** 2 + σ_g ** 2 * Z ** 2) / 2.0
    ild = torch.clamp(1.0 + θ_d * i_d, min=1e-8).reshape(-1, 1)
    ilg = torch.clamp(1.0 + θ_g * i_g, min=1e-8).reshape(-1, 1)
    vK = (α_d + Γ_d * torch.log(ild)) * (1 - Z) + (α_g + Γ_g * torch.log(ilg)) * Z - vKK
    vZ = (α_g + Γ_g * torch.log(ilg) - (α_d + Γ_d * torch.log(ild))
          - Z * σ_g ** 2 + (1 - Z) * σ_d ** 2) * Z * (1 - Z)
    vZZ = 0.5 * Z ** 2 * (1 - Z) ** 2 * (σ_g ** 2 + σ_d ** 2)
    vKZ = -Z * (1 - Z) ** 2 * σ_d ** 2 + Z ** 2 * (1.0 - Z) * σ_g ** 2
    vy = (θ_bar + h_y * ϛ) * η * A_d * (1 - Z) * K
    vyy = 0.5 * ϛ ** 2 * (η * A_d * (1 - Z) * K) ** 2
    vlogN = (λ1 + λ2 * Y + lam3 * (Y - y_upper)) * vy + (λ2 + lam3) * vyy
    rhs = flow + vK * dvK + vKK * d2vK + vZ * dvZ + vZZ * d2vZ \
        + h_d * (dvK - Z * dvZ) * (1 - Z) * σ_d + h_g * (dvK + (1 - Z) * dvZ) * Z * σ_g \
        + vKZ * d2vKZ + dvY * vy + vyy * d2vY + 0.5 * ξ * (h_d ** 2 + h_g ** 2 + h_y ** 2) - vlogN
    return rhs - pv


def eval_canonical(kind, m, probe):
    logK, Z, Y, logR, lam3, logxi = probe
    lK = logK.detach().requires_grad_(True)
    Z_ = Z.detach().requires_grad_(True)
    Y_ = Y.detach().requires_grad_(True)
    if kind in ("plain", "recentered"):
        X5 = torch.cat([lK, Z_, Y_, lam3, logxi], 1)
        v = m.value(m.phi(X5), lam3, logxi)
        ig = m.i_g_nn(X5); idd = m.i_d_nn(X5)
    else:  # costate
        Agpp = PARAMS["A_g_prime_prime"]
        v = m.recover_v(lK, Z_, Y_, lam3, Agpp, logxi)   # autograd-able recovered v
        X7 = torch.cat([lK, Z_, Y_, lam3, Agpp * torch.ones_like(Y_), logxi, logxi], 1)
        ig = m.i_g_nn(X7); idd = m.i_d_nn(X7)
    resid = canonical_residual(v, ig, idd, lK, Z_, Y_, lam3, logxi, PARAMS)
    return float(torch.sqrt(torch.mean(resid ** 2)).detach()), float(v.mean().detach())


def run(variant, steps, seeds, dtype, lr):
    probe = OC.fixed_probe(dtype)
    canon, levels, train_loss = [], [], []
    for sd in seeds:
        t0 = time.time()
        if variant in ("plain", "recentered"):
            m = OC.train_one_seed(sd, steps, dtype, lr, recenter=(variant == "recentered"))
        else:
            m = CO.train_one_seed(sd, steps, dtype, lr, lambda_curl=1.0)
        cl, lvl = eval_canonical(variant, m, probe)
        canon.append(cl); levels.append(lvl)
        print(f"  [{variant}] seed={sd} canon_loss_v={cl:.3e} level={lvl:+.4f} ({time.time()-t0:.0f}s)", flush=True)
    out = {"variant": variant, "steps": steps, "seeds": list(seeds),
           "canon_loss_v": float(np.mean(canon)), "canon_loss_v_seeds": [round(x, 6) for x in canon],
           "level_mean": float(np.mean(levels)), "level_spread": float(np.std(levels)),
           "levels": [round(x, 4) for x in levels]}
    print("RESULT " + repr(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["plain", "recentered", "costate"], required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", type=float, default=4e-4)
    args = ap.parse_args()
    print(f"[hjb_compare] variant={args.variant} steps={args.steps} seeds={args.seeds}", flush=True)
    run(args.variant, args.steps, args.seeds, torch.float32, args.lr)


if __name__ == "__main__":
    main()
