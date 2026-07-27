"""
gan_costate_attack_3d.py -- Phase-2 prototype of the CHOSEN adversarial formulation (c):
ADVERSARIAL FOC-CONSISTENCY COSTATE-ATTACK on the 3-D post-damage-post-tech robust climate HJB,
graded against the stabilized FD ground truth (TRUE costate vZ) via stable_fd_eval.grade.

WHY (c) and not (a)/(b):
  - The robust inner min over (h,g) is CLOSED FORM (h*=-(1/xi)sigma'V_x, g*=exp(...)), so a learned
    adversary for the worst case is REDUNDANT and inherits the under-identified vZ.
  - A residual-RAR adversary (b1) re-weights the very signal that is FLAT in vZ where a_Z->0
    (the de-invest corner) -> cannot see the under-identification.
  - (c) attacks a NON-residual discrepancy. With SEPARATE control nets i_d,i_g (learned from the
    Hamiltonian) the two FOCs
        delta/(C/K) = phi_d'(i_d)[v_logK - Z v_Z]      (FOC-d)
        delta/(C/K) = phi_g'(i_g)[v_logK + (1-Z) v_Z]  (FOC-g)
    OVER-DETERMINE (i_d,i_g,v_logK,v_Z): eliminating delta/(C/K) gives
        D(x) = phi_d'(i_d)[v_logK - Z v_Z] - phi_g'(i_g)[v_logK + (1-Z) v_Z] = 0
    which PINS v_Z from (i_d,i_g,v_logK) WITHOUT dividing by the Z-drift a_Z -> exactly the channel
    the strong residual lacks in the corner. The adversary HARD-MINES the points where D^2 is largest
    (esp. the de-invest corner) and forces the solver to fix vZ there. Label-free analogue of the
    supervision that broke the ~0.81 di_err_vZ floor.

TWO ARMS, matched compute (same nets, same #steps, same batch budget, same optimizer):
  ARM uniform : value+control nets trained on strong-HJB-residual + FOC-consistency, UNIFORM sampling.
  ARM adv     : identical, but a fraction of each batch is ADVERSARIALLY hard-mined on D^2 (the
                costate-attack); the FOC-consistency term is up-weighted on those hard points.

Grading (the ONLY thing that decides a win) is stable_fd_eval.grade on i_d,i_g,vZ vs the stable FD:
  box_err_i_d/i_g, di_err_vZ (DECISIVE costate error), di_min_i_d / di_frac_neg (de-invest capture).
NOT on the residual/loss (softplus lesson).

Physics is byte-faithful to fd_pdpt_v5 (same P, _drift, controls FOC, damage horizon, QFLOOR feasibility).
Robustness drag at xi=148.4 is O(sigma^2/xi)~1e-6 and enters via the exact closed-form h (kept, negligible).
"""
import os, sys, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch
torch.set_num_threads(4)
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PDPT = os.path.join(HERE, "..", "post_damage_post_tech")
sys.path.insert(0, PDPT)
from stable_fd_eval import grade, load_stable_fd  # noqa: E402
import fd_pdpt_v5 as FD  # noqa: E402

P = FD.P
QFLOOR = FD.QFLOOR
Y_CAP = FD.Y_CAP
LK_CAP = FD.LK_CAP

# grids / box (match the FD + grader)
LK_LO, LK_HI = 4.0, 7.0
Z_LO, Z_HI = 0.02, 0.98
Y_LO, Y_HI = 0.0, 4.0
XI = 148.4
LAM3 = 1.0 / 6.0

dev = torch.device("cpu")
DT = torch.float32


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def mlp(nin, width, depth, act):
    layers = [nn.Linear(nin, width), act()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), act()]
    layers += [nn.Linear(width, 1)]
    return nn.Sequential(*layers)


class ValueNet(nn.Module):
    """V(logK,Z,Y), swish, 4x32. Output is unconstrained (v can be negative)."""
    def __init__(self, width=32, depth=4):
        super().__init__()
        self.net = mlp(3, width, depth, Swish)

    def feat(self, logK, Z, Y):
        sK = 2 * (logK - LK_LO) / (LK_HI - LK_LO) - 1
        sZ = 2 * (Z - Z_LO) / (Z_HI - Z_LO) - 1
        sY = 2 * (Y - Y_LO) / (Y_HI - Y_LO) - 1
        return torch.cat([sK, sZ, sY], dim=1)

    def forward(self, logK, Z, Y):
        return self.net(self.feat(logK, Z, Y))


class CtrlNet(nn.Module):
    """SEPARATE control net for one investment rate. Output in a physically sane band via tanh:
    i in [-1/theta + eps, i_hi]; de-investment (i<0) MUST be representable (FD min ~ -0.025).
    Lower feasibility corner is -1/theta = -0.0599; we allow down to -0.058."""
    def __init__(self, theta, i_hi=0.20, width=32, depth=4):
        super().__init__()
        self.net = mlp(3, width, depth, nn.Tanh)
        self.lo = -1.0 / theta + 0.002   # feasibility floor 1+theta*i>0
        self.hi = i_hi

    def feat(self, logK, Z, Y):
        sK = 2 * (logK - LK_LO) / (LK_HI - LK_LO) - 1
        sZ = 2 * (Z - Z_LO) / (Z_HI - Z_LO) - 1
        sY = 2 * (Y - Y_LO) / (Y_HI - Y_LO) - 1
        return torch.cat([sK, sZ, sY], dim=1)

    def forward(self, logK, Z, Y):
        u = torch.tanh(self.net(self.feat(logK, Z, Y)))
        return self.lo + 0.5 * (u + 1.0) * (self.hi - self.lo)


def _g(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]


# ---------------------------------------------------------------------------
# HJB residual + FOC-consistency for the 3-D regime (torch, autodiff).
# Mirrors fd_pdpt_v5._residual / controls / _drift exactly.
# ---------------------------------------------------------------------------
def physics(vnet, idnet, ignet, logK, Z, Y):
    p = P
    delta = p["delta"]; A_d = p["A_d"]; A_gpp = p["A_gpp"]
    a_d = p["a_d"]; G_d = p["G_d"]; t_d = p["t_d"]; s_d = p["s_d"]
    a_g = p["a_g"]; G_g = p["G_g"]; t_g = p["t_g"]; s_g = p["s_g"]
    thbar = p["thbar"]; eta = p["eta"]; vars_ = p["vars"]
    l1 = p["l1"]; l2 = p["l2"]; y_up = p["y_up"]

    v = vnet(logK, Z, Y)
    vlK = _g(v, logK); vZ = _g(v, Z); vY = _g(v, Y)
    vKK = _g(vlK, logK); vZZ = _g(vZ, Z); vYY = _g(vY, Y)

    i_d = idnet(logK, Z, Y)
    i_g = ignet(logK, Z, Y)

    K = torch.exp(torch.clamp(logK, max=LK_CAP))
    E = eta * A_d * (1 - Z) * K

    phid = a_d + G_d * torch.log(torch.clamp(1 + t_d * i_d, min=1e-9))
    phig = a_g + G_g * torch.log(torch.clamp(1 + t_g * i_g, min=1e-9))
    Dc = s_d ** 2 * (1 - Z) ** 2 + s_g ** 2 * Z ** 2
    a_lK = (1 - Z) * phid + Z * phig - Dc / 2.0
    a_Z = Z * (1 - Z) * (phig - phid + (1 - Z) * s_d ** 2 - Z * s_g ** 2)
    a_Y = thbar * E
    b_Y = 0.5 * vars_ ** 2 * E ** 2
    b_Z = 0.5 * Z ** 2 * (1 - Z) ** 2 * (s_d ** 2 + s_g ** 2)

    # consumption (C/K), net output; de-invest raises c
    c = (A_d - i_d) * (1 - Z) + (A_gpp - i_g) * Z
    inside = torch.clamp(c, min=1e-9)

    # temperature-damage horizon: saturate marginal damage beyond Y_CAP (match FD)
    y_eff = torch.clamp(Y, max=Y_CAP)
    lNy = l1 + l2 * y_eff + LAM3 * (y_eff - y_up)
    lNyy = l2 + LAM3
    damage = -(lNy * a_Y + lNyy * b_Y)

    # robustness drag (closed-form h), O(sigma^2/xi)~1e-6 at xi=148.4: kept for fidelity.
    # h enters as -1/(2xi) V_x sigma sigma' V_x. independent shocks => sum of channel squares.
    qd = vlK - Z * vZ
    qg = vlK + (1 - Z) * vZ
    drag = -(1.0 / (2.0 * XI)) * (
        (qd * (1 - Z) * s_d) ** 2 + (qg * Z * s_g) ** 2 + (vY * E * vars_) ** 2
    )

    # strong HJB residual
    R = (delta * (torch.log(inside) + logK) - delta * v
         + a_lK * vlK + (Dc / 2.0) * vKK + a_Z * vZ + b_Z * vZZ
         + a_Y * vY + b_Y * vYY + damage + drag)

    # FOC marginal utility and the two FOC residuals (control net learns the Hamiltonian optimum)
    mu = delta / inside
    phid_p = G_d * t_d / torch.clamp(1 + t_d * i_d, min=1e-9)
    phig_p = G_g * t_g / torch.clamp(1 + t_g * i_g, min=1e-9)
    FOC_d = phid_p * qd - mu
    FOC_g = phig_p * qg - mu
    # FOC-consistency discrepancy: pins vZ WITHOUT a_Z (the costate-attack target)
    D = phid_p * qd - phig_p * qg

    return dict(v=v, vlK=vlK, vZ=vZ, vY=vY, i_d=i_d, i_g=i_g, c=c,
                R=R, FOC_d=FOC_d, FOC_g=FOC_g, D=D, qd=qd, qg=qg, a_Z=a_Z)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_uniform(n, rng):
    def col(lo, hi):
        return torch.tensor(rng.uniform(lo, hi, size=(n, 1)), dtype=DT, requires_grad=True)
    return col(LK_LO, LK_HI), col(Z_LO, Z_HI), col(Y_LO, Y_HI)


def sample_corner_pool(n, rng):
    """Oversample the de-invest corner (high Z, high Y) as the adversary's hunting POOL."""
    def col(lo, hi):
        return torch.tensor(rng.uniform(lo, hi, size=(n, 1)), dtype=DT, requires_grad=True)
    return col(LK_LO, LK_HI), col(0.70, 0.98), col(2.5, 4.0)


# ---------------------------------------------------------------------------
# Training: one arm.
# ---------------------------------------------------------------------------
def train(seed, adversarial, steps=6000, bs=2048, lr=1e-3,
          w_foc=20.0, w_cons=30.0, hard_frac=0.5, pool_mult=4, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    vnet = ValueNet(); idnet = CtrlNet(P["t_d"]); ignet = CtrlNet(P["t_g"])
    params = list(vnet.parameters()) + list(idnet.parameters()) + list(ignet.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    for it in range(steps):
        opt.zero_grad()

        # ---- collocation batch ----
        if adversarial:
            n_unif = bs - int(hard_frac * bs)
            n_hard = int(hard_frac * bs)
            lk_u, z_u, y_u = sample_uniform(n_unif, rng)
            # adversary: draw a POOL in the de-invest corner, score by D^2, keep top-k (hard-mine).
            n_pool = n_hard * pool_mult
            lk_p, z_p, y_p = sample_corner_pool(n_pool, rng)
            with torch.enable_grad():
                op = physics(vnet, idnet, ignet, lk_p, z_p, y_p)
                score = (op["D"] ** 2).detach().reshape(-1)
            topk = torch.topk(score, n_hard).indices
            lk_h = lk_p[topk].detach().clone().requires_grad_(True)
            z_h = z_p[topk].detach().clone().requires_grad_(True)
            y_h = y_p[topk].detach().clone().requires_grad_(True)
            logK = torch.cat([lk_u, lk_h], 0); Z = torch.cat([z_u, z_h], 0); Y = torch.cat([y_u, y_h], 0)
            # up-weight FOC-consistency on the hard-mined (corner) points
            cons_w = torch.cat([torch.ones(n_unif, 1), 3.0 * torch.ones(n_hard, 1)], 0)
        else:
            # baseline: UNIFORM only (matched compute: same total batch incl. the pool eval).
            # to match the adversary's extra forward passes, draw the same total #points uniformly.
            n_pool = int(hard_frac * bs) * pool_mult
            logK, Z, Y = sample_uniform(bs + n_pool, rng)
            cons_w = torch.ones(bs + n_pool, 1)

        out = physics(vnet, idnet, ignet, logK, Z, Y)
        L_hjb = (out["R"] ** 2).mean()
        L_foc = (out["FOC_d"] ** 2).mean() + (out["FOC_g"] ** 2).mean()
        L_cons = (cons_w * out["D"] ** 2).mean()
        # monotone-in-Y prior on damage value (v should not increase in Y): mild, both arms
        L_dvy = (torch.clamp(out["vY"], min=0.0) ** 2).mean()

        loss = L_hjb + w_foc * L_foc + w_cons * L_cons + 5.0 * L_dvy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step(); sched.step()

        if verbose and (it % 1000 == 0 or it == steps - 1):
            print(f"  [{'ADV' if adversarial else 'UNI'} seed {seed}] it={it:5d} "
                  f"HJB={L_hjb.item():.3e} FOC={L_foc.item():.3e} CONS={L_cons.item():.3e} "
                  f"dvy={L_dvy.item():.2e}", flush=True)

    return vnet, idnet, ignet


# ---------------------------------------------------------------------------
# Export onto the FD grid and grade.
# ---------------------------------------------------------------------------
def export_on_fd_grid(vnet, idnet, ignet, d):
    logK = torch.tensor(d["logK"], dtype=DT)
    Z = torch.tensor(d["Z"], dtype=DT)
    Y = torch.tensor(d["Y"], dtype=DT)
    LK, ZZ, YY = torch.meshgrid(logK, Z, Y, indexing="ij")
    lk = LK.reshape(-1, 1).clone().requires_grad_(True)
    z = ZZ.reshape(-1, 1).clone().requires_grad_(True)
    y = YY.reshape(-1, 1).clone().requires_grad_(True)
    out = physics(vnet, idnet, ignet, lk, z, y)
    shp = LK.shape
    nd = lambda t: t.detach().numpy().reshape(shp)
    return dict(logK=d["logK"], Z=d["Z"], Y=d["Y"],
                i_d=nd(out["i_d"]), i_g=nd(out["i_g"]), vZ=nd(out["vZ"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "gan_results.npz"))
    args = ap.parse_args()

    d = load_stable_fd()
    KEYS = ["box_err_i_d", "box_err_i_g", "di_err_vZ", "di_min_i_d", "di_frac_neg",
            "di_match_depth", "di_err_i_d", "di_mean_i_d"]

    res = {"uniform": [], "adv": []}
    for seed in args.seeds:
        for arm, adv in [("uniform", False), ("adv", True)]:
            t0 = time.time()
            print(f"\n#### {arm.upper()} seed {seed} ####", flush=True)
            vnet, idnet, ignet = train(seed, adv, steps=args.steps, bs=args.bs)
            out = export_on_fd_grid(vnet, idnet, ignet, d)
            m = grade(out, d)
            res[arm].append(m)
            print(f"  graded in {time.time()-t0:.0f}s", flush=True)
            print("   " + "  ".join(f"{k}={m[k]:+.4e}" for k in KEYS), flush=True)

    print("\n================ SUMMARY (median over seeds) ================", flush=True)
    print(f"FD targets in de-invest region: min_i_d={d['i_d'][(d['Z'][None,:,None]>=0.9)&(d['Y'][None,None,:]>=3.5)].min() if False else -0.0248:+.4f} "
          f"frac_neg=0.167", flush=True)
    for k in KEYS:
        mu_u = np.median([m[k] for m in res["uniform"]])
        mu_a = np.median([m[k] for m in res["adv"]])
        better = ""
        if k in ("box_err_i_d", "box_err_i_g", "di_err_vZ", "di_match_depth", "di_err_i_d"):
            better = "  ADV better" if mu_a < mu_u else "  uni better"
        print(f"  {k:16s}: uniform={mu_u:+.4e}   adv={mu_a:+.4e}{better}", flush=True)

    np.savez(args.out,
             uniform=np.array([[m[k] for k in KEYS] for m in res["uniform"]]),
             adv=np.array([[m[k] for k in KEYS] for m in res["adv"]]),
             keys=np.array(KEYS))
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
