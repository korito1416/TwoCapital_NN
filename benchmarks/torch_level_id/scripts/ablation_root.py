"""
Ablation study on the WARM-STARTED (TF-surrogate) torch root regime (PostDamagePostTech).

Goal: from the warm start, change ONE knob at a time and measure its effect on
  (1) value residual floor   loss_v = rms(rhs - pv)
  (2) control losses         FOC_d, FOC_g
  (3) VALUE-LEVEL drift      mean_probe( V_after - V_before ),  V = pv / delta
      (a probe of whether continued training moves the absolute value LEVEL --
       the quantity that is only weakly pinned by the -delta*V term, delta=0.01)

V is recovered as pv/delta because the HJB residual is rhs - pv with pv = delta*V.
A fixed seeded probe set (256 pts) is evaluated before/after each config.

NOT a fix for weak-id; this is the exploratory ablation requested before that discussion.
"""
import os, sys, time, copy
import numpy as np
import torch

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
HERE = os.path.join(ROOT, "models_torch_train")
MT = os.path.join(ROOT, "models_torch")
for p in (HERE, MT):
    if p not in sys.path:
        sys.path.insert(0, p)

from tf_to_torch_loader import build_root_torch_model  # noqa
from params import PARAMS  # noqa
from train_root import sample_box, compute_terms, WarmupCosine  # noqa

DELTA = float(PARAMS.get("δ", PARAMS.get("delta", 0.01)))
NPROBE = 256
STEPS = 500


def fixed_probe(dtype):
    g = torch.Generator().manual_seed(12345)
    def col(lo, hi):
        return (lo + (hi - lo) * torch.rand((NPROBE, 1), generator=g)).to(dtype)
    return [col(4, 7), col(0.01, 0.99), col(0, 4),
            col(1, 6), col(0, 1/3.), col(-3, 5)]


def probe_V_resid(model, probe):
    logK = probe[0].detach().requires_grad_(True)
    Z = probe[1].detach().requires_grad_(True)
    Y = probe[2].detach().requires_grad_(True)
    logR, lam3, logxi = probe[3], probe[4], probe[5]
    rhs, pv, *_ = model.pde_rhs(logK, Z, Y, logR, lam3, logxi)
    V = (pv / DELTA).detach().cpu().numpy().reshape(-1)
    resid = (rhs - pv).detach().cpu().numpy().reshape(-1)
    return V, float(np.sqrt(np.mean(resid ** 2)))


def fresh_model(dtype):
    m = build_root_torch_model(load_surrogate=True, dtype=dtype)
    for net in (m.v_nn, m.i_g_nn, m.i_d_nn):
        net.to(dtype=dtype); net.train()
    return m


def train_adam(m, params, dtype, steps, lr, value_only=False, concentrate=False):
    opt_v = torch.optim.Adam(m.v_nn.parameters(), lr=lr)
    opt_c = torch.optim.Adam(list(m.i_g_nn.parameters()) + list(m.i_d_nn.parameters()), lr=lr)
    sch = WarmupCosine(lr, steps, max(1, steps // 100), 1e-5)
    for step in range(1, steps + 1):
        for g in opt_v.param_groups: g["lr"] = sch(step)
        for g in opt_c.param_groups: g["lr"] = sch(step)
        s = list(sample_box(params, 128, dtype, torch.device("cpu")))
        if concentrate:  # bias Y toward the damage/high-Y region [1.5,4]
            s[2] = (1.5 + 2.5 * torch.rand((128, 1), dtype=dtype))
        vloss, _, diag = compute_terms(m, params, *s)
        opt_v.zero_grad(set_to_none=True); vloss.backward(); opt_v.step()
        if not value_only:
            s2 = list(sample_box(params, 128, dtype, torch.device("cpu")))
            if concentrate:
                s2[2] = (1.5 + 2.5 * torch.rand((128, 1), dtype=dtype))
            _, closs, _ = compute_terms(m, params, *s2)
            opt_c.zero_grad(set_to_none=True); closs.backward(); opt_c.step()
    return diag


def train_lbfgs_value(m, params, dtype, iters):
    # L-BFGS on the VALUE net over a FIXED large batch (proper for quasi-Newton).
    s = sample_box(params, 2048, dtype, torch.device("cpu"))
    opt = torch.optim.LBFGS(m.v_nn.parameters(), max_iter=iters,
                            history_size=20, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(set_to_none=True)
        vloss, _, _ = compute_terms(m, params, *s)
        vloss.backward()
        return vloss
    opt.step(closure)
    _, _, diag = compute_terms(m, params, *s)
    return diag


def run_config(name, dtype, runner):
    t0 = time.time()
    m = fresh_model(dtype)
    probe = fixed_probe(dtype)
    V0, r0 = probe_V_resid(m, probe)
    diag = runner(m, PARAMS, dtype)
    V1, r1 = probe_V_resid(m, probe)
    dV = V1 - V0
    out = {
        "config": name, "dtype": str(dtype).replace("torch.", ""),
        "loss_v_start": r0, "loss_v_final": r1,
        "FOC_d": diag["loss_FOC_d"], "FOC_g": diag["loss_FOC_g"],
        "levelDrift_mean": float(np.mean(dV)), "levelDrift_std": float(np.std(dV)),
        "levelDrift_absmax": float(np.max(np.abs(dV))),
        "sec": round(time.time() - t0, 1),
    }
    print("ROW " + repr(out), flush=True)
    return out


def main():
    torch.manual_seed(0); np.random.seed(0)
    print(f"delta={DELTA}  steps={STEPS}  nprobe={NPROBE}")
    # fast f32 configs FIRST so partial results survive any time limit; slow ones last.
    configs = [
        ("A_base_f32_adam",      torch.float32, lambda m,p,d: train_adam(m,p,d,STEPS,4e-4)),
        ("D_value_only_f32",     torch.float32, lambda m,p,d: train_adam(m,p,d,STEPS,4e-4,value_only=True)),
        ("F_concentrateY_f32",   torch.float32, lambda m,p,d: train_adam(m,p,d,STEPS,4e-4,concentrate=True)),
        ("G_highLR1e-3_f32",     torch.float32, lambda m,p,d: train_adam(m,p,d,STEPS,1e-3)),
        ("E_longer2000_f32",     torch.float32, lambda m,p,d: train_adam(m,p,d,2000,4e-4)),
        ("B_float64_adam",       torch.float64, lambda m,p,d: train_adam(m,p,d,STEPS,4e-4)),
        ("C_lbfgs_value_f64",    torch.float64, lambda m,p,d: train_lbfgs_value(m,p,d,100)),
    ]
    rows = []
    for name, dt, runner in configs:
        try:
            rows.append(run_config(name, dt, runner))
        except Exception as e:
            print(f"ROW_ERR {name}: {type(e).__name__}: {e}")
    print("\n==== ABLATION SUMMARY (warm-start root regime) ====")
    hdr = f"{'config':22s} {'dt':7s} {'loss_v0':>9s} {'loss_v1':>9s} {'FOC_d':>9s} {'FOC_g':>9s} {'dV_mean':>9s} {'dV_std':>8s} {'dV_absmax':>9s} {'s':>5s}"
    print(hdr)
    for r in rows:
        print(f"{r['config']:22s} {r['dtype']:7s} {r['loss_v_start']:9.3e} {r['loss_v_final']:9.3e} "
              f"{r['FOC_d']:9.3e} {r['FOC_g']:9.3e} {r['levelDrift_mean']:+9.3e} {r['levelDrift_std']:8.2e} "
              f"{r['levelDrift_absmax']:9.3e} {r['sec']:5.0f}")


if __name__ == "__main__":
    main()
