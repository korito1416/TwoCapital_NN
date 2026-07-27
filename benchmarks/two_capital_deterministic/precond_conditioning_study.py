"""
Conditioning / preconditioning study for the DGM-PIA loss, on the fast deterministic
two-capital benchmark (numpy only, login-node).

Motivation (Lars's ask): the robust HJB is a convex-concave saddle (concave in the
controls i^d,i^g; convex in the distortions h). The NN loss, however, sums RMS norms
of residuals with VERY different curvatures -- a flat HJB-residual direction (weak v'
identification) and steep FOC directions -- so SGD is ill-conditioned. We quantify the
conditioning and test preconditioners. The deterministic benchmark carries the same
3-loss structure (HJB residual R + FOC_d + FOC_g) with closed-form algebra, so every
curvature here is analytic.

Control-dependent Hamiltonian at a point (Z, slope p=v'):
    h(i_d,i_g) = delta*log c + (1-Z) q_d phi_d(i_d) + Z q_g phi_g(i_g),
    c = (A_d-i_d)(1-Z)+(A_g-i_g)Z,  q_d=1-Z p,  q_g=1+(1-Z)p,
    phi_j = alpha_j + Gamma_j log(1+theta_j i_j).
Analytic 2x2 control Hessian (negative definite => concave):
    H_dd = -delta(1-Z)^2/c^2 - (1-Z) q_d Gamma_d theta_d^2/(1+theta_d i_d)^2
    H_gg = -delta Z^2/c^2     - Z     q_g Gamma_g theta_g^2/(1+theta_g i_g)^2
    H_dg = -delta(1-Z)Z/c^2
The FOC curvature scale is Gamma*theta^2; HALF adjustment cost keeps phi'(0)=Gamma*theta
fixed (=1.002) but takes Gamma*theta^2: 16.73 -> 8.37, i.e. the control problem is half
as stiff -- directly relevant to the half-adjcost retraining.

Weak v' identification (envelope; controls at their FOC):
    dR/dp   = mu_Z = Z(1-Z)(phi_g-phi_d)              (tiny)
    dFOC_d/dp = (-Z)    Gamma_d theta_d/(1+theta_d i_d)
    dFOC_g/dp = (1-Z)  Gamma_g theta_g/(1+theta_g i_g)  (O(1))
so the HJB residual carries ~mu_Z * v' information while the FOCs carry ~Gamma*theta
information -- orders of magnitude apart. This is why FOC losses are essential and why
preconditioning the HJB-residual block matters.

Preconditioners tested on the global Gauss-Newton Hessian J^T J (J = d[R;FOC_d;FOC_g]/d[v;i_d;i_g]):
  #1 pointwise HJB-residual weighting : scale R rows by 1/(|mu_Z|+eps)
  #3 term/block balancing             : scale each block (R,FOC_d,FOC_g) by 1/rms(row-norm)
  #2 Newton column (Jacobi) scaling   : scale each variable column by 1/||col||
We report the spectral condition number kappa(J^T J) for each.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import two_capital_model as M
from theta_sensitivity import solve_fd

OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OD, exist_ok=True)


def make_P(half):
    P = M.load_calibration("A_g_prime_prime")
    if half:
        P["theta_d"] = P["theta_g"] = 8.35
        P["Gamma_d"] = P["Gamma_g"] = 0.12
    return P


# ---------------------------------------------------------------------------
# Analytic control Hessian and v'-sensitivity on a grid (uses the FD solution)
# ---------------------------------------------------------------------------
def fields_on_grid(P, Zc):
    """Interpolate the fine-FD solution (v, slope, i_d, i_g, c) onto coarse Zc."""
    fine = solve_fd(P, n=4000)
    Zf = fine["Z"]
    v = np.interp(Zc, Zf, fine["v"])
    p = np.interp(Zc, Zf, fine["slope"])     # v'
    i_d, i_g, c = M.controls(Zc, p, P)
    return v, p, i_d, i_g, c


def control_hessian(Zc, p, i_d, i_g, c, P):
    d, g, th_d, th_g, G_d, G_g, dl = (1 - Zc), Zc, P["theta_d"], P["theta_g"], \
        P["Gamma_d"], P["Gamma_g"], P["delta"]
    q_d = 1.0 - Zc * p
    q_g = 1.0 + (1.0 - Zc) * p
    H_dd = -dl * d**2 / c**2 - d * q_d * G_d * th_d**2 / (1.0 + th_d * i_d)**2
    H_gg = -dl * g**2 / c**2 - g * q_g * G_g * th_g**2 / (1.0 + th_g * i_g)**2
    H_dg = -dl * d * g / c**2
    return H_dd, H_gg, H_dg, q_d, q_g


def cond_2x2(a, b, off):
    """Condition number of [[a,off],[off,b]] (symmetric)."""
    tr = a + b
    det = a * b - off**2
    disc = np.sqrt(np.maximum(tr**2 - 4 * det, 0.0))
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    lo = np.minimum(np.abs(l1), np.abs(l2))
    hi = np.maximum(np.abs(l1), np.abs(l2))
    return hi / np.maximum(lo, 1e-300)


# ---------------------------------------------------------------------------
# Global Gauss-Newton Jacobian and preconditioned condition numbers
# ---------------------------------------------------------------------------
def residuals(u, Zc, dZ, v0, vN, P):
    """u = [v(0..n-1), i_d, i_g] -> F = [R, FOC_d, FOC_g]."""
    n = len(Zc)
    v = u[:n]; i_d = u[n:2 * n]; i_g = u[2 * n:]
    th_d, th_g, G_d, G_g, dl = P["theta_d"], P["theta_g"], P["Gamma_d"], P["Gamma_g"], P["delta"]
    A_d, A_g = P["A_d"], P["A_g"]
    # v' by central difference with fixed Dirichlet ends
    vext = np.empty(n + 2); vext[1:-1] = v; vext[0] = v0; vext[-1] = vN
    p = (vext[2:] - vext[:-2]) / (2.0 * dZ)
    q_d = 1.0 - Zc * p
    q_g = 1.0 + (1.0 - Zc) * p
    c = (A_d - i_d) * (1 - Zc) + (A_g - i_g) * Zc
    phi_d = P["alpha_d"] + G_d * np.log(np.maximum(1 + th_d * i_d, 1e-12))
    phi_g = P["alpha_g"] + G_g * np.log(np.maximum(1 + th_g * i_g, 1e-12))
    mu = Zc * (1 - Zc) * (phi_g - phi_d)
    R = dl * np.log(np.maximum(c, 1e-12)) - dl * v + (1 - Zc) * phi_d + Zc * phi_g + mu * p
    FOC_d = -dl / c + q_d * G_d * th_d / (1 + th_d * i_d)
    FOC_g = -dl / c + q_g * G_g * th_g / (1 + th_g * i_g)
    return np.concatenate([R, FOC_d, FOC_g]), mu


def jacobian(u, Zc, dZ, v0, vN, P, eps=1e-6):
    F0, _ = residuals(u, Zc, dZ, v0, vN, P)
    m, N = len(F0), len(u)
    J = np.zeros((m, N))
    for j in range(N):
        du = np.zeros(N); du[j] = eps * max(1.0, abs(u[j]))
        Fp, _ = residuals(u + du, Zc, dZ, v0, vN, P)
        J[:, j] = (Fp - F0) / du[j]
    return J


def kappa(J):
    s = np.linalg.svd(J, compute_uv=False)
    s = s[s > 0]
    return (s[0] / s[-1])**2  # cond(J^T J) = cond(J)^2


def study(half):
    tag = "half" if half else "base"
    P = make_P(half)
    n = 60
    Zc = np.linspace(0.08, 0.92, n)
    dZ = Zc[1] - Zc[0]
    v0, vN = M.boundary_values(P)
    v, p, i_d, i_g, c = fields_on_grid(P, Zc)

    # --- analytic control Hessian conditioning ---
    H_dd, H_gg, H_dg, q_d, q_g = control_hessian(Zc, p, i_d, i_g, c, P)
    kc = cond_2x2(-H_dd, -H_gg, -H_dg)
    # diagonal Newton (#2) scaling D=diag(1/sqrt(-Hdd),1/sqrt(-Hgg)) -> cond of D(-H)D
    sd, sg = 1.0 / np.sqrt(-H_dd), 1.0 / np.sqrt(-H_gg)
    kc_newton = cond_2x2(1.0, 1.0, -H_dg * sd * sg)

    # --- weak v' identification ---
    dR_dp = Zc * (1 - Zc) * (
        (P["alpha_g"] + P["Gamma_g"] * np.log(1 + P["theta_g"] * i_g))
        - (P["alpha_d"] + P["Gamma_d"] * np.log(1 + P["theta_d"] * i_d)))
    dFOCd_dp = np.abs(-Zc * P["Gamma_d"] * P["theta_d"] / (1 + P["theta_d"] * i_d))
    dFOCg_dp = np.abs((1 - Zc) * P["Gamma_g"] * P["theta_g"] / (1 + P["theta_g"] * i_g))
    ratio = np.maximum(dFOCd_dp, dFOCg_dp) / np.maximum(np.abs(dR_dp), 1e-300)

    # --- global Gauss-Newton conditioning + preconditioners ---
    u = np.concatenate([v, i_d, i_g])
    J = jacobian(u, Zc, dZ, v0, vN, P)
    _, mu = residuals(u, Zc, dZ, v0, vN, P)
    eps = 1e-4
    kappas = {"unpreconditioned": kappa(J)}
    # #1 pointwise HJB-residual weight on the R block (rows 0..n-1)
    w1 = np.ones(3 * n); w1[:n] = 1.0 / (np.abs(mu) + eps)
    kappas["#1 HJB-residual weight"] = kappa(w1[:, None] * J)
    # #3 block balancing: scale each block by 1/rms(row-norm)
    rn = np.linalg.norm(J, axis=1)
    w3 = np.ones(3 * n)
    for b in range(3):
        sl = slice(b * n, (b + 1) * n)
        w3[sl] = 1.0 / max(np.sqrt(np.mean(rn[sl]**2)), 1e-300)
    kappas["#3 block balancing"] = kappa(w3[:, None] * J)
    # #1 + #3 (naive product -- double-rescales the R block, shown to backfire)
    w13 = w1 * w3
    kappas["#1+#3 (naive)"] = kappa(w13[:, None] * J)
    # #1 THEN #3 (principled): apply #1 pointwise, re-balance blocks on the result
    Jw1 = w1[:, None] * J
    rn1 = np.linalg.norm(Jw1, axis=1)
    w3b = np.ones(3 * n)
    for b in range(3):
        sl = slice(b * n, (b + 1) * n)
        w3b[sl] = 1.0 / max(np.sqrt(np.mean(rn1[sl]**2)), 1e-300)
    kappas["#1 then #3 (re-balanced)"] = kappa(w3b[:, None] * Jw1)
    # #2 Newton/Jacobi column scaling
    cn = np.linalg.norm(J, axis=0)
    Jc = J / np.maximum(cn, 1e-300)[None, :]
    kappas["#2 Jacobi column"] = kappa(Jc)
    Jbest = w3b[:, None] * Jw1
    kappas["#1 then #3 + #2 col"] = kappa(Jbest / np.maximum(
        np.linalg.norm(Jbest, axis=0), 1e-300)[None, :])

    return dict(tag=tag, P=P, Zc=Zc, kc=kc, kc_newton=kc_newton, q_d=q_d, q_g=q_g,
                dR_dp=np.abs(dR_dp), dFOCd_dp=dFOCd_dp, dFOCg_dp=dFOCg_dp,
                ratio=ratio, kappas=kappas)


def main():
    base = study(half=False)
    half = study(half=True)

    print("\n================ CONTROL HESSIAN (analytic) ================")
    for s in (base, half):
        gth = s["P"]["Gamma_d"] * s["P"]["theta_d"]
        gth2 = s["P"]["Gamma_d"] * s["P"]["theta_d"]**2
        print(f"[{s['tag']:>4}] Gamma*theta={gth:.3f}  Gamma*theta^2={gth2:.3f} "
              f"| cond(-H) median={np.median(s['kc']):.2f} max={np.max(s['kc']):.2f} "
              f"-> after Newton(#2) diag: median={np.median(s['kc_newton']):.3f}")

    print("\n================ WEAK v' IDENTIFICATION ====================")
    for s in (base, half):
        print(f"[{s['tag']:>4}] |dR/dv'| median={np.median(s['dR_dp']):.2e}  "
              f"|dFOC/dv'| median={np.median(np.maximum(s['dFOCd_dp'], s['dFOCg_dp'])):.2e}  "
              f"ratio median={np.median(s['ratio']):.1f}  max={np.max(s['ratio']):.1f}")

    print("\n================ GLOBAL GAUSS-NEWTON cond(J^T J) ===========")
    order = ["unpreconditioned", "#1 HJB-residual weight", "#3 block balancing",
             "#1+#3 (naive)", "#1 then #3 (re-balanced)", "#2 Jacobi column",
             "#1 then #3 + #2 col"]
    for s in (base, half):
        print(f"--- {s['tag']} ---")
        k0 = s["kappas"]["unpreconditioned"]
        for k in order:
            v = s["kappas"][k]
            print(f"    {k:<26} kappa={v:.3e}   ({k0/v:6.1f}x better)")

    # ---------------- Figures ----------------
    # Fig 1: control-Hessian condition number vs Z (base vs half) + Newton-fixed
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for s, col in ((base, "#1f77b4"), (half, "#d62728")):
        ax[0].plot(s["Zc"], s["kc"], color=col, lw=2, label=f"{s['tag']} (raw)")
        ax[0].plot(s["Zc"], s["kc_newton"], color=col, lw=1.4, ls="--",
                   label=f"{s['tag']} + Newton(#2)")
    ax[0].set_yscale("log"); ax[0].set_xlabel("Z"); ax[0].set_ylabel(r"cond$(-H_{\rm ctrl})$")
    ax[0].set_title("Control-Hessian conditioning\n(diagonal Newton scaling #2 -> ~1)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

    for s, col in ((base, "#1f77b4"), (half, "#d62728")):
        ax[1].plot(s["Zc"], s["ratio"], color=col, lw=2, label=f"{s['tag']}")
    ax[1].set_yscale("log"); ax[1].set_xlabel("Z")
    ax[1].set_ylabel(r"$|\partial FOC/\partial v'| \, / \, |\partial R/\partial v'|$")
    ax[1].set_title("Weak $v'$ identification: FOC vs HJB-residual\nsensitivity (motivates #1/#3)")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p1 = os.path.join(OD, "precond_control_and_weakID.png")
    fig.savefig(p1, dpi=150); print("\nsaved", p1)

    # Fig 2: global condition-number bar chart, base vs half
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = order
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w/2, [base["kappas"][k] for k in labels], w, label="baseline (theta=16.7)", color="#1f77b4")
    ax.bar(x + w/2, [half["kappas"][k] for k in labels], w, label="half adjcost (theta=8.35)", color="#d62728")
    ax.set_yscale("log"); ax.set_ylabel(r"$\kappa(J^TJ)$ (spectral)")
    ax.set_title("Global Gauss-Newton condition number: preconditioning the 3-loss")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.legend(); ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    p2 = os.path.join(OD, "precond_global_condition.png")
    fig.savefig(p2, dpi=150); print("saved", p2)


if __name__ == "__main__":
    main()
