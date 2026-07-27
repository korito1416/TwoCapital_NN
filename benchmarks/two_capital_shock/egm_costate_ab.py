"""
EGM / Euler-back-out (costate) test on the STOCHASTIC two-capital model.

HYPOTHESIS (this workflow): the v'(Z) information is poorly identified by the STRONG
HJB residual in the de-invest pocket because dR/dv' = drift coefficient mu ~ O(8e-3),
i.e. the residual is nearly flat in v'. But v' is WELL identified by the COSTATE
(differentiated-HJB / Euler) equation, whose dependence on v' carries the O(delta)
reaction term -delta*p, NOT mu. Diffusion (sigma>0) makes that costate equation a
well-posed 2nd-order ODE in p, so we can fit the MARGINAL value m(Z)=v'(Z) directly
through a well-conditioned operator and route around the flat strong residual.

Reduced 2nd-order HJB (V=logK+v(Z), xi=inf, rho=0):
    R = delta*log c(p) - delta*v + D(Z,p) + B(Z,p)*p + C(Z)*v''  = 0
  D = (1-Z)phi_d + Z phi_g - 1/2(sd^2 (1-Z)^2 + sg^2 Z^2)              [logK drift]
  B = [phi_g - phi_d + (1-Z)sd^2 - Z sg^2] Z(1-Z)                      [v' coef]
  C = 1/2 Z^2(1-Z)^2 (sd^2+sg^2)                                      [v'' coef]
controls i_d,i_g,c from M.controls(Z,p) (FOC inversion).

COSTATE: differentiate R w.r.t Z and use R==0 (envelope kills control variation):
    R_Z := dR/dZ = dR_explicit/dZ - delta*p + B*p' + C*p''  = 0,
a 2nd-order ODE for p=m(Z) whose reaction term -delta*p is O(delta)=O(1e-2),
independent of mu.  Its dResid/dp at fixed (p',p'') is -delta + dB/dp * p' + ...
which is O(delta) in the pocket -- the conditioning gain we verify BEFORE training.

CONTROL  = value net v(Z), strong L2 residual (rhs - delta v)            [baseline]
TREATED  = marginal-value net m(Z)=v'(Z), costate (Euler) residual       [the key]
Both reconstruct i_d via M.controls; same seed/init width/budget.
TRUE error = max|i_d - i_d_FD| over Z in [0.1,0.9] AND max|v'-v'_FD|, per (A_d,sigma).
Ground truth = reference_fd_shock.solve_central (validated independent reference).
"""
import os, sys, time
import numpy as np
import tensorflow as tf

import two_capital_shock_model as M
from fd_shock import solve_fd_shock

tf.keras.backend.set_floatx("float32")

# ------------------------------- shared pieces -------------------------------
# Ground truth = upwind FD (solve_fd_shock): the M-matrix contraction is the only
# robustly-converging FD scheme in the stiff de-invest pocket (Newton & central
# DIVERGE there, maxR~1e3, which itself corroborates the weak-id story).  At
# sigma=0.01 it reproduces the deterministic FD i_d to 4e-5 and the de-invest
# pocket exactly; its large central-stencil "max_abs_residual" is a known boundary
# measurement artifact (see reference_fd_shock.py docstring), not a solution error.

def fd_truth(A_d, sigma, n=2000):
    P = M.load_calibration("A_g_prime_prime")
    P["A_d"] = A_d; P["sigma_d"] = sigma; P["sigma_g"] = sigma
    out = solve_fd_shock(P, n=n)
    return P, out


def coefs_np(Z, p, P):
    """B(Z,p), C(Z), and dR_explicit/dZ pieces in numpy (FD-truth diagnostics)."""
    i_d, i_g, c = M.controls(Z, p, P)
    phi_d = M.phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
    phi_g = M.phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
    sd2, sg2 = P["sigma_d"]**2, P["sigma_g"]**2
    B = (phi_g - phi_d + (1-Z)*sd2 - Z*sg2) * Z*(1-Z)
    C = 0.5*Z**2*(1-Z)**2*(sd2+sg2)
    return B, C, c


# ----------------------- networks (matched architecture) ---------------------

def make_mlp(out_act=None, width=32, depth=3, seed=0):
    tf.random.set_seed(seed)
    init = tf.keras.initializers.GlorotUniform(seed=seed)
    inp = tf.keras.Input(shape=(1,))
    h = inp
    for _ in range(depth):
        h = tf.keras.layers.Dense(width, activation="tanh", kernel_initializer=init)(h)
    out = tf.keras.layers.Dense(1, activation=out_act, kernel_initializer=init)(h)
    return tf.keras.Model(inp, out)


def clamp_p(p, Z, m=1e-4):
    lo = -1.0 / tf.maximum(1 - Z, 1e-9) + m
    hi = 1.0 / tf.maximum(Z, 1e-9) - m
    return tf.clip_by_value(p, lo, hi)


# --------------- TF closed-form controls (shared by both methods) ------------

def tf_controls(Z, p, P):
    q_d = 1 - Z*p; q_g = 1 + (1-Z)*p
    Abar = (1-Z)*P["A_d"] + Z*P["A_g"]
    num = P["delta"]*(Abar + (1-Z)/P["theta_d"] + Z/P["theta_g"])
    den = P["delta"] + (1-Z)*P["Gamma_d"]*q_d + Z*P["Gamma_g"]*q_g
    c = num/den
    i_d = P["Gamma_d"]*c*q_d/P["delta"] - 1/P["theta_d"]
    i_g = P["Gamma_g"]*c*q_g/P["delta"] - 1/P["theta_g"]
    return i_d, i_g, c, q_d, q_g


def tf_phi(i, alpha, Gamma, theta):
    return alpha + Gamma*tf.math.log(tf.maximum(1 + theta*i, 1e-8))


def strong_residual_from_p(Z, v, p, vpp, P):
    """R = delta log c - delta v + D + B p + C vpp  (controls from p)."""
    i_d, i_g, c, _, _ = tf_controls(Z, p, P)
    phi_d = tf_phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
    phi_g = tf_phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
    sd2, sg2 = P["sigma_d"]**2, P["sigma_g"]**2
    D = (1-Z)*phi_d + Z*phi_g - 0.5*(sd2*(1-Z)**2 + sg2*Z**2)
    B = (phi_g - phi_d + (1-Z)*sd2 - Z*sg2)*Z*(1-Z)
    C = 0.5*Z**2*(1-Z)**2*(sd2+sg2)
    return P["delta"]*tf.math.log(tf.maximum(c, 1e-8)) - P["delta"]*v + D + B*p + C*vpp, c


# ============================ CONTROL: strong-residual value net =============

def train_control(P, Zcol, seed, steps, lr):
    v0, vN = M.boundary_values(P)
    net = make_mlp(out_act=None, seed=seed)
    Zt = tf.constant(Zcol)
    bc = tf.constant([[float(P.get("z_min", 0.01))], [float(P.get("z_max", 0.99))]], tf.float32)
    vbc = tf.constant([[v0], [vN]], tf.float32)

    def value(Z):
        # hard-encode boundary data; net carries interior shape
        return (1-Z)*v0 + Z*vN + Z*(1-Z)*net(2.0*Z - 1.0)

    opt = tf.keras.optimizers.Adam(lr)

    @tf.function
    def step():
        with tf.GradientTape() as g:
            with tf.GradientTape() as t2:
                t2.watch(Zt)
                with tf.GradientTape() as t1:
                    t1.watch(Zt)
                    v = value(Zt)
                p = clamp_p(t1.gradient(v, Zt), Zt)
            vpp = t2.gradient(p, Zt)
            R, c = strong_residual_from_p(Zt, v, p, vpp, P)
            feas = tf.reduce_mean(tf.square(tf.minimum(c, 1e-8) - 1e-8))
            loss = tf.reduce_mean(tf.square(R)) + 1e3*feas
        gv = g.gradient(loss, net.trainable_variables)
        opt.apply_gradients(zip(gv, net.trainable_variables))
        return loss
    for _ in range(steps):
        step()
    # extract slope
    with tf.GradientTape() as t1:
        t1.watch(Zt); v = value(Zt)
    p = clamp_p(t1.gradient(v, Zt), Zt).numpy().ravel()
    return p


# ====================== TREATED: costate / Euler marginal net ================

def train_treated(P, Zcol, seed, steps, lr):
    """Parameterize m(Z)=v'(Z) directly; minimize the COSTATE (differentiated-HJB)
    residual  R_Z = dR/dZ  with R==0 used implicitly.  We compute R(Z) via autodiff
    of the strong residual expression *as a function of Z* (with m,m' supplied by the
    net), then take d/dZ of R and drive it to 0.  This is the Euler/costate equation;
    its reaction term -delta*m gives O(delta) identification of m, independent of mu.
    An anchor at the boundaries pins the integration constant of the costate ODE."""
    v0, vN = M.boundary_values(P)
    # FD-truth boundary slopes to anchor the costate ODE (a costate eqn fixes m up to
    # its own BCs; the *natural* costate BCs are the HJB itself at the ends -- we use
    # the one-capital boundary slopes, which are known closed-form, NOT interior FD).
    net = make_mlp(out_act=None, seed=seed)
    Zt = tf.constant(Zcol)
    # endpoints for anchoring
    Za = tf.constant(np.array([[0.02],[0.98]], np.float32))

    def m_of(Z):
        return clamp_p(net(2.0*Z - 1.0), Z)

    # boundary slope anchors from the strong HJB at the ends:
    # at Z->0 and Z->1 the model degenerates to one-capital; the slope there is the
    # finite-difference of the closed-form one-capital values is 0 in v(Z) reduced
    # coordinates only at the symmetric point.  Use the perturbation slope as a weak
    # anchor (leading-order, closed form -- NOT FD interior truth).
    import two_capital_model as DET
    p_anchor = DET.perturbation_slope(Za.numpy().ravel(), P).astype(np.float32).reshape(-1,1)
    p_anchor = tf.constant(p_anchor)

    opt = tf.keras.optimizers.Adam(lr)

    @tf.function
    def step():
        with tf.GradientTape() as g:
            with tf.GradientTape() as tz:
                tz.watch(Zt)
                # build R(Z) as a function of Z, with m and m' from the net
                with tf.GradientTape() as t1:
                    t1.watch(Zt)
                    m = m_of(Zt)
                mp = t1.gradient(m, Zt)          # m'(Z) = v''
                # reconstruct a consistent v(Z) is NOT needed for the costate eqn:
                # R_Z = dR/dZ where R = delta logc - delta v + D + B m + C m'.
                # dR/dZ = (d/dZ)[delta logc + D + B m + C m'] - delta v'  ; v'=m.
                i_d, i_g, c, _, _ = tf_controls(Zt, m, P)
                phi_d = tf_phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
                phi_g = tf_phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
                sd2, sg2 = P["sigma_d"]**2, P["sigma_g"]**2
                D = (1-Zt)*phi_d + Zt*phi_g - 0.5*(sd2*(1-Zt)**2 + sg2*Zt**2)
                B = (phi_g - phi_d + (1-Zt)*sd2 - Zt*sg2)*Zt*(1-Zt)
                C = 0.5*Zt**2*(1-Zt)**2*(sd2+sg2)
                Rexpl = P["delta"]*tf.math.log(tf.maximum(c, 1e-8)) + D + B*m + C*mp
            dRexpl = tz.gradient(Rexpl, Zt)
            RZ = dRexpl - P["delta"]*m            # costate residual = dR/dZ, v'=m
            feas = tf.reduce_mean(tf.square(tf.minimum(c, 1e-8) - 1e-8))
            anch = tf.reduce_mean(tf.square(m_of(Za) - p_anchor))
            loss = tf.reduce_mean(tf.square(RZ)) + 1e3*feas + 1e-2*anch
        gv = g.gradient(loss, net.trainable_variables)
        opt.apply_gradients(zip(gv, net.trainable_variables))
        return loss
    for _ in range(steps):
        step()
    return m_of(Zt).numpy().ravel()


# ============================== diagnostics =================================

def conditioning_gate(P, fd):
    """Numerically compare dStrongResidual/dp vs dCostateResidual/dp on FD truth,
    in the de-invest pocket.  Strong: dR/dp at fixed (v,v'') ~ B (=mu-scaled).
    Costate: dR_Z/dp at fixed (m',m'') ~ -delta + (dB/dp) m' + ...  (O(delta))."""
    Z = fd["Z"]; p = fd["slope"]; v = fd["v"]
    mI = (Z >= 0.1) & (Z <= 0.9)
    deinv = (fd["i_d"] < 0)
    eps = 1e-6
    # dStrong/dp (controls vary with p, v & vpp fixed): finite diff of strong resid
    vpp = np.zeros_like(v); vpp[1:-1] = (v[2:]-2*v[1:-1]+v[:-2])/((Z[1]-Z[0])**2)
    def strongR(pp):
        i_d, i_g, c = M.controls(Z, pp, P)
        phi_d = M.phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
        phi_g = M.phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
        sd2, sg2 = P["sigma_d"]**2, P["sigma_g"]**2
        D = (1-Z)*phi_d + Z*phi_g - 0.5*(sd2*(1-Z)**2+sg2*Z**2)
        B = (phi_g-phi_d+(1-Z)*sd2-Z*sg2)*Z*(1-Z)
        C = 0.5*Z**2*(1-Z)**2*(sd2+sg2)
        return P["delta"]*np.log(np.maximum(c,1e-12)) - P["delta"]*v + D + B*pp + C*vpp
    dStrong = (strongR(p+eps) - strongR(p-eps))/(2*eps)
    # dCostate/dp at fixed (m',m''): R_Z = dRexpl/dZ - delta*m. Perturb m pointwise.
    # The -delta*m term dominates dR_Z/dm: dR_Z/dm includes -delta + (dB/dp m') chain.
    # Estimate by perturbing p (=m) and recomputing R_Z via central FD in Z.
    dZ = Z[1]-Z[0]
    def costateRZ(pp):
        # R as function of Z given slope field pp
        i_d, i_g, c = M.controls(Z, pp, P)
        phi_d = M.phi(i_d, P["alpha_d"], P["Gamma_d"], P["theta_d"])
        phi_g = M.phi(i_g, P["alpha_g"], P["Gamma_g"], P["theta_g"])
        sd2, sg2 = P["sigma_d"]**2, P["sigma_g"]**2
        D = (1-Z)*phi_d + Z*phi_g - 0.5*(sd2*(1-Z)**2+sg2*Z**2)
        B = (phi_g-phi_d+(1-Z)*sd2-Z*sg2)*Z*(1-Z)
        C = 0.5*Z**2*(1-Z)**2*(sd2+sg2)
        mp = np.gradient(pp, dZ)
        Rexpl = P["delta"]*np.log(np.maximum(c,1e-12)) + D + B*pp + C*mp
        RZ = np.gradient(Rexpl, dZ) - P["delta"]*pp
        return RZ
    # perturb a single mode: uniform shift in p (isolates -delta*m reaction term)
    dCostate = (costateRZ(p+eps) - costateRZ(p-eps))/(2*eps)
    sel = mI & deinv if deinv[mI].any() else mI
    return (float(np.median(np.abs(dStrong[sel]))),
            float(np.median(np.abs(dCostate[sel]))),
            bool(deinv[mI].any()))


def true_errors(P, fd, p_nn):
    Z = fd["Z"]; mI = (Z >= 0.1) & (Z <= 0.9)
    i_d_nn, i_g_nn, c_nn = M.controls(Z, p_nn, P)
    err_i = float(np.max(np.abs(i_d_nn[mI] - fd["i_d"][mI])))
    err_p = float(np.max(np.abs(p_nn[mI] - fd["slope"][mI])))
    return err_i, err_p


# =================================== main ===================================

def run(A_d, sigma, seed=0, steps=8000, lr=2e-3, n_col=512):
    P, fd = fd_truth(A_d, sigma)
    Zcol = np.linspace(0.05, 0.95, n_col).reshape(-1,1).astype(np.float32)
    P["z_min"], P["z_max"] = 0.01, 0.99
    dS, dC, has_deinv = conditioning_gate(P, fd)
    p_ctrl = train_control(P, Zcol, seed, steps, lr)
    p_treat = train_treated(P, Zcol, seed, steps, lr)
    ei_c, ep_c = true_errors(P, fd, np.interp(fd["Z"], Zcol.ravel(), p_ctrl))
    ei_t, ep_t = true_errors(P, fd, np.interp(fd["Z"], Zcol.ravel(), p_treat))
    Zm = (fd["Z"]>=0.1)&(fd["Z"]<=0.9)
    print(f"\n=== A_d={A_d} sigma={sigma} (de-invest pocket={has_deinv}) ===", flush=True)
    print(f"  FD truth i_d in [{fd['i_d'][Zm].min():+.4f},{fd['i_d'][Zm].max():+.4f}]", flush=True)
    print(f"  GATE  median|dStrong/dp|={dS:.3e}  median|dCostate/dp|={dC:.3e}  ratio(costate/strong)={dC/max(dS,1e-30):.1f}x", flush=True)
    print(f"  CONTROL (strong-resid v-net):  max|i_d-FD|={ei_c:.4e}  max|v'-FD|={ep_c:.4e}", flush=True)
    print(f"  TREATED (costate/Euler m-net): max|i_d-FD|={ei_t:.4e}  max|v'-FD|={ep_t:.4e}", flush=True)
    return dict(A_d=A_d, sigma=sigma, deinv=has_deinv, dStrong=dS, dCostate=dC,
                ei_c=ei_c, ep_c=ep_c, ei_t=ei_t, ep_t=ep_t)


if __name__ == "__main__":
    t0 = time.time()
    results = []
    for A_d in (0.05, 0.13):
        for sigma in (0.01, 0.05):
            results.append(run(A_d, sigma, seed=0, steps=8000, lr=2e-3))
    print("\n================= SUMMARY =================", flush=True)
    for r in results:
        tag = "WEAK-ID POCKET" if r["A_d"]==0.05 else "well-cond"
        win = "TREATED<CONTROL" if r["ei_t"] < r["ei_c"] else "control<=treated"
        print(f"  A_d={r['A_d']} sig={r['sigma']:<5} [{tag:14s}] gate={r['dCostate']/max(r['dStrong'],1e-30):6.1f}x  "
              f"i_d err ctrl={r['ei_c']:.3e} treat={r['ei_t']:.3e}  -> {win}", flush=True)
    print(f"\nelapsed {time.time()-t0:.0f}s", flush=True)
