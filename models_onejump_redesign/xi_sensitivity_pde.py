"""Residual of the xi-SENSITIVITY PDE, obtained by differentiating the HJB w.r.t. theta = 1/xi.

DERIVATION (terminal regime, no jumps).  Substituting the closed-form worst-case drift h* = -theta*s
collapses every distortion term into a single quadratic, so the HJB residual is

    R = N[v] - delta*v - (theta/2)||s||^2,      s = (s_d, s_g, s_y)

with N[.] the NEUTRAL (baseline-drift) operator and s the sigma'-grad-v loadings.  Differentiating
R = 0 in theta and invoking the envelope theorem (which kills d(alpha)/d(theta) at the optimum) gives
a LINEAR PDE for v_theta = dv/dtheta:

    N[v_theta] - delta*v_theta - theta*(s . s_theta) - (1/2)||s||^2 = 0

where s_theta is s with grad-v replaced by grad-v_theta.  The source term -(1/2)||s||^2 is the SAME
ORDER as the robustness drag itself, so v_theta is determined at its own natural scale instead of
being inferred from a residual it perturbs by ~0.05%.

WHY MEASURE IT.  The pointwise FOC is satisfied to ~0.05% relative, but the xi-DERIVATIVE of the FOC
is violated by 5-8% (measured).  If the same is true here -- the HJB residual small, its theta
derivative large -- then the xi-direction is effectively unconstrained by the current objective, and
this PDE is the missing condition.

CONVERSION.  The network input is logxi and theta = exp(-logxi), so
    d/dtheta = -(1/theta) d/d(logxi).
"""
import os, sys, glob
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
sys.path.insert(0, f"{ROOT}/models_onejump_redesign")
import tensorflow as tf
from params import PARAMS as P, investment_rate_activation
from feedforward_subnet import FeedForwardSubNet

δ = P["δ"]; A_d, AG2 = P["A_d"], P["A_g_prime_prime"]
α_d, Γ_d, θ_d, σ_d = P["α_d"], P["Γ_d"], P["θ_d"], P["σ_d"]
α_g, Γ_g, θ_g, σ_g = P["α_g"], P["Γ_g"], P["θ_g"], P["σ_g"]
η, ϛ, θ_bar = P["η"], P["ϛ"], P["θ_bar"]
λ1, λ2, YHAT = P["λ1"], P["λ2"], P["y_upper"]
SUB = "PostDamagePostTech"


def load(run):
    cfg = lambda n, a, f: {"num_hiddens": [32]*4, "use_bias": True, "activation": a,
                           "dim": 1, "nn_name": n, "final_activation": f}
    out = {}
    for nm, a, f in [("v_nn", "swish", "softplus"),
                     ("i_g_nn", "tanh", investment_rate_activation(θ_g)),
                     ("i_d_nn", "tanh", investment_rate_activation(θ_d))]:
        net = FeedForwardSubNet(cfg(nm, a, f)); net.build((None, 7))
        ck = os.path.join(run, SUB, f"{nm}_checkpoint_{SUB}")
        if not glob.glob(ck + "*"):
            return None
        net.load_weights(ck).expect_partial(); out[nm] = net
    return out


def analyse(run, xi=0.1, l3=1/6., n=1500, seed=17):
    net = load(run)
    th = 1.0 / xi
    r = np.random.default_rng(seed)
    lk = tf.Variable(r.uniform(4.0, 7.0, (n, 1)).astype(np.float32))
    z = tf.Variable(r.uniform(0.10, 0.90, (n, 1)).astype(np.float32))
    y = tf.Variable(r.uniform(2.5, 4.0, (n, 1)).astype(np.float32))
    lxi = tf.Variable(np.full((n, 1), np.log(xi), np.float32))

    def fwd(a, b, c, lx):
        o = tf.ones_like(a)
        return net["v_nn"](tf.concat([a, b, c, o*l3, o*AG2, lx, lx], 1), training=False)

    # v and its state derivatives, AND the same for v_logxi (one extra tape level)
    with tf.GradientTape(persistent=True) as T3:
        T3.watch([lk, z, y, lxi])
        with tf.GradientTape(persistent=True) as T2:
            T2.watch([lk, z, y, lxi])
            with tf.GradientTape(persistent=True) as T1:
                T1.watch([lk, z, y, lxi])
                v = fwd(lk, z, y, lxi)
            dK = T1.gradient(v, lk); dZ = T1.gradient(v, z); dY = T1.gradient(v, y)
            dX = T1.gradient(v, lxi)
        dKK = T2.gradient(dK, lk); dZZ = T2.gradient(dZ, z); dYY = T2.gradient(dY, y)
        dKZ = T2.gradient(dK, z)
        # derivatives of v_logxi
        xK = T2.gradient(dX, lk); xZ = T2.gradient(dX, z); xY = T2.gradient(dX, y)
    xKK = T3.gradient(xK, lk); xZZ = T3.gradient(xZ, z); xYY = T3.gradient(xY, y)
    xKZ = T3.gradient(xK, z)
    del T1, T2, T3

    N = lambda t: t.numpy()
    Zn, Kn, Yn = N(z), np.exp(N(lk)), N(y)
    x = tf.concat([lk, z, y, tf.ones_like(lk)*l3, tf.ones_like(lk)*AG2, lxi, lxi], 1)
    i_g = N(net["i_g_nn"](x, training=False)); i_d = N(net["i_d_nn"](x, training=False))

    # theta-derivatives:  d/dtheta = -(1/theta) d/dlogxi
    f = -1.0/th
    vt, vtK, vtZ, vtY = f*N(dX), f*N(xK), f*N(xZ), f*N(xY)
    vtKK, vtZZ, vtYY, vtKZ = f*N(xKK), f*N(xZZ), f*N(xYY), f*N(xKZ)
    dKn, dZn, dYn = N(dK), N(dZ), N(dY)

    # coefficients of the NEUTRAL operator (frozen controls -- envelope theorem)
    E = η*A_d*(1-Zn)*Kn
    φd = α_d + Γ_d*np.log(np.maximum(1+θ_d*i_d, 1e-8))
    φg = α_g + Γ_g*np.log(np.maximum(1+θ_g*i_g, 1e-8))
    cKK = (σ_d**2*(1-Zn)**2 + σ_g**2*Zn**2)/2.0
    muK = φd*(1-Zn) + φg*Zn - cKK
    muZ = (φg - φd - Zn*σ_g**2 + (1-Zn)*σ_d**2)*Zn*(1-Zn)
    cZZ = 0.5*Zn**2*(1-Zn)**2*(σ_g**2+σ_d**2)
    cKZ = -Zn*(1-Zn)**2*σ_d**2 + Zn**2*(1-Zn)*σ_g**2
    cYY = 0.5*ϛ**2*E**2
    lnY = λ1 + λ2*Yn + l3*(Yn-YHAT)

    def Nop(a, aK, aZ, aY, aKK, aZZ, aYY, aKZ):
        """Neutral (baseline-drift) operator applied to a field."""
        return (muK*aK + cKK*aKK + muZ*aZ + cZZ*aZZ + cKZ*aKZ
                + aY*θ_bar*E + cYY*aYY)

    # loadings s (regime-correct V_Y inside s_y), and s_theta with grad v -> grad v_theta
    s_d = (dKn - Zn*dZn)*(1-Zn)*σ_d
    s_g = (dKn + (1-Zn)*dZn)*Zn*σ_g
    s_y = (dYn - lnY)*E*ϛ
    t_d = (vtK - Zn*vtZ)*(1-Zn)*σ_d
    t_g = (vtK + (1-Zn)*vtZ)*Zn*σ_g
    t_y = (vtY)*E*ϛ                     # (logN)_Y is theta-independent

    c = np.maximum((A_d - i_d)*(1-Zn) + (AG2 - i_g)*Zn, 1e-10)
    flow = δ*(np.log(c) + N(lk))
    base = Nop(N(v), dKn, dZn, dYn, N(dKK), N(dZZ), N(dYY), N(dKZ)) \
        - lnY*θ_bar*E - (λ2+l3)*cYY
    R_hjb = flow + base - δ*N(v) - 0.5*th*(s_d**2 + s_g**2 + s_y**2)

    # SENSITIVITY PDE residual
    src = -0.5*(s_d**2 + s_g**2 + s_y**2)
    R_sens = (Nop(vt, vtK, vtZ, vtY, vtKK, vtZZ, vtYY, vtKZ) - δ*vt
              - th*(s_d*t_d + s_g*t_g + s_y*t_y) + src)

    rms = lambda a: float(np.sqrt((a**2).mean()))
    print(f"run={os.path.basename(run)}  xi={xi} (theta={th:.1f})  n={n}\n")
    print(f"  HJB residual                 RMS = {rms(R_hjb):.4e}")
    print(f"  SENSITIVITY-PDE residual     RMS = {rms(R_sens):.4e}")
    print(f"  its source term  -||s||^2/2  RMS = {rms(src):.4e}")
    print(f"  v_theta                      RMS = {rms(vt):.4e}   mean = {vt.mean():+.4e}")
    print(f"\n  RELATIVE violation of the sensitivity PDE = {rms(R_sens)/max(rms(src),1e-30):.1%}"
          f"   (residual / its own source)")
    print(f"  for comparison, HJB residual / its dominant term (delta*v ~ {rms(δ*N(v)):.3e})"
          f" = {rms(R_hjb)/rms(δ*N(v)):.2%}")
    return dict(hjb=rms(R_hjb), sens=rms(R_sens), src=rms(src), vt=rms(vt))


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/output_redesign_20260726/A0_baseline_seed1"
    for xi in [float(x) for x in os.environ.get("SENS_XIS", "1.0,0.1,0.05").split(",")]:
        analyse(run, xi=xi)
        print("-"*78)
