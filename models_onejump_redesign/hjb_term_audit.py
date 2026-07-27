"""Term-by-term magnitude audit of the HJB: which parts of the economy dynamics does the solver
actually SEE, and which sit below its own error floor?

Motivation.  The training residual is ~2e-3.  Any term whose contribution is far below that is
NUMERICALLY INVISIBLE: the network cannot be resolving it, so the economics that term encodes is not
actually being solved for.  The second-order (diffusion) coefficients are tiny at the calibrated
volatilities --

    ½ς²E²                 2.50e-04     (temperature)
    σ_κ²/2                3.04e-05     (knowledge)
    (σ_d²(1-Z)²+σ_g²Z²)/2 2.90e-05     (capital)
    −Z(1-Z)²σ_d²+Z²(1-Z)σ_g²  8.40e-06 (capital-share cross)
    ½Z²(1-Z)²(σ_g²+σ_d²)  4.41e-06     (share)

-- so unless the second derivatives are enormous (|V_KK| ~ 70, |V_ZZ| ~ 450 to reach 2e-3), the whole
diffusion block is below the floor.  This script measures the ACTUAL contributions on a trained net
and ranks every term, so the claim is measured rather than argued.
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
σ_κ, η, ϛ, θ_bar = P["σ_κ"], P["η"], P["ϛ"], P["θ_bar"]
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


def audit(run, xi=0.05, l3=1/6., n=2000, seed=5):
    net = load(run)
    lx = float(np.log(xi))
    r = np.random.default_rng(seed)
    logK = tf.Variable(r.uniform(4.0, 7.0, (n, 1)).astype(np.float32))
    Z = tf.Variable(r.uniform(0.05, 0.95, (n, 1)).astype(np.float32))
    Y = tf.Variable(r.uniform(2.5, 4.0, (n, 1)).astype(np.float32))

    def inp():
        o = tf.ones_like(logK)
        return tf.concat([logK, Z, Y, o*l3, o*AG2, o*lx, o*lx], 1)

    with tf.GradientTape(persistent=True) as t2:
        t2.watch([logK, Z, Y])
        with tf.GradientTape(persistent=True) as t1:
            t1.watch([logK, Z, Y])
            v = net["v_nn"](inp(), training=False)
        dK = t1.gradient(v, logK); dZ = t1.gradient(v, Z); dY = t1.gradient(v, Y)
    dKK = t2.gradient(dK, logK); dZZ = t2.gradient(dZ, Z); dYY = t2.gradient(dY, Y)
    dKZ = t2.gradient(dK, Z)
    del t1, t2

    x = inp()
    i_g = net["i_g_nn"](x, training=False).numpy(); i_d = net["i_d_nn"](x, training=False).numpy()
    Zn, Kn, Yn = Z.numpy(), np.exp(logK.numpy()), Y.numpy()
    vN, dKn, dZn, dYn = v.numpy(), dK.numpy(), dZ.numpy(), dY.numpy()
    dKKn, dZZn, dYYn, dKZn = dKK.numpy(), dZZ.numpy(), dYY.numpy(), dKZ.numpy()

    th = 1.0/xi
    E = η*A_d*(1-Zn)*Kn
    φd = α_d + Γ_d*np.log(np.maximum(1+θ_d*i_d, 1e-8))
    φg = α_g + Γ_g*np.log(np.maximum(1+θ_g*i_g, 1e-8))
    vKK_c = (σ_d**2*(1-Zn)**2 + σ_g**2*Zn**2)/2.0
    muK = φd*(1-Zn) + φg*Zn - vKK_c
    muZ = (φg - φd - Zn*σ_g**2 + (1-Zn)*σ_d**2)*Zn*(1-Zn)
    vZZ_c = 0.5*Zn**2*(1-Zn)**2*(σ_g**2+σ_d**2)
    vKZ_c = -Zn*(1-Zn)**2*σ_d**2 + Zn**2*(1-Zn)*σ_g**2
    vYY_c = 0.5*ϛ**2*E**2
    logN_Y = λ1 + λ2*Yn + l3*(Yn-YHAT)
    s_d = (dKn - Zn*dZn)*(1-Zn)*σ_d
    s_g = (dKn + (1-Zn)*dZn)*Zn*σ_g
    s_y = (dYn - logN_Y)*E*ϛ
    h_y = -th*s_y
    c = np.maximum((A_d - i_d)*(1-Zn) + (AG2 - i_g)*Zn, 1e-10)

    terms = {
        "flow  δ(log(C/K)+logK)":      δ*(np.log(c) + logK.numpy()),
        "-δV":                          -δ*vN,
        "drift_logK · V_logK":          muK*dKn,
        "drift_Z · V_Z":                muZ*dZn,
        "climate  V_Y·(θ̄+ςh_y)·E":     dYn*((θ_bar + ϛ*h_y)*E),
        "damage  -(logN)_Y·(...)E":     -(logN_Y)*((θ_bar + ϛ*h_y)*E),
        "robust penalty  θ|s|²/2":      0.5*th*(s_d**2 + s_g**2 + s_y**2),
        "2nd: V_KK · coef":             vKK_c*dKKn,
        "2nd: V_ZZ · coef":             vZZ_c*dZZn,
        "2nd: V_KZ · coef":             vKZ_c*dKZn,
        "2nd: V_YY · coef":             vYY_c*dYYn,
        "2nd: damage -(λ2+λ3)·½ς²E²":   -(λ2+l3)*vYY_c*np.ones_like(vYY_c),
    }
    print(f"run={os.path.basename(run)}  xi={xi}  n={n}\n")
    print(f"{'HJB term':34}{'mean':>12}{'RMS':>12}{'max|·|':>12}")
    order = sorted(terms.items(), key=lambda kv: -np.sqrt((kv[1]**2).mean()))
    for k, val in order:
        print(f"  {k:32}{val.mean():12.3e}{np.sqrt((val**2).mean()):12.3e}{np.abs(val).max():12.3e}")

    second = sum(np.sqrt((terms[k]**2).mean()) for k in terms if k.startswith("2nd"))
    first = sum(np.sqrt((terms[k]**2).mean()) for k in terms if not k.startswith("2nd"))
    print(f"\n  SUM of RMS over 2nd-order (diffusion) terms : {second:.3e}")
    print(f"  SUM of RMS over 0th/1st-order terms         : {first:.3e}")
    print(f"  ratio 2nd/1st                               : {second/first:.2e}")
    print(f"  training residual (this run)                : ~2.0e-03")
    print(f"\n  => the diffusion block is {'BELOW' if second < 2e-3 else 'ABOVE'} the solver's own error floor")
    print(f"     second derivatives measured: |V_KK|~{np.abs(dKKn).mean():.3f}  |V_ZZ|~{np.abs(dZZn).mean():.3f}"
          f"  |V_YY|~{np.abs(dYYn).mean():.3f}  |V_KZ|~{np.abs(dKZn).mean():.3f}")
    print(f"     to reach 2e-3 they would need: |V_KK|~{2e-3/vKK_c.mean():.0f}  |V_ZZ|~{2e-3/vZZ_c.mean():.0f}"
          f"  |V_YY|~{2e-3/vYY_c.mean():.1f}")


if __name__ == "__main__":
    audit(sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/output_redesign_20260726/A0_baseline_seed1",
          xi=float(os.environ.get("AUDIT_XI", 0.05)))
