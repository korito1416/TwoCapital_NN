"""Recover the value gradient from the CONTROLS by inverting the FOCs, then integrate it back to V.

IDEA.  The investment FOCs are invertible in the marginal values:

    MU = delta/(C/K),
    q_d = MU / phi_d'(i_d) = MU (1 + theta_d i_d)/(Gamma_d theta_d)
    q_g = MU / phi_g'(i_g) = MU (1 + theta_g i_g)/(Gamma_g theta_g)

and the coordinate transform inverts too:

    V_logK = Z q_g + (1-Z) q_d          V_Z = q_g - q_d

So the POLICY NETWORKS ALONE determine the value gradient in the CONTROLLED directions -- no use of
the value network's own derivatives.  Three things this buys:

  1. SELF-CONSISTENCY TEST.  Compare the FOC-implied gradient with the value net's autodiff gradient.
     They are trained by different loss terms, so agreement is a real (not tautological) check.

  2. WHICH DIRECTIONS ARE PINNED BY WHAT.  Y is NOT a control, so no FOC constrains V_Y: the
     temperature direction is pinned by the HJB residual ALONE.  Likewise the LEVEL is pinned by the
     delta term alone.  Everything else (capital, knowledge) is over-determined (controls AND HJB).

  3. INTEGRATION.  Line-integrating the FOC-implied gradient recovers V up to ONE constant --
     a constructive proof that the level is exactly one scalar, and a way to rebuild the value SHAPE
     from the better-identified object (FOC residuals ~1e-4 vs HJB residual ~2e-3).

CONSERVATIVITY.  A recovered gradient field is only integrable if it is curl-free.  With the
FOC-implied field we can test d(V_logK)/dZ == d(V_Z)/dlogK pointwise; any violation bounds how much
of the value function CANNOT be reconstructed from the policy, no matter what anchor is used.
"""
import os, sys, glob, json
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
sys.path.insert(0, f"{ROOT}/models_onejump_redesign")
import tensorflow as tf
from params import PARAMS, investment_rate_activation
from feedforward_subnet import FeedForwardSubNet

P = PARAMS
δ = P["δ"]; A_d = P["A_d"]; AG2 = P["A_g_prime_prime"]
Γ_d, θ_d, Γ_g, θ_g = P["Γ_d"], P["θ_d"], P["Γ_g"], P["θ_g"]
YHAT = P["y_upper"]; λ1, λ2 = P["λ1"], P["λ2"]
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


def X(logK, Z, Y, l3, lx):
    o = tf.ones_like(logK)
    return tf.concat([logK, Z, Y, o*l3, o*AG2, o*lx, o*lx], 1)


def foc_gradient(net, logK, Z, Y, l3, lx):
    """Marginal values implied by the CONTROLS alone (never touches v_nn)."""
    x = X(logK, Z, Y, l3, lx)
    i_g = net["i_g_nn"](x, training=False).numpy()
    i_d = net["i_d_nn"](x, training=False).numpy()
    Zn = Z.numpy()
    cok = (A_d - i_d)*(1-Zn) + (AG2 - i_g)*Zn
    MU = δ / np.maximum(cok, 1e-10)
    q_d = MU * (1 + θ_d*i_d) / (Γ_d*θ_d)
    q_g = MU * (1 + θ_g*i_g) / (Γ_g*θ_g)
    return dict(q_g=q_g, q_d=q_d, V_logK=Zn*q_g + (1-Zn)*q_d, V_Z=q_g - q_d, cok=cok)


def nn_gradient(net, logK, Z, Y, l3, lx):
    """Marginal values from the VALUE net's autodiff gradient (the usual object)."""
    with tf.GradientTape(persistent=True) as tp:
        tp.watch([logK, Z, Y])
        v = net["v_nn"](X(logK, Z, Y, l3, lx), training=False)
    g = dict(V_logK=tp.gradient(v, logK).numpy(), V_Z=tp.gradient(v, Z).numpy(),
             v_Y=tp.gradient(v, Y).numpy(), v=v.numpy())
    del tp
    return g


def col(a):
    return tf.constant(np.asarray(a, np.float32).reshape(-1, 1), tf.float32)


def main(run, xi=0.05, l3=1/6.):
    net = load(run)
    if net is None:
        raise SystemExit(f"no checkpoints in {run}")
    lx = float(np.log(xi))
    rng = np.random.default_rng(11)
    n = 3000
    lk = col(rng.uniform(4.0, 7.0, n)); z = col(rng.uniform(0.05, 0.95, n))
    y = col(rng.uniform(2.5, 4.0, n))

    f = foc_gradient(net, lk, z, y, l3, lx)
    g = nn_gradient(net, lk, z, y, l3, lx)

    print(f"run={os.path.basename(run)}  xi={xi}  n={n}\n")
    print("=== 1. SELF-CONSISTENCY: gradient from CONTROLS vs gradient from VALUE NET ===")
    for k in ["V_logK", "V_Z"]:
        a, b = f[k].ravel(), g[k].ravel()
        rel = np.abs(a-b)/np.maximum(np.abs(b), 1e-8)
        print(f"  {k:8} FOC-implied mean={a.mean():8.5f}   NN mean={b.mean():8.5f}   "
              f"median|rel diff|={np.median(rel):.2e}   p90={np.percentile(rel,90):.2e}")

    print("\n=== 2. WHICH DIRECTIONS ARE PINNED BY A CONTROL? ===")
    print("  V_logK, V_Z : pinned by the i_d/i_g FOCs AND the HJB  -> over-determined")
    print("  V_Y         : NO control on temperature -> pinned by the HJB residual ALONE")
    print("  level       : pinned by the delta term ALONE")

    print("\n=== 3. CONSERVATIVITY of the FOC-implied field (is it integrable at all?) ===")
    # finite-difference cross derivatives on a small grid: d(V_logK)/dZ vs d(V_Z)/dlogK
    h = 1e-3
    lk0 = col(np.full(400, 6.0)); z0 = col(rng.uniform(0.15, 0.85, 400)); y0 = col(np.full(400, YHAT))
    def fg(a, b):
        return foc_gradient(net, a, b, y0, l3, lx)
    dVk_dZ = (fg(lk0, col(z0.numpy()+h))["V_logK"] - fg(lk0, col(z0.numpy()-h))["V_logK"])/(2*h)
    dVz_dK = (fg(col(lk0.numpy()+h), z0)["V_Z"] - fg(col(lk0.numpy()-h), z0)["V_Z"])/(2*h)
    curl = (dVk_dZ - dVz_dK).ravel()
    scale = np.maximum(np.abs(dVk_dZ).ravel(), 1e-8)
    print(f"  curl = d(V_logK)/dZ - d(V_Z)/dlogK :  mean|curl|={np.abs(curl).mean():.3e}   "
          f"median relative={np.median(np.abs(curl)/scale):.2e}")
    print("  (a curl-free field is exactly integrable; a large curl bounds what NO anchor can fix)")

    print("\n=== 4. INTEGRATE the FOC-implied gradient along logK, compare to the value net ===")
    zf, yf = 0.70, YHAT
    grid = np.linspace(4.0, 7.0, 121)
    lkg = col(grid); zg = col(np.full_like(grid, zf)); yg = col(np.full_like(grid, yf))
    dV = foc_gradient(net, lkg, zg, yg, l3, lx)["V_logK"].ravel()
    V_int = np.concatenate([[0.0], np.cumsum(0.5*(dV[1:]+dV[:-1])*np.diff(grid))])
    v_nn = nn_gradient(net, lkg, zg, yg, l3, lx)["v"].ravel()
    V_nn = v_nn - v_nn[0]                       # both anchored at logK=4 so only SHAPE is compared
    err = V_int - V_nn
    print(f"  shape recovered from CONTROLS vs value net, over logK in [4,7] at Z={zf}, Y={yf}:")
    print(f"    max|diff| = {np.abs(err).max():.5f}   rms = {np.sqrt((err**2).mean()):.5f}"
          f"   (total rise {V_nn[-1]:.4f})")
    print(f"    relative shape error = {np.abs(err).max()/max(abs(V_nn[-1]),1e-9)*100:.2f}%")
    json.dump({"max_shape_err": float(np.abs(err).max()), "total_rise": float(V_nn[-1]),
               "curl_mean_abs": float(np.abs(curl).mean())},
              open(os.path.join(run, "foc_inversion.json"), "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/output_redesign_20260726/A0_baseline_seed1",
         xi=float(os.environ.get("FOC_XI", 0.05)))
