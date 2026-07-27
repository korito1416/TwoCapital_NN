"""
DECISIVE TEST: does the validated NN's value function actually satisfy the HJB?
The FOC-check (~6e-5) only ties controls to the value GRADIENT; it never checks the value
PDE. Here we compute the NN's TRUE HJB residual via autodiff (exact 1st+2nd derivatives) and
compare to the FD's. lambda3=1/6, xi=148.4. Also reports the per-term breakdown and vlK.
"""
import os, sys
import numpy as np
import tensorflow as tf

import plot_pretrained_climate as C
from params import PARAMS

p = dict(delta=PARAMS["δ"], A_d=PARAMS["A_d"], A_gpp=PARAMS["A_g_prime_prime"],
         a_d=PARAMS["α_d"], G_d=PARAMS["Γ_d"], t_d=PARAMS["θ_d"], s_d=PARAMS["σ_d"],
         a_g=PARAMS["α_g"], G_g=PARAMS["Γ_g"], t_g=PARAMS["θ_g"], s_g=PARAMS["σ_g"],
         thbar=PARAMS["θ_bar"], eta=PARAMS["η"], vars=PARAMS["ϛ"],
         l1=PARAMS["λ1"], l2=PARAMS["λ2"], y_up=PARAMS["y_upper"])
LAM3, XI, LOGXI = 1/6.0, 148.4, 5.0


def nn_residual(net, logK, Z, Y):
    """NN HJB residual at points (logK,Z,Y) [each (n,1)] via autodiff. Returns R and parts."""
    v_nn, i_d_nn, i_g_nn = net
    n = logK.shape[0]
    lK = tf.constant(logK, tf.float32); Zt = tf.constant(Z, tf.float32); Yt = tf.constant(Y, tf.float32)
    l3 = tf.constant(np.full((n, 1), LAM3), tf.float32)
    Ag = tf.constant(np.full((n, 1), p["A_gpp"]), tf.float32)
    lx = tf.constant(np.full((n, 1), LOGXI), tf.float32)
    with tf.GradientTape(persistent=True) as t2:
        t2.watch([lK, Zt, Yt])
        with tf.GradientTape(persistent=True) as t1:
            t1.watch([lK, Zt, Yt])
            X = tf.concat([lK, Zt, Yt, l3, Ag, lx, lx], 1)
            v = v_nn(X, training=False)
        vlK = t1.gradient(v, lK); vZ = t1.gradient(v, Zt); vY = t1.gradient(v, Yt)
    vKK = t2.gradient(vlK, lK); vZZ = t2.gradient(vZ, Zt); vYY = t2.gradient(vY, Yt)
    vKZ = t2.gradient(vlK, Zt)
    del t1, t2
    X = tf.concat([lK, Zt, Yt, l3, Ag, lx, lx], 1)
    i_d = i_d_nn(X, training=False); i_g = i_g_nn(X, training=False)
    g = lambda a: a.numpy().ravel()
    return _resid(g(v), g(vlK), g(vZ), g(vY), g(vKK), g(vZZ), g(vYY), g(vKZ),
                  g(i_d), g(i_g), logK.ravel(), Z.ravel(), Y.ravel())


def _resid(v, vlK, vZ, vY, vKK, vZZ, vYY, vKZ, i_d, i_g, logK, Z, Y):
    K = np.exp(logK); E = p["eta"] * p["A_d"] * (1 - Z) * K
    sd2, sg2 = p["s_d"]**2, p["s_g"]**2
    c = (p["A_d"] - i_d) * (1 - Z) + (p["A_gpp"] - i_g) * Z
    phid = p["a_d"] + p["G_d"] * np.log(np.maximum(1 + p["t_d"] * i_d, 1e-9))
    phig = p["a_g"] + p["G_g"] * np.log(np.maximum(1 + p["t_g"] * i_g, 1e-9))
    Dc = sd2 * (1 - Z)**2 + sg2 * Z**2
    a_lK = (1 - Z) * phid + Z * phig - Dc / 2; b_lK = Dc / 2
    a_Z = Z * (1 - Z) * (phig - phid + (1 - Z) * sd2 - Z * sg2); b_Z = 0.5 * Z**2 * (1 - Z)**2 * (sd2 + sg2)
    cross = -Z * (1 - Z)**2 * sd2 + Z**2 * (1 - Z) * sg2
    a_Y = p["thbar"] * E; b_Y = 0.5 * p["vars"]**2 * E**2
    lNy = p["l1"] + p["l2"] * Y + LAM3 * (Y - p["y_up"]); lNyy = p["l2"] + LAM3
    qd = vlK - Z * vZ; qg = vlK + (1 - Z) * vZ
    E_d = (1 - Z) * p["s_d"] * qd; E_g = Z * p["s_g"] * qg; E_y = p["vars"] * E * (vY - lNy)
    robust = -0.5 / XI * (E_d**2 + E_g**2 + E_y**2)
    damage = -(lNy * a_Y + lNyy * b_Y)
    flow = p["delta"] * (np.log(np.maximum(c, 1e-12)) + logK)
    parts = dict(flow=flow, disc=-p["delta"] * v, lK_drift=a_lK * vlK, lK_diff=b_lK * vKK,
                 Z_drift=a_Z * vZ, Z_diff=b_Z * vZZ, cross=cross * vKZ,
                 Y_drift=a_Y * vY, Y_diff=b_Y * vYY, robust=robust, damage=damage)
    R = sum(parts.values())
    return R, parts, dict(vlK=vlK, c=c, i_d=i_d, i_g=i_g)


def main():
    net = C.build_and_load()
    print(f"[NN HJB check] lambda3={LAM3:.3f}, xi={XI}", flush=True)
    # interior grid (away from boundaries)
    g = lambda lo, hi, n: np.linspace(lo, hi, n)
    LKv, Zv, Yv = g(4.5, 6.8, 20), g(0.2, 0.8, 20), g(0.3, 3.5, 20)
    LK, ZZ, YY = np.meshgrid(LKv, Zv, Yv, indexing="ij")
    pts = (LK.reshape(-1, 1), ZZ.reshape(-1, 1), YY.reshape(-1, 1))
    R, parts, info = nn_residual(net, *pts)
    aR = np.abs(R)
    print(f"\n=== NN HJB residual (autodiff, {len(R)} interior pts) ===", flush=True)
    print(f"  max={aR.max():.3e}  p99={np.percentile(aR,99):.3e}  median={np.median(aR):.3e}  mean={aR.mean():.3e}", flush=True)
    print("  per-term RMS magnitude (which terms dominate the residual):", flush=True)
    for k, val in sorted(parts.items(), key=lambda kv: -np.sqrt(np.mean(kv[1]**2))):
        print(f"    {k:10s} rms={np.sqrt(np.mean(val**2)):.4f}", flush=True)
    # at the reference point
    R0, _, info0 = nn_residual(net, np.array([[np.log(880)]]), np.array([[0.7]]), np.array([[3.0]]))
    print(f"\n  at logK=6.78,Z=0.7,Y=3.0: NN |HJB residual|={abs(R0[0]):.3e}  vlK={info0['vlK'][0]:.3f} "
          f"i_d={info0['i_d'][0]:+.4f} c={info0['c'][0]:.4f}", flush=True)
    print("  (NN FOC residual at this pt was ~6e-6; FD HJB residual ~4.6e-3, vlK_FD~0.33, i_d_FD~-0.018)", flush=True)
    print("\n  VERDICT: if NN |HJB residual| << 4.6e-3 -> NN solves the HJB (NN right, FD has the bug);", flush=True)
    print("           if NN |HJB residual| >> 4.6e-3 -> NN is only FOC-consistent, NOT an HJB solution (NN wrong).", flush=True)


if __name__ == "__main__":
    main()
