"""FALSIFIABILITY CHECK for the Feynman-Kac level-identification route.

The claim under test: the value LEVEL can be identified by simulating the model's own policy and
matching the realised discounted payoff, instead of inferring it from a residual that only sees the
level through delta = 0.01.

The route dies immediately if EITHER of these fails, so measure them BEFORE building anything:

  (a) HORIZON.  delta = 0.01 => discount half-life 69 years.  Truncating the integral at T years
      leaves a bias ~ exp(-delta*T)*V:  T=300 -> ~0.2 (same size as the xi-effect we report, useless),
      T=600 -> ~0.01 (acceptable).  So we must either simulate ~600+ years, or close the integral with
      an analytically-known terminal value.  The S1 finding gives one: the absorbing Z=1 full-green
      BGP where v = logK + W0 in closed form.  QUESTION: do paths actually REACH that corner, and how
      fast?  If arrival takes far longer than the horizon we can afford, the terminal value cannot be
      used and the truncation bias sinks the method.

  (b) NOISE.  The Monte-Carlo standard error of the level estimate must be SMALLER than the quantity
      it is meant to pin -- the cross-regime gap / xi-effect, ~0.01.  If the MC standard error is
      >> 0.01, the estimate cannot discriminate.

FK REPRESENTATION USED (robust control, so the expectation is under the WORST-CASE measure and the
penalty is part of the flow):
    V(x0) = E^{h*}[ \int_0^inf e^{-delta t} ( delta*(log(C/K) + logK - logN(Y)) + xi|h|^2/2 ) dt ]
Distorted drifts (independent Brownians, so dW -> dW + h dt):
    dlogK += [(1-Z) sigma_d h_d + Z sigma_g h_g] dt
    dZ    += Z(1-Z)[sigma_g h_g - sigma_d h_d] dt
    dY     = E (thetabar + varsigma h_y) dt + E varsigma dW_y,   E = eta A_d (1-Z) K
"""
import os, sys, glob, json
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
sys.path.insert(0, f"{ROOT}/models_onejump_redesign")
import tensorflow as tf
from params import PARAMS
from feedforward_subnet import FeedForwardSubNet

P = PARAMS
δ, σ_d, σ_g = P["δ"], P["σ_d"], P["σ_g"]
α_d, Γ_d, θ_d = P["α_d"], P["Γ_d"], P["θ_d"]
α_g, Γ_g, θ_g = P["α_g"], P["Γ_g"], P["θ_g"]
A_d, AG2 = P["A_d"], P["A_g_prime_prime"]
η, ϛ, θ_bar = P["η"], P["ϛ"], P["θ_bar"]
λ1, λ2, YHAT = P["λ1"], P["λ2"], P["y_upper"]
SUB = "PostDamagePostTech"


def nets(run_dir):
    from params import investment_rate_activation
    cfg = lambda n, a, f: {"num_hiddens": [32]*4, "use_bias": True, "activation": a,
                           "dim": 1, "nn_name": n, "final_activation": f}
    out = {}
    for nm, a, f in [("v_nn", "swish", "softplus"),
                     ("i_g_nn", "tanh", investment_rate_activation(θ_g)),
                     ("i_d_nn", "tanh", investment_rate_activation(θ_d))]:
        net = FeedForwardSubNet(cfg(nm, a, f)); net.build((None, 7))
        ck = os.path.join(run_dir, SUB, f"{nm}_checkpoint_{SUB}")
        if not glob.glob(ck + "*"):
            return None
        net.load_weights(ck).expect_partial(); out[nm] = net
    return out


def inputs(logK, Z, Y, l3, lx):
    n = logK.shape[0]
    o = tf.ones([n, 1], tf.float32)
    return tf.concat([logK, Z, Y, o*l3, o*AG2, o*lx, o*lx], 1)


def simulate(run_dir, xi=0.05, lam3=1/6., n_paths=2048, years=800, dt=0.25, seed=7):
    net = nets(run_dir)
    if net is None:
        raise SystemExit(f"no checkpoints in {run_dir}")
    lx = float(np.log(xi))
    rng = np.random.default_rng(seed)
    n, steps = n_paths, int(years / dt)
    logK = tf.constant(np.full((n, 1), 6.7799), tf.float32)
    Z = tf.constant(np.full((n, 1), 0.70), tf.float32)
    Y = tf.constant(np.full((n, 1), YHAT), tf.float32)

    disc_payoff = np.zeros(n)
    hit_time = np.full(n, np.nan)      # first time Z >= 0.99
    sq = np.sqrt(dt)
    for k in range(steps):
        t = k * dt
        x = inputs(logK, Z, Y, lam3, lx)
        with tf.GradientTape(persistent=True) as tp:
            tp.watch([logK, Z, Y])
            xin = inputs(logK, Z, Y, lam3, lx)
            v = net["v_nn"](xin, training=False)
        dvK = tp.gradient(v, logK); dvZ = tp.gradient(v, Z); dvY = tp.gradient(v, Y)
        del tp
        i_g = net["i_g_nn"](x, training=False); i_d = net["i_d_nn"](x, training=False)

        Zn = Z.numpy(); Kn = np.exp(logK.numpy()); Yn = Y.numpy()
        ig, idd = i_g.numpy(), i_d.numpy()
        dvKn, dvZn, dvYn = dvK.numpy(), dvZ.numpy(), dvY.numpy()

        # worst-case drift distortions (theta = 1/xi)
        th = 1.0 / xi
        s_d = (dvKn - Zn*dvZn) * (1-Zn) * σ_d
        s_g = (dvKn + (1-Zn)*dvZn) * Zn * σ_g
        E = η * A_d * (1-Zn) * Kn
        logN_Y = λ1 + λ2*Yn + lam3*(Yn - YHAT)
        s_y = (dvYn - logN_Y) * E * ϛ
        h_d, h_g, h_y = -th*s_d, -th*s_g, -th*s_y

        c = (A_d - idd)*(1-Zn) + (AG2 - ig)*Zn
        c = np.maximum(c, 1e-8)
        logN = λ1*Yn + 0.5*λ2*Yn**2 + 0.5*lam3*np.maximum(Yn-YHAT, 0)**2
        pen = 0.5*th*(s_d**2 + s_g**2 + s_y**2)          # xi|h|^2/2 in finite theta form
        U = δ*(np.log(c) + logK.numpy() - logN) + pen
        disc_payoff += (np.exp(-δ*t) * U * dt).ravel()

        # drifts
        φd = α_d + Γ_d*np.log(np.maximum(1+θ_d*idd, 1e-8))
        φg = α_g + Γ_g*np.log(np.maximum(1+θ_g*ig, 1e-8))
        vk = (σ_d**2*(1-Zn)**2 + σ_g**2*Zn**2)/2.0
        muK = φd*(1-Zn) + φg*Zn - vk + ((1-Zn)*σ_d*h_d + Zn*σ_g*h_g)
        muZ = Zn*(1-Zn)*(φg - φd - Zn*σ_g**2 + (1-Zn)*σ_d**2) \
              + Zn*(1-Zn)*(σ_g*h_g - σ_d*h_d)
        muY = E*(θ_bar + ϛ*h_y)

        dWd = rng.standard_normal((n, 1))*sq; dWg = rng.standard_normal((n, 1))*sq
        dWy = rng.standard_normal((n, 1))*sq
        nlogK = logK.numpy() + muK*dt + (1-Zn)*σ_d*dWd + Zn*σ_g*dWg
        nZ = Zn + muZ*dt + Zn*(1-Zn)*(σ_g*dWg - σ_d*dWd)
        nY = Yn + muY*dt + E*ϛ*dWy

        newly = (np.isnan(hit_time)) & (nZ.ravel() >= 0.99)
        hit_time[newly] = t + dt

        logK = tf.constant(np.clip(nlogK, 2.0, 12.0), tf.float32)
        Z = tf.constant(np.clip(nZ, 1e-4, 1-1e-6), tf.float32)
        Y = tf.constant(np.clip(nY, 0.0, 12.0), tf.float32)

    return disc_payoff, hit_time, float(np.exp(-δ*years))


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/output_redesign_20260726/A0_baseline_seed1"
    years = float(os.environ.get("FK_YEARS", 800))
    npaths = int(os.environ.get("FK_PATHS", 2048))
    xi = float(os.environ.get("FK_XI", 0.05))
    print(f"FK feasibility | run={os.path.basename(run)} xi={xi} paths={npaths} horizon={years}y")
    pay, hit, tail = simulate(run, xi=xi, n_paths=npaths, years=years)

    net = nets(run)
    x0 = inputs(tf.constant([[6.7799]], tf.float32), tf.constant([[0.70]], tf.float32),
                tf.constant([[YHAT]], tf.float32), 1/6., float(np.log(xi)))
    v_claim = float(net["v_nn"](x0, training=False)[0, 0])
    logN0 = λ1*YHAT + 0.5*λ2*YHAT**2
    V_claim = v_claim - logN0

    fk, se = pay.mean(), pay.std(ddof=1)/np.sqrt(len(pay))
    reached = np.mean(~np.isnan(hit))
    print(f"\n(a) HORIZON")
    print(f"    paths reaching Z>=0.99 within {years:.0f}y : {100*reached:.1f}%")
    if reached > 0:
        ht = hit[~np.isnan(hit)]
        print(f"    arrival time (yrs)  median={np.median(ht):.0f}  p10={np.percentile(ht,10):.0f}"
              f"  p90={np.percentile(ht,90):.0f}")
    print(f"    truncation weight exp(-delta*T) = {tail:.4f}  -> bias ~ {tail*abs(V_claim):.4f}")
    print(f"\n(b) NOISE")
    print(f"    FK estimate  = {fk:.4f}   MC s.e. = {se:.4f}   (need << 0.01)")
    print(f"    claimed V(x0)= {V_claim:.4f}   gap = {fk - V_claim:+.4f}")
    print(f"\nVERDICT")
    ok_n = se < 0.01
    ok_h = (tail*abs(V_claim)) < 0.01 or reached > 0.9
    print(f"    noise gate  : {'PASS' if ok_n else 'FAIL'} (s.e. {se:.4f} vs 0.01)")
    print(f"    horizon gate: {'PASS' if ok_h else 'FAIL'} (bias {tail*abs(V_claim):.4f}, reached {100*reached:.0f}%)")
    print(f"    => FK route {'VIABLE so far' if (ok_n and ok_h) else 'BLOCKED on the failing gate'}")
    json.dump({"fk": float(fk), "se": float(se), "V_claim": float(V_claim),
               "reached_frac": float(reached), "tail": tail, "years": years},
              open(os.path.join(run, "fk_feasibility.json"), "w"), indent=2)
