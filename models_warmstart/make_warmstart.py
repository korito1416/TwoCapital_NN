"""Write a warm-start checkpoint set into an arm directory (warm-start RCT).

Modes
  perturb  : base run's nets + relative Gaussian noise on dense kernels
  analytic : v fitted to the closed-form guess v = logK + C0; investment rules
             fitted to constant reference rates (level put right from step 0)
  anchor   : v of every non-terminal jump state fitted (supervised) to the ARM'S
             OWN trained post-damage/post-tech value function; controls cold.
             Run AFTER the arm's PostDamagePostTech stage has finished.
  levelshift: base run's v fitted (supervised) to base_v(x) + delta -- same value
             SHAPE (marginal values) hence same correct policy, level moved by delta;
             control nets copied UNCHANGED. Probe for a welfare-level degree of freedom
             within the economically-correct basin. Requires --base and --delta.

The trainer then consumes the arm dir itself via pretrained_path (same pattern
as the 2026-07 perturb experiments). Honors MODEL_INIT_SEED for any cold nets.

Usage
  python make_warmstart.py --mode perturb  --out ARM --base BASE --noise 0.1 --seed 11
  python make_warmstart.py --mode analytic --out ARM [--c0 -2.57] [--seed 1]
  python make_warmstart.py --mode anchor   --out ARM [--seed 1]   (ARM must hold PostDamagePostTech)
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import tensorflow as tf
from feedforward_subnet import FeedForwardSubNet
from params import PARAMS, investment_rate_activation

REGS = ["PreDamagePreTech", "PreDamagePostTech", "PostDamagePreTech", "PostDamagePostTech"]
DIMS = {"PreDamagePreTech": 7, "PreDamagePostTech": 6, "PostDamagePreTech": 8, "PostDamagePostTech": 7}
NETS = {"PreDamagePreTech": ["v", "i_g", "i_d", "i_r"], "PreDamagePostTech": ["v", "i_g", "i_d"],
        "PostDamagePreTech": ["v", "i_g", "i_d", "i_r"], "PostDamagePostTech": ["v", "i_g", "i_d"]}
AGPP, L3 = 0.1567, 1.0 / 6.0
REF_RATES = {"i_g": 0.070, "i_d": 0.036, "i_r": 0.0067}   # RUNA t=0 investment rates

def _cfg(nm):
    act = {"v": ("swish", "softplus"),
           "i_g": ("tanh", investment_rate_activation(PARAMS["θ_g"])),
           "i_d": ("tanh", investment_rate_activation(PARAMS["θ_d"])),
           "i_r": ("softplus", "softplus")}[nm]
    return {"num_hiddens": [32] * 4, "use_bias": True, "activation": act[0],
            "dim": 1, "nn_name": f"{nm}_nn", "final_activation": act[1]}

def fresh(nm, dim):
    n = FeedForwardSubNet(_cfg(nm)); n(tf.zeros([1, dim])); return n

def load(root, reg, nm):
    n = fresh(nm, DIMS[reg])
    n.load_weights(f"{root}/{reg}/{nm}_nn_checkpoint_{reg}").expect_partial()
    return n

def save(net, out, reg, nm):
    os.makedirs(f"{out}/{reg}", exist_ok=True)
    net.save_weights(f"{out}/{reg}/{nm}_nn_checkpoint_{reg}")

def lhs(n, bounds, rng):
    return [((lo + (hi - lo) * (rng.permutation(n) + rng.rand(n)) / n)
             .reshape(-1, 1).astype(np.float32)) for lo, hi in bounds]

def X_of(reg, lk, Z, Y, lr, lx):
    ones = np.ones_like(Y)
    if reg == "PreDamagePreTech":   return np.hstack([lk, Z, Y, lr, lx, lx, lx])
    if reg == "PreDamagePostTech":  return np.hstack([lk, Z, Y, AGPP * ones, lx, lx])
    if reg == "PostDamagePreTech":  return np.hstack([lk, Z, Y, lr, L3 * ones, lx, lx, lx])
    if reg == "PostDamagePostTech": return np.hstack([lk, Z, Y, L3 * ones, AGPP * ones, lx, lx])

def fit(net, X, target, steps=5000, lr=1e-3, tag=""):
    opt = tf.keras.optimizers.Adam(lr)
    Xt, Tt = tf.constant(X), tf.constant(target.astype(np.float32))
    n = len(X)
    for it in range(steps):
        idx = np.random.randint(0, n, 1024)
        xb, tb = tf.gather(Xt, idx), tf.gather(Tt, idx)
        with tf.GradientTape() as tp:
            loss = tf.reduce_mean(tf.square(net(xb, training=True) - tb))
        opt.apply_gradients(zip(tp.gradient(loss, net.trainable_variables), net.trainable_variables))
        if it % 2000 == 0:
            print(f"    fit {tag} step {it}: mse={float(loss):.3e}")
    return net

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["perturb", "analytic", "anchor", "levelshift"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--c0", type=float, default=-2.57, help="v = logK + c0 analytic target")
    ap.add_argument("--delta", type=float, default=0.0,
                    help="levelshift: constant added to base value level")
    a = ap.parse_args()
    tf.random.set_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.RandomState(a.seed)

    if a.mode == "perturb":
        assert a.base, "--base required for perturb"
        for reg in REGS:
            for nm in NETS[reg]:
                net = load(a.base, reg, nm)
                for k in net.trainable_variables:
                    if "kernel" in k.name:
                        k.assign(k + a.noise * tf.math.reduce_std(k) * tf.random.normal(tf.shape(k)))
                save(net, a.out, reg, nm)
            print(f"  perturbed {reg} (noise={a.noise}, seed={a.seed})")
        return

    if a.mode == "levelshift":
        assert a.base, "--base required for levelshift"
        for reg in REGS:
            ylo = 2.5 if reg.startswith("PostDamage") else 0.0
            lk, Z, Y, lr_, l3s = lhs(16384, [(4, 7), (0.01, 0.99), (ylo, 4), (1, 6), (0, 1 / 3)], rng)
            lx = (np.log(0.05) + (np.log(148.6) - np.log(0.05)) * rng.rand(16384, 1)).astype(np.float32)
            X = X_of(reg, lk, Z, Y, lr_, lx)
            v_ref = load(a.base, reg, "v")
            target = v_ref(tf.constant(X), training=False).numpy() + a.delta
            v = fit(fresh("v", DIMS[reg]), X, target, tag=f"{reg}/v+{a.delta:+.2f}")
            save(v, a.out, reg, "v")
            for nm in NETS[reg][1:]:            # controls copied UNCHANGED (policy preserved)
                save(load(a.base, reg, nm), a.out, reg, nm)
            print(f"  levelshift {reg}: v <- base_v {a.delta:+.3f}, controls unchanged")
        return

    # shared LHS sample for the supervised fits
    for reg in REGS:
        ylo = 2.5 if reg.startswith("PostDamage") else 0.0
        lk, Z, Y, lr_, l3s = lhs(16384, [(4, 7), (0.01, 0.99), (ylo, 4), (1, 6), (0, 1 / 3)], rng)
        lx = (np.log(0.05) + (np.log(148.6) - np.log(0.05)) * rng.rand(16384, 1)).astype(np.float32)
        X = X_of(reg, lk, Z, Y, lr_, lx)

        if a.mode == "analytic":
            v_t = lk + a.c0
            v = fit(fresh("v", DIMS[reg]), X, v_t, tag=f"{reg}/v")
            save(v, a.out, reg, "v")
            for nm in NETS[reg][1:]:
                net = fit(fresh(nm, DIMS[reg]), X, np.full_like(lk, REF_RATES[nm]),
                          steps=2500, tag=f"{reg}/{nm}")
                save(net, a.out, reg, nm)
            print(f"  analytic init written for {reg}")

        elif a.mode == "anchor":
            if reg == "PostDamagePostTech":
                continue  # terminal state trains cold; it IS the anchor
            vpp = load(a.out, "PostDamagePostTech", "v")
            Xpp = X_of("PostDamagePostTech", lk, Z, Y, lr_, lx)
            target = vpp(tf.constant(Xpp), training=False).numpy()
            v = fit(fresh("v", DIMS[reg]), X, target, tag=f"{reg}/v<-postpost")
            save(v, a.out, reg, "v")
            for nm in NETS[reg][1:]:
                save(fresh(nm, DIMS[reg]), a.out, reg, nm)   # controls cold
            print(f"  anchor init written for {reg}")

if __name__ == "__main__":
    main()
