"""Build a warm-start checkpoint set from an ECONOMY MAP (the economy-zoo pipeline).

A map module is a python file defining:
    MAP_NAME = "..."                      # provenance tag
    PROVENANCE = {...}                    # free-form dict: model, solution method, files
    def fields(reg, lk, Z, Y, lr, l3, lx) -> dict
        # inputs: (n,1) float arrays of PRODUCTION states
        #   reg in {PreDamagePreTech, PreDamagePostTech, PostDamagePreTech, PostDamagePostTech}
        # returns dict with keys:
        #   v    (n,1)  value target
        #   i_d  (n,1)  dirty investment rate
        #   i_g  (n,1)  green investment rate
        #   i_r  (n,1) or None  R&D rate I_r/K (None for post-tech regimes / inactive)

Net conventions honored (verified against models_warmstart training code):
    v_nn   : direct fit (swish/softplus head)
    i_d/i_g: direct rate fit (tanh / bounded investment-rate head)
    i_r_nn : the net outputs -log(i_r)  [training uses i_r = exp(-i_r_nn(X))]
             -> fit target is -log(max(i_r_map, 1e-8)). ACTIVE, not fresh.

Usage
  python make_map_anchor.py --map economy_maps/<econ>.py --out ARM [--seed 1]
"""
import argparse, os, sys, json, datetime, importlib.util
import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "models_warmstart"))
from feedforward_subnet import FeedForwardSubNet
from params import PARAMS, investment_rate_activation

REGS = ["PreDamagePreTech", "PreDamagePostTech", "PostDamagePreTech", "PostDamagePostTech"]
DIMS = {"PreDamagePreTech": 7, "PreDamagePostTech": 6, "PostDamagePreTech": 8, "PostDamagePostTech": 7}
NETS = {"PreDamagePreTech": ["v", "i_g", "i_d", "i_r"], "PreDamagePostTech": ["v", "i_g", "i_d"],
        "PostDamagePreTech": ["v", "i_g", "i_d", "i_r"], "PostDamagePostTech": ["v", "i_g", "i_d"]}
AGPP = 0.1567

def _cfg(nm):
    act = {"v": ("swish", "softplus"),
           "i_g": ("tanh", investment_rate_activation(PARAMS["θ_g"])),
           "i_d": ("tanh", investment_rate_activation(PARAMS["θ_d"])),
           "i_r": ("softplus", "softplus")}[nm]
    return {"num_hiddens": [32] * 4, "use_bias": True, "activation": act[0],
            "dim": 1, "nn_name": f"{nm}_nn", "final_activation": act[1]}

def fresh(nm, dim):
    n = FeedForwardSubNet(_cfg(nm)); n(tf.zeros([1, dim])); return n

def save(net, out, reg, nm):
    os.makedirs(f"{out}/{reg}", exist_ok=True)
    net.save_weights(f"{out}/{reg}/{nm}_nn_checkpoint_{reg}")

def lhs(n, bounds, rng):
    return [((lo + (hi - lo) * (rng.permutation(n) + rng.rand(n)) / n)
             .reshape(-1, 1).astype(np.float32)) for lo, hi in bounds]

def X_of(reg, lk, Z, Y, lr, l3, lx):
    ones = np.ones_like(Y)
    if reg == "PreDamagePreTech":   return np.hstack([lk, Z, Y, lr, lx, lx, lx])
    if reg == "PreDamagePostTech":  return np.hstack([lk, Z, Y, AGPP * ones, lx, lx])
    if reg == "PostDamagePreTech":  return np.hstack([lk, Z, Y, lr, l3, lx, lx, lx])
    if reg == "PostDamagePostTech": return np.hstack([lk, Z, Y, l3, AGPP * ones, lx, lx])

def fit(net, X, target, steps=6000, lr=1e-3, tag=""):
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
            print(f"    fit {tag} step {it}: mse={float(loss):.3e}", flush=True)
    return net

def load_map(path):
    spec = importlib.util.spec_from_file_location("economy_map", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "fields") and hasattr(mod, "MAP_NAME"), "map module needs MAP_NAME + fields()"
    return mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--samples", type=int, default=16384)
    a = ap.parse_args()
    tf.random.set_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.RandomState(a.seed)
    m = load_map(a.map)
    print(f"economy map: {m.MAP_NAME}", flush=True)

    report = {}
    for reg in REGS:
        ylo = 2.5 if reg.startswith("PostDamage") else 0.0
        lk, Z, Y, lr_, l3 = lhs(a.samples, [(4, 7), (0.01, 0.99), (ylo, 4), (1, 6), (0, 1/3)], rng)
        lx = (np.log(0.05) + (np.log(148.6) - np.log(0.05)) * rng.rand(a.samples, 1)).astype(np.float32)
        X = X_of(reg, lk, Z, Y, lr_, l3, lx)
        F = m.fields(reg, lk.astype(np.float64), Z.astype(np.float64), Y.astype(np.float64),
                     lr_.astype(np.float64), l3.astype(np.float64), lx.astype(np.float64))
        v = fit(fresh("v", DIMS[reg]), X, np.asarray(F["v"]), tag=f"{reg}/v")
        save(v, a.out, reg, "v")
        for nm in ("i_g", "i_d"):
            net = fit(fresh(nm, DIMS[reg]), X, np.asarray(F[nm]), steps=3000, tag=f"{reg}/{nm}")
            save(net, a.out, reg, nm)
        if "i_r" in NETS[reg]:
            ir = F.get("i_r", None)
            if ir is None:
                save(fresh("i_r", DIMS[reg]), a.out, reg, "i_r")
                print(f"    {reg}/i_r: map gave None -> fresh net", flush=True)
            else:
                tgt = -np.log(np.maximum(np.asarray(ir), 1e-8))   # net outputs -log(i_r)
                net = fit(fresh("i_r", DIMS[reg]), X, tgt, steps=3000, tag=f"{reg}/i_r[-log]")
                save(net, a.out, reg, "i_r")
        report[reg] = {k: [float(np.min(np.asarray(F[k]))), float(np.max(np.asarray(F[k])))]
                       for k in F if F[k] is not None}
        print(f"  map-anchor written for {reg}", flush=True)

    manifest = dict(created=str(datetime.date.today()),
                    tool="models_terminal_anchor/make_map_anchor.py",
                    map=a.map, map_name=m.MAP_NAME,
                    provenance=getattr(m, "PROVENANCE", {}),
                    seed=a.seed, field_ranges=report,
                    ir_convention="i_r net fit to -log(i_r_map); training uses i_r=exp(-net)")
    with open(os.path.join(a.out, "MANIFEST_MAP_ANCHOR.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("manifest written; MAP-ANCHOR COMPLETE", flush=True)

if __name__ == "__main__":
    main()
