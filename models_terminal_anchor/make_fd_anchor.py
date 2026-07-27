"""Build an FD-ANCHORED warm-start checkpoint set (the reproducible-science init).

The anchor is DERIVED from the verified PIBYS finite-difference solutions of the
terminal Post-Damage Post-Tech regime (Richardson grid-converged + analytic-corner
checked), NOT inherited from any NN checkpoint. Fully traceable:
  FD solve (provenance json) -> this supervised fit (manifest) -> training chain.

Targets
  terminal PostDamagePostTech : v, i_g, i_d fitted to the five-lambda3 FD family,
      linear-interpolated in lambda3, flat in logxi (terminal robustness O(sigma^2/xi)).
  non-terminal regimes        : v fitted to the same FD value broadcast (flat in logR
      where applicable) -- one consistent welfare level across the whole system;
      i_g/i_d fitted to the FD controls; i_r fresh (R&D unanchored until Stage 2).

Variant mode (--variant fd_<tag>.npz): the value/control targets become
      variant^{lam3=1/6}(logK,Z,Y) + [base^{lam3} - base^{1/6}](logK,Z,Y)
  i.e. the economically-coherent variant level+shape with the verified lambda3
  structure of the baseline family. Used to build DIFFERENT-LEVEL decarbonizing
  warm starts (e.g. the delta=0.008 planner).

Usage
  python make_fd_anchor.py --out ARM [--variant PATH.npz] [--seed 1]
"""
import argparse, os, sys, json, datetime
import numpy as np
import tensorflow as tf
from scipy.interpolate import RegularGridInterpolator as RGI

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "models_warmstart"))
from feedforward_subnet import FeedForwardSubNet
from params import PARAMS, investment_rate_activation

OD = os.path.join(ROOT, "benchmarks", "post_damage_post_tech", "outputs")
TAGS = ["0000", "0083", "0167", "0250", "0333"]
LAM3S = np.array([0.0, 1/12, 1/6, 1/4, 1/3])
AGPP = 0.1567

REGS = ["PreDamagePreTech", "PreDamagePostTech", "PostDamagePreTech", "PostDamagePostTech"]
DIMS = {"PreDamagePreTech": 7, "PreDamagePostTech": 6, "PostDamagePreTech": 8, "PostDamagePostTech": 7}
NETS = {"PreDamagePreTech": ["v", "i_g", "i_d", "i_r"], "PreDamagePostTech": ["v", "i_g", "i_d"],
        "PostDamagePreTech": ["v", "i_g", "i_d", "i_r"], "PostDamagePostTech": ["v", "i_g", "i_d"]}

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

class FDFamily:
    """v / i_d / i_g interpolators over (logK,Z,Y) for each of the five lambda3, plus
    optional variant-shift: target = variant^{1/6} + (base^{lam3} - base^{1/6})."""
    def __init__(self, variant_npz=None):
        self.base = {}
        for t, l3 in zip(TAGS, LAM3S):
            d = np.load(os.path.join(OD, f"fd_pdpt_v5_stable_lam3_{t}_xi148.npz"))
            self.base[t] = {k: RGI((d["logK"], d["Z"], d["Y"]), d[k],
                                   bounds_error=False, fill_value=None)
                            for k in ("v", "i_d", "i_g")}
        self.variant = None
        if variant_npz:
            d = np.load(variant_npz)
            self.variant = {k: RGI((d["logK"], d["Z"], d["Y"]), d[k],
                                   bounds_error=False, fill_value=None)
                            for k in ("v", "i_d", "i_g")}

    def field(self, key, pts, l3col):
        """pts: (n,3) [logK,Z,Y clipped]; l3col: (n,) lambda3 values -> target (n,)."""
        vals = np.stack([self.base[t][key](pts) for t in TAGS], axis=1)  # (n,5)
        # linear interp across lambda3
        idx = np.clip(np.searchsorted(LAM3S, l3col) - 1, 0, 3)
        w = (l3col - LAM3S[idx]) / (LAM3S[idx + 1] - LAM3S[idx])
        out = vals[np.arange(len(pts)), idx] * (1 - w) + vals[np.arange(len(pts)), idx + 1] * w
        if self.variant is not None:
            base16 = self.base["0167"][key](pts)
            out = self.variant[key](pts) + (out - base16)
        return out.reshape(-1, 1)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default=None, help="variant npz for a different-level anchor")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    tf.random.set_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.RandomState(a.seed)
    fam = FDFamily(a.variant)

    for reg in REGS:
        ylo = 2.5 if reg.startswith("PostDamage") else 0.0
        lk, Z, Y, lr_, l3 = lhs(16384, [(4, 7), (0.01, 0.99), (ylo, 4), (1, 6), (0, 1/3)], rng)
        lx = (np.log(0.05) + (np.log(148.6) - np.log(0.05)) * rng.rand(16384, 1)).astype(np.float32)
        X = X_of(reg, lk, Z, Y, lr_, l3, lx)
        # FD fields cover (logK,Z,Y) with Y in [0,4]; broadcast the FD value AT THE SAME
        # STATE to every regime (keeps the climate shape and ONE consistent level; the
        # pre-damage regimes' lambda3 is not yet realized -> use the central lam3=1/6).
        pts = np.hstack([lk, Z, np.clip(Y, 0.0, 4.0)]).astype(np.float64)
        l3col = (l3 if reg in ("PostDamagePreTech", "PostDamagePostTech")
                 else np.full_like(l3, 1/6)).ravel().astype(np.float64)

        v_t = fam.field("v", pts, l3col)
        v = fit(fresh("v", DIMS[reg]), X, v_t, tag=f"{reg}/v")
        save(v, a.out, reg, "v")
        for nm in [n for n in NETS[reg] if n in ("i_g", "i_d")]:
            t = fam.field(nm, pts, l3col)
            net = fit(fresh(nm, DIMS[reg]), X, t, steps=3000, tag=f"{reg}/{nm}")
            save(net, a.out, reg, nm)
        if "i_r" in NETS[reg]:
            save(fresh("i_r", DIMS[reg]), a.out, reg, "i_r")   # R&D unanchored (Stage 2)
        print(f"  fd-anchor written for {reg}", flush=True)

    manifest = dict(
        created=str(datetime.date.today()), tool="models_terminal_anchor/make_fd_anchor.py",
        anchor="five-lambda3 PIBYS FD family (Richardson-converged, analytic-corner checked)",
        fd_sources=[f"fd_pdpt_v5_stable_lam3_{t}_xi148.npz" for t in TAGS],
        variant=a.variant, seed=a.seed,
        notes="terminal exact fit; non-terminal v anchored at entry slice Y=2.5; i_r fresh")
    with open(os.path.join(a.out, "MANIFEST_FD_ANCHOR.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("manifest written; FD-ANCHOR COMPLETE", flush=True)

if __name__ == "__main__":
    main()
