"""JAX spike: faithful port of the PostDamagePostTech train step.

Steps
  1. Extract RUNA's trained TF weights (explicit per-layer attributes -> pytree).
  2. Rebuild the network in JAX. Key simplification, verified exactly: BatchNorm
     moving stats are frozen at initialization (mean=0, var=1) because pde_rhs
     always calls the nets with training=False -> BN is a learnable affine
     y = gamma * x / sqrt(1+1e-6) + beta.
  3. Parity: HJB residual RMS on the same LHS sample, TF vs JAX, same weights.
  4. Bench: jit-compiled train step (value + control Adam updates), steps/s.

Run: module load python/anaconda-2021.05; python -u port_postpost.py
"""
import os, sys, time, json
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "models"))
RUNA = os.path.join(REPO, "output_001",
    "OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_"
    "#HiddenLayer_4_num_iterations1000000")

# ---------------------------------------------------------------- 1) TF side
import tensorflow as tf
from feedforward_subnet import FeedForwardSubNet
from params import PARAMS, investment_rate_activation

def tf_cfg(act, fin, nm):
    return {"num_hiddens": [32] * 4, "use_bias": True, "activation": act,
            "dim": 1, "nn_name": nm, "final_activation": fin}

CFG = {"v": ("swish", "softplus"), "i_g": ("tanh", investment_rate_activation(PARAMS["θ_g"])),
       "i_d": ("tanh", investment_rate_activation(PARAMS["θ_d"]))}

def extract(nm):
    act, fin = CFG[nm]
    n = FeedForwardSubNet(tf_cfg(act, fin, f"{nm}_nn"))
    n(tf.zeros([1, 7]))
    n.load_weights(f"{RUNA}/PostDamagePostTech/{nm}_nn_checkpoint_PostDamagePostTech").expect_partial()
    P = {}
    for j, bn in enumerate(n.bn_layers):
        assert abs(bn.moving_mean.numpy()).max() == 0.0 and abs(bn.moving_variance.numpy() - 1).max() < 1e-12
        P[f"bn{j}"] = {"g": bn.gamma.numpy(), "b": bn.beta.numpy()}
    for j, d in enumerate(n.dense_layers):
        P[f"d{j}"] = {"W": d.kernel.numpy(), "b": d.bias.numpy()}
    return P

TFP = {nm: extract(nm) for nm in CFG}
print("TF weights extracted (BN frozen-affine assumption verified exactly)")

# ---------------------------------------------------------------- 2) JAX side
import jax
import jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")

p = PARAMS
δ, A_d, Agpp = p["δ"], p["A_d"], p["A_g_prime_prime"]
α_d, Γ_d, θ_d, σ_d = p["α_d"], p["Γ_d"], p["θ_d"], p["σ_d"]
α_g, Γ_g, θ_g, σ_g = p["α_g"], p["Γ_g"], p["θ_g"], p["σ_g"]
θ_bar, η, ς = p["θ_bar"], p["η"], p["ϛ"]
λ1, λ2, y_upper = p["λ1"], p["λ2"], p["y_upper"]
EPS_BN = 1.0 / np.sqrt(1.0 + 1e-6)

def to_pytree(P):
    return {k: {kk: jnp.asarray(vv) for kk, vv in v.items()} for k, v in P.items()}

PARAMS_J = {nm: to_pytree(TFP[nm]) for nm in TFP}

def swish(x): return x * jax.nn.sigmoid(x)
def inv_rate(theta): return lambda x: 1.0 - (1.0 + 1.0 / theta) / (jnp.exp(2.0 * x) + 1.0)
ACTS = {"v": (swish, jax.nn.softplus), "i_g": (jnp.tanh, inv_rate(θ_g)), "i_d": (jnp.tanh, inv_rate(θ_d))}

def net_apply(P, x, act, fin):
    """bn0 -> (dense_i[act] -> bn_{i+1})*4 -> sum of the four bn outputs -> dense[fin]."""
    h = P["bn0"]["g"] * x * EPS_BN + P["bn0"]["b"]
    outs = []
    for i in range(4):
        h = act(h @ P[f"d{i}"]["W"] + P[f"d{i}"]["b"])
        h = P[f"bn{i+1}"]["g"] * h * EPS_BN + P[f"bn{i+1}"]["b"]
        outs.append(h)
    s = outs[0] + outs[1] + outs[2] + outs[3]
    return fin(s @ P["d4"]["W"] + P["d4"]["b"])

def make_scalar(nm):
    act, fin = ACTS[nm]
    def f(P, s3, pseudo):     # s3=(logK,Z,Y); pseudo=(lam3, Agpp, logxi, logxi)
        x = jnp.concatenate([s3, pseudo])
        return net_apply(P, x, act, fin)[0]
    return f

v_f, ig_f, id_f = make_scalar("v"), make_scalar("i_g"), make_scalar("i_d")

def pde_rhs_point(Pv, Pg, Pd, s3, lam3, logxi):
    pseudo = jnp.array([lam3, Agpp, logxi, logxi], dtype=jnp.float32)
    logK, Z, Y = s3
    ξ, K = jnp.exp(logxi), jnp.exp(logK)
    v = v_f(Pv, s3, pseudo)
    i_g = ig_f(Pg, s3, pseudo)
    i_d = id_f(Pd, s3, pseudo)
    g = jax.grad(v_f, argnums=1)(Pv, s3, pseudo)                 # (3,)
    H = jax.hessian(v_f, argnums=1)(Pv, s3, pseudo)              # (3,3)
    dvK, dvZ, dvY = g
    d2K, d2Z, d2Y, d2KZ = H[0, 0], H[1, 1], H[2, 2], H[0, 1]
    h_d = -1.0 / ξ * ((dvK - Z * dvZ) * (1 - Z) * σ_d)
    h_g = -1.0 / ξ * ((dvK + (1 - Z) * dvZ) * Z * σ_g)
    h_y = -1.0 / ξ * (dvY - (λ1 + λ2 * Y + lam3 * (Y - y_upper))) * η * A_d * (1 - Z) * K * ς
    pv = δ * v
    c = (A_d - i_d) * (1 - Z) + (Agpp - i_g) * Z
    inside = jnp.maximum(c, 1e-8)
    flow = δ * (jnp.log(inside) + logK)
    ilid = jnp.maximum(1.0 + θ_d * i_d, 1e-8)
    ilig = jnp.maximum(1.0 + θ_g * i_g, 1e-8)
    vKK_t = (σ_d ** 2 * (1 - Z) ** 2 + σ_g ** 2 * Z ** 2) / 2.0
    vK_t = (α_d + Γ_d * jnp.log(ilid)) * (1 - Z) + (α_g + Γ_g * jnp.log(ilig)) * Z - vKK_t
    vZ_t = (α_g + Γ_g * jnp.log(ilig) - (α_d + Γ_d * jnp.log(ilid)) - Z * σ_g ** 2 + (1 - Z) * σ_d ** 2) * Z * (1 - Z)
    vZZ_t = 0.5 * Z ** 2 * (1 - Z) ** 2 * (σ_g ** 2 + σ_d ** 2)
    vKZ_t = -Z * (1 - Z) ** 2 * σ_d ** 2 + Z ** 2 * (1 - Z) * σ_g ** 2
    vy_t = (θ_bar + h_y * ς) * η * A_d * (1 - Z) * K
    vyy_t = 0.5 * ς ** 2 * (η * A_d * (1 - Z) * K) ** 2
    vlogN_t = (λ1 + λ2 * Y + lam3 * (Y - y_upper)) * vy_t + (λ2 + lam3) * vyy_t
    rhs = (flow + vK_t * dvK + vKK_t * d2K + vZ_t * dvZ + vZZ_t * d2Z
           + h_d * (dvK - Z * dvZ) * (1 - Z) * σ_d + h_g * (dvK + (1 - Z) * dvZ) * Z * σ_g
           + vKZ_t * d2KZ + dvY * vy_t + vyy_t * d2Y
           + 0.5 * ξ * (h_d ** 2 + h_g ** 2 + h_y ** 2) - vlogN_t)
    mu_c = δ / inside
    FOC_d = -mu_c + Γ_d * θ_d / ilid * (dvK - Z * dvZ)
    FOC_g = -mu_c + Γ_g * θ_g / ilig * (dvK + (1 - Z) * dvZ)
    return rhs - pv, FOC_d, FOC_g, dvY

pde_batch = jax.vmap(pde_rhs_point, in_axes=(None, None, None, 0, 0, 0))

# ---------------------------------------------------------------- 3) parity
def lhs_np(n, bounds, seed=0):
    rng = np.random.RandomState(seed)
    return [((lo + (hi - lo) * (rng.permutation(n) + rng.rand(n)) / n).astype(np.float32))
            for lo, hi in bounds]

N = 8192
lk, Z, Y, lr, l3 = lhs_np(N, [(4, 7), (0.01, 0.99), (2.5, 4), (1, 6), (0, 1 / 3)])
XI = 0.05
lx = np.full(N, np.log(XI), dtype=np.float32)
s3 = jnp.stack([jnp.asarray(lk), jnp.asarray(Z), jnp.asarray(Y)], axis=1)
res_j, fd_j, fg_j, _ = pde_batch(PARAMS_J["v"], PARAMS_J["i_g"], PARAMS_J["i_d"],
                                 s3, jnp.asarray(l3), jnp.asarray(lx))
jax_rms = float(jnp.sqrt(jnp.mean(res_j ** 2)))

# TF reference on the identical points with the identical weights
sys.path.insert(0, os.path.join(REPO, "benchmarks", "solution_comparison"))
import solution_loader as SL
m = SL.make_model("PostDamagePostTech", RUNA)
import tensorflow as tf2
_T = [tf2.constant(a.reshape(-1, 1)) for a in (lk, Z, Y, lr, l3)]
_lx = tf2.constant(np.full((N, 1), np.log(XI), dtype=np.float32))
_res = m.objective_fn(_T[0], _T[1], _T[2], _T[3], _T[4], _lx, training=False)
r = {"res": float(_res[0]), "FOC_d": float(_res[1]), "FOC_g": float(_res[2])}
tf_rms = r["res"]
print(f"\nPARITY  residual RMS: TF={tf_rms:.6e}  JAX={jax_rms:.6e}  "
      f"rel.diff={abs(tf_rms - jax_rms) / tf_rms:.2e}")
print(f"        FOC_d RMS: TF={r['FOC_d']:.4e} JAX={float(jnp.sqrt(jnp.mean(fd_j**2))):.4e} | "
      f"FOC_g RMS: TF={r['FOC_g']:.4e} JAX={float(jnp.sqrt(jnp.mean(fg_j**2))):.4e}")

# ---------------------------------------------------------------- 4) bench
def value_loss(Pv, Pg, Pd, s3, lam3, logxi, Ys):
    res, fd, fg, dvY = pde_batch(Pv, Pg, Pd, s3, lam3, logxi)
    mono = dvY * (Ys > y_upper) * (dvY > 0) + 1e-7
    return (jnp.sqrt(jnp.mean(res ** 2)) + jnp.sqrt(jnp.mean(fg ** 2))
            + jnp.sqrt(jnp.mean(fd ** 2)) + jnp.sqrt(jnp.mean(mono ** 2)))

def ctrl_loss(Pg, Pd, Pv, s3, lam3, logxi):
    res, fd, fg, _ = pde_batch(Pv, Pg, Pd, s3, lam3, logxi)
    return -jnp.mean(res) + jnp.sqrt(jnp.mean(fg ** 2)) + jnp.sqrt(jnp.mean(fd ** 2))

def adam_update(g, m_, v_, t, lr, b1=0.9, b2=0.999, e=1e-7):
    m_ = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m_, g)
    v_ = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v_, g)
    def upd(mm, vv):
        return lr * (mm / (1 - b1 ** t)) / (jnp.sqrt(vv / (1 - b2 ** t)) + e)
    return jax.tree_util.tree_map(upd, m_, v_), m_, v_

def sample(key, n):
    ks = jax.random.split(key, 5)
    lo = jnp.array([4.0, 0.01, 0.0, 0.0, np.log(0.05)])
    hi = jnp.array([7.0, 0.99, 4.0, 1 / 3, np.log(148.6)])
    u = jnp.stack([jax.random.uniform(ks[i], (n,)) for i in range(5)], axis=1)
    z = lo + (hi - lo) * u
    return jnp.stack([z[:, 0], z[:, 1], z[:, 2]], axis=1), z[:, 3], z[:, 4]

def train_step(state, key, batch):
    Pv, Pg, Pd, mv, vv, mc, vc, t = state
    s3, lam3, logxi = sample(key, batch)
    gv = jax.grad(value_loss)(Pv, Pg, Pd, s3, lam3, logxi, s3[:, 2])
    dv, mv, vv = adam_update(gv, mv, vv, t, 1e-5)
    Pv = jax.tree_util.tree_map(lambda a, b: a - b, Pv, dv)
    gc = jax.grad(ctrl_loss, argnums=(0, 1))(Pg, Pd, Pv, s3, lam3, logxi)
    dc, mc, vc = adam_update(gc, mc, vc, t, 4e-4)
    Pg = jax.tree_util.tree_map(lambda a, b: a - b, Pg, dc[0])
    Pd = jax.tree_util.tree_map(lambda a, b: a - b, Pd, dc[1])
    return (Pv, Pg, Pd, mv, vv, mc, vc, t + 1)

for BATCH in [128, 512, 2048]:
    zeros = lambda P: jax.tree_util.tree_map(jnp.zeros_like, P)
    st = (PARAMS_J["v"], PARAMS_J["i_g"], PARAMS_J["i_d"],
          zeros(PARAMS_J["v"]), zeros(PARAMS_J["v"]),
          (zeros(PARAMS_J["i_g"]), zeros(PARAMS_J["i_d"])),
          (zeros(PARAMS_J["i_g"]), zeros(PARAMS_J["i_d"])), 1)
    step = jax.jit(train_step, static_argnums=2)
    key = jax.random.PRNGKey(0)
    t0 = time.time()
    st = step(st, key, BATCH)               # compile
    jax.block_until_ready(st[0])
    compile_s = time.time() - t0
    WARM, TIMED = 20, 300
    for i in range(WARM):
        key, k = jax.random.split(key)
        st = step(st, k, BATCH)
    jax.block_until_ready(st[0])
    t0 = time.time()
    for i in range(TIMED):
        key, k = jax.random.split(key)
        st = step(st, k, BATCH)
    jax.block_until_ready(st[0])
    dt = time.time() - t0
    print(f"BENCH  batch={BATCH}: {TIMED/dt:.1f} steps/s   (compile {compile_s:.1f}s)   "
          f"[TF reference: 158-169 steps/s @128, 27 @2048, 4cpu]")
print("done")
