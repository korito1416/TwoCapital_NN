"""Diagnose the FOC mismatch: replicate validation (random points over the full training
ranges) and also probe FOC vs logxi / lambda3 region, plus an autodiff-vs-finite-diff check."""
import os, sys
import numpy as np, tensorflow as tf
ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
sys.path.insert(0, os.path.join(ROOT, "models"))
from feedforward_subnet import FeedForwardSubNet
from params import PARAMS, investment_rate_activation
import plot_pretrained_climate as P

vnn, idnn, ignn = P.build_and_load()
A_d, A_gpp = PARAMS["A_d"], PARAMS["A_g_prime_prime"]
Gd, td, Gg, tg, dl = PARAMS["Γ_d"], PARAMS["θ_d"], PARAMS["Γ_g"], PARAMS["θ_g"], PARAMS["δ"]

def foc(logK, Z, Y, l3, lx):
    n = len(Z)
    cols = [tf.constant(a.reshape(-1,1), tf.float32) for a in (logK, Z, Y, l3,
            np.full(n, A_gpp), lx, lx)]
    lK, Zt, Yt = cols[0], cols[1], cols[2]
    with tf.GradientTape(persistent=True) as t:
        t.watch([lK, Zt]); X = tf.concat([lK, Zt, Yt, cols[3], cols[4], cols[5], cols[6]], 1)
        v = vnn(X, training=False)
    vlK = t.gradient(v, lK).numpy().ravel(); vZ = t.gradient(v, Zt).numpy().ravel(); del t
    X = tf.concat([lK, Zt, Yt, cols[3], cols[4], cols[5], cols[6]], 1)
    i_d = idnn(X, training=False).numpy().ravel(); i_g = ignn(X, training=False).numpy().ravel()
    c = (A_d-i_d)*(1-Z) + (A_gpp-i_g)*Z
    qd = vlK - Z*vZ; qg = vlK + (1-Z)*vZ
    Fd = -dl/np.maximum(c,1e-8) + Gd*td/(1+td*i_d)*qd
    Fg = -dl/np.maximum(c,1e-8) + Gg*tg/(1+tg*i_g)*qg
    return Fd, Fg, vlK, vZ, i_d, i_g, c

# (1) replicate validation: random points over the FULL training ranges
rng = np.random.RandomState(0); N = 4000
logK = rng.uniform(4,7,N); Z = rng.uniform(0.01,0.99,N); Y = rng.uniform(0,4,N)
l3 = rng.uniform(0,1/3,N); lx = rng.uniform(-3,5,N)
Fd, Fg, *_ = foc(logK, Z, Y, l3, lx)
print(f"[FULL-RANGE random {N}] RMS FOC_d={np.sqrt(np.mean(Fd**2)):.2e} RMS FOC_g={np.sqrt(np.mean(Fg**2)):.2e}  (training ~5e-5)")
print(f"   pct |FOC_d|>0.01: {100*np.mean(np.abs(Fd)>0.01):.1f}%   median|FOC_d|={np.median(np.abs(Fd)):.2e}  max={np.max(np.abs(Fd)):.2e}")

# (2) FOC vs logxi and lambda3 at a central (logK,Z,Y)
print("\n[FOC vs region]  logK=5.5, Z=0.5, Y=2.0")
for lxv in (-2.0, 0.0, 2.0, 4.0, 4.9):
    Fd,_,_,_,_,_,_ = foc(np.array([5.5]),np.array([0.5]),np.array([2.0]),np.array([1/6]),np.array([lxv]))
    print(f"   logxi={lxv:+.1f}: |FOC_d|={abs(Fd[0]):.2e}")
for l3v in (0.0, 1/12, 1/6, 1/4, 1/3):
    Fd,_,_,_,_,_,_ = foc(np.array([5.5]),np.array([0.5]),np.array([2.0]),np.array([l3v]),np.array([0.0]))
    print(f"   lambda3={l3v:.3f}: |FOC_d|={abs(Fd[0]):.2e}")

# (3) autodiff vs finite-difference for v_logK at one point
def vval(logK,Z,Y,l3,lx):
    n=len(Z); cols=[tf.constant(a.reshape(-1,1),tf.float32) for a in (logK,Z,Y,l3,np.full(n,A_gpp),lx,lx)]
    return vnn(tf.concat(cols,1), training=False).numpy().ravel()
z0=np.array([0.5]); base=dict(Z=z0,Y=np.array([2.0]),l3=np.array([1/6]),lx=np.array([0.0]))
eps=1e-3
vp=vval(np.array([5.5+eps]),**base); vm=vval(np.array([5.5-eps]),**base)
fd=(vp-vm)/(2*eps)
_,_,vlK,_,_,_,_=foc(np.array([5.5]),base['Z'],base['Y'],base['l3'],base['lx'])
print(f"\n[autodiff vs FD]  v_logK: autodiff={vlK[0]:.4f}  finite-diff={fd[0]:.4f}")

# (4) batch dependence: point alone vs embedded in random 1024 batch
pt=dict(logK=np.array([5.5]),Z=np.array([0.5]),Y=np.array([2.0]),l3=np.array([1/6]),lx=np.array([0.0]))
Fd_alone,_,_,_,_,_,_=foc(**pt)
big=dict(logK=np.r_[5.5,rng.uniform(4,7,1023)],Z=np.r_[0.5,rng.uniform(.01,.99,1023)],
         Y=np.r_[2.0,rng.uniform(0,4,1023)],l3=np.r_[1/6,rng.uniform(0,1/3,1023)],lx=np.r_[0.0,rng.uniform(-3,5,1023)])
Fd_big,_,_,_,_,_,_=foc(**big)
print(f"[batch dep]  |FOC_d| alone={abs(Fd_alone[0]):.2e}   same pt in 1024-batch={abs(Fd_big[0]):.2e}")
