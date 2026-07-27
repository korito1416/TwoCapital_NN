"""
KEYSTONE adjudication (identification-free): which investment policy is OPTIMAL?

The HJB residual cannot settle NN-vs-FD here because v_Z is weakly identified (the Z
direction is advection-dominated, cell-Peclet ~50-140). So we DON'T use any derivative.
Instead we forward-integrate the controlled economy under each candidate FEEDBACK policy
from the same x0 and compare the REALIZED discounted welfare

    J = integral_0^inf  e^{-delta t} * delta * log(C_t / N(Y_t)) dt          (this is V(x0))

with C=c*K, c=(A_d-i_d)(1-Z)+(A_g''-i_g)Z, logN(Y)=l1 Y+0.5 l2 Y^2+0.5 l3 (Y-ybar)^2.
The additive logN constant cancels between the two policies, and v_Z never appears.
HIGHER J = closer-to-optimal policy. This DIRECTLY answers "is de-investment (i_d<0) right?"

sigma=0.01 => diffusion is O(sigma^2)~1e-4, so the value of the (nearly) deterministic
problem = the deterministic drift-integral to O(sigma^2). We integrate the drift ODE (RK4).
xi=148.4 => worst-case drift distortion h~0, integrate under the approximating measure.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
from params import PARAMS as P                                                    # noqa: E402
import plot_pretrained_climate as C                                              # noqa: E402

OD = C.OD
dl = P["δ"]; A_d = P["A_d"]; A_g = P["A_g_prime_prime"]
ad, Gd, td, sd = P["α_d"], P["Γ_d"], P["θ_d"], P["σ_d"]
ag, Gg, tg, sg = P["α_g"], P["Γ_g"], P["θ_g"], P["σ_g"]
thbar, eta, vs = P["θ_bar"], P["η"], P["ϛ"]
l1, l2, ybar = P["λ1"], P["λ2"], P["y_upper"]
LAM3 = 1.0 / 6.0

# ---- the model's deterministic drifts given a state and controls ----
def phi_d(i): return ad + Gd * np.log(np.maximum(1 + td * i, 1e-9))
def phi_g(i): return ag + Gg * np.log(np.maximum(1 + tg * i, 1e-9))

def drifts(logK, Z, Y, i_d, i_g):
    pd, pg = phi_d(i_d), phi_g(i_g)
    Dc = sd**2 * (1 - Z)**2 + sg**2 * Z**2
    a_lK = (1 - Z) * pd + Z * pg - Dc / 2
    a_Z = Z * (1 - Z) * (pg - pd + (1 - Z) * sd**2 - Z * sg**2)
    E = eta * A_d * (1 - Z) * np.exp(logK)
    a_Y = thbar * E
    return a_lK, a_Z, a_Y

def felicity(logK, Z, Y, i_d, i_g):
    c = (A_d - i_d) * (1 - Z) + (A_g - i_g) * Z
    logN = l1 * Y + 0.5 * l2 * Y**2 + 0.5 * LAM3 * (Y - ybar)**2      # const dropped (cancels)
    return dl * (np.log(np.maximum(c, 1e-12)) + logK - logN), c

# ---- trilinear interpolation of a control grid, clamped at edges ----
class Grid3:
    def __init__(self, lk, z, y, fields):
        self.lk, self.z, self.y = lk, z, y; self.f = fields
    def _ix(self, ax, v):
        i = np.searchsorted(ax, v) - 1; i = min(max(i, 0), len(ax) - 2)
        w = (v - ax[i]) / (ax[i + 1] - ax[i]); w = min(max(w, 0.0), 1.0)   # clamp -> flat extrap
        return i, w
    def __call__(self, key, logK, Z, Y):
        a = self.f[key]
        i, wi = self._ix(self.lk, logK); j, wj = self._ix(self.z, Z); k, wk = self._ix(self.y, Y)
        c = 0.0
        for di, wdi in ((0, 1 - wi), (1, wi)):
            for dj, wdj in ((0, 1 - wj), (1, wj)):
                for dk, wdk in ((0, 1 - wk), (1, wk)):
                    c += wdi * wdj * wdk * a[i + di, j + dj, k + dk]
        return c

# ---- build the NN control+value grid (one vectorized call per logK slice) ----
def build_nn_grid(lk, z, y):
    net = C.build_and_load()
    ZZ, YY = np.meshgrid(z, y, indexing="ij")
    Zf, Yf = ZZ.ravel(), YY.ravel()
    i_d = np.zeros((len(lk), len(z), len(y))); i_g = np.zeros_like(i_d); v = np.zeros_like(i_d)
    foc = 0.0
    for a, lkv in enumerate(lk):
        o = C.evaluate(*net, Zf, Yf, logK=lkv, lam3=LAM3, logxi=5.0)
        i_d[a] = o["i_d"].reshape(len(z), len(y))
        i_g[a] = o["i_g"].reshape(len(z), len(y))
        v[a] = o["v"].reshape(len(z), len(y))
        foc = max(foc, np.sqrt(np.mean(o["FOC_d"]**2 + o["FOC_g"]**2)))
    print(f"[NN grid] built {i_d.shape}, RMS FOC over grid = {foc:.2e}", flush=True)
    return Grid3(lk, z, y, dict(i_d=i_d, i_g=i_g, v=v))

def load_fd_grid(npz):
    d = np.load(npz)
    return Grid3(d["logK"], d["Z"], d["Y"], dict(i_d=d["i_d"], i_g=d["i_g"], v=d["v"]))

# ---- forward-integrate one feedback policy from x0; return realized welfare J ----
CHECKPTS = [25, 50, 100, 200, 400, 800, 1500]

def evaluate_policy(pol, x0, T=1500.0, dt=0.25, name=""):
    logK, Z, Y = x0
    J = 0.0; t = 0.0
    nstep = int(T / dt)
    traj = []                          # (t, logK, Z, Y, i_d, i_g, c) every 25y
    Jckpt = {}                         # cumulative discounted welfare at each horizon
    t_exit = None                      # first time Y leaves the calibrated domain [0,4]
    ci = 0
    for s in range(nstep):
        i_d = float(pol("i_d", logK, Z, Y)); i_g = float(pol("i_g", logK, Z, Y))
        f, c = felicity(logK, Z, Y, i_d, i_g)
        J += np.exp(-dl * t) * f * dt
        if ci < len(CHECKPTS) and t >= CHECKPTS[ci]:
            Jckpt[CHECKPTS[ci]] = J; ci += 1
        if t_exit is None and Y > 4.0:
            t_exit = t
        if s % int(25 / dt) == 0:
            traj.append((t, logK, Z, Y, i_d, i_g, c))
        # RK4 on the drift ODE
        def rhs(lk, z, yy):
            idl = float(pol("i_d", lk, z, yy)); igl = float(pol("i_g", lk, z, yy))
            return drifts(lk, z, yy, idl, igl)
        k1 = rhs(logK, Z, Y)
        k2 = rhs(logK + 0.5*dt*k1[0], Z + 0.5*dt*k1[1], Y + 0.5*dt*k1[2])
        k3 = rhs(logK + 0.5*dt*k2[0], Z + 0.5*dt*k2[1], Y + 0.5*dt*k2[2])
        k4 = rhs(logK + dt*k3[0], Z + dt*k3[1], Y + dt*k3[2])
        logK += dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        Z    += dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1]); Z = min(max(Z, 1e-4), 1-1e-4)
        Y    += dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
        t += dt
    for cp in CHECKPTS:
        Jckpt.setdefault(cp, J)
    return J, traj, Jckpt, t_exit

def main():
    # fine NN domain must cover the Y-excursion of the paths (Y drifts UP, a_Y>0)
    lk = np.linspace(3.5, 8.0, 46); z = np.linspace(0.03, 0.97, 64); y = np.linspace(0.0, 7.0, 71)
    nn = build_nn_grid(lk, z, y)
    # SAVE the NN control+value grid so downstream verification needs no TensorFlow
    np.savez(os.path.join(OD, "nn_control_grid.npz"), logK=lk, Z=z, Y=y,
             i_d=nn.f["i_d"], i_g=nn.f["i_g"], v=nn.f["v"])
    print(f"[saved] nn_control_grid.npz", flush=True)
    fd = load_fd_grid(os.path.join(OD, "fd_pdpt_v2_lam3_0167_xi148.npz"))

    refs = [(C.logK_fix, 0.7, 3.0), (C.logK_fix, 0.7, 1.0), (C.logK_fix, 0.5, 2.0), (C.logK_fix, 0.5, 0.5)]
    print("\n" + "=" * 78)
    print("KEYSTONE: realized discounted welfare J under each feedback policy (higher=better)")
    print("=" * 78)

    def show_traj(tag, traj, t_exit):
        ex = f"Y exits [0,4] at t={t_exit:.0f}y" if t_exit is not None else "Y stays in [0,4]"
        print(f"  {tag} trajectory ({ex}):")
        print(f"    {'t':>5} {'logK':>7} {'Z':>7} {'Y':>9} {'i_d':>8} {'i_g':>8} {'c':>8}")
        for (t, lK, Z, Y, idv, igv, c) in traj[:13]:
            print(f"    {t:5.0f} {lK:7.2f} {Z:7.3f} {Y:9.2f} {idv:+8.4f} {igv:+8.4f} {c:8.4f}")

    for x0 in refs:
        idn = float(nn("i_d", *x0)); ign = float(nn("i_g", *x0))
        idf = float(fd("i_d", *x0)); igf = float(fd("i_g", *x0))
        Jn, tn, Jck_n, te_n = evaluate_policy(nn, x0, name="NN")
        Jf, tf, Jck_f, te_f = evaluate_policy(fd, x0, name="FD")
        winner = "NN (no de-invest)" if Jn > Jf else "FD (de-invest)"
        print(f"\n--- x0: logK={x0[0]:.2f}, Z={x0[1]}, Y={x0[2]} ---")
        print(f"  NN policy at x0: i_d={idn:+.4f} i_g={ign:+.4f}  (full-horizon J = {Jn:.4g})")
        print(f"  FD policy at x0: i_d={idf:+.4f} i_g={igf:+.4f}  (full-horizon J = {Jf:.4g})")
        print(f"  cumulative discounted welfare J(0->T):")
        print(f"    {'T(yr)':>7} " + " ".join(f"{cp:>11}" for cp in CHECKPTS))
        print(f"    {'NN':>7} " + " ".join(f"{Jck_n[cp]:11.4g}" for cp in CHECKPTS))
        print(f"    {'FD':>7} " + " ".join(f"{Jck_f[cp]:11.4g}" for cp in CHECKPTS))
        print(f"  WINNER: {winner}")
        show_traj("NN", tn, te_n)
        show_traj("FD", tf, te_f)
    print("\nRead the J(0->T) row: if NN stays close to FD up to the domain edge and only diverges")
    print("AFTER Y exits [0,4], the runaway is an extrapolation tail; if NN is already worse while")
    print("BOTH are still in-domain, de-investment is optimal on calibrated ground, not extrapolation.")

if __name__ == "__main__":
    main()
