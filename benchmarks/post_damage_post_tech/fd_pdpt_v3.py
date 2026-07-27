"""
Robust FD for the post-damage post-tech climate HJB, using the expert team's PROVEN
monotone scheme but a DIRECT sparse solve (sparse LU) for the frozen-policy linear system
inside Howard policy iteration -- so it cannot diverge (unlike the ADI inner solver in v2).

Scheme (from the team's audit, all in derivation.tex):
  - Howard policy iteration: freeze FOC-optimal controls (q under-relaxed) -> linear PDE -> solve.
  - Implicit M-matrix A = delta*I - L: upwind drift (a>0 fwd, a<0 bwd) + central diffusion.
  - Cross term v_logKZ lagged to the RHS (small: |cross*v_KZ|~5e-6, contraction 0.10).
  - Boundaries: OUTFLOW at Y=4 (a_Y=thbar*E>0 everywhere -> backward upwind, NO diffusion);
    Neumann (one-sided drift, zero 2nd-deriv) at Y=0, logK ends; Z=0,1 auto-degenerate.
Direct solve via scipy splu (RCM-reordered). Login/compute node, numpy/scipy only.
"""
import os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import reverse_cuthill_mckee

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models"))
from params import PARAMS  # noqa: E402

P = dict(delta=PARAMS["δ"], A_d=PARAMS["A_d"], A_gpp=PARAMS["A_g_prime_prime"],
         a_d=PARAMS["α_d"], G_d=PARAMS["Γ_d"], t_d=PARAMS["θ_d"], s_d=PARAMS["σ_d"],
         a_g=PARAMS["α_g"], G_g=PARAMS["Γ_g"], t_g=PARAMS["θ_g"], s_g=PARAMS["σ_g"],
         thbar=PARAMS["θ_bar"], eta=PARAMS["η"], vars=PARAMS["ϛ"],
         l1=PARAMS["λ1"], l2=PARAMS["λ2"], y_up=PARAMS["y_upper"])
QFLOOR = 0.02


def controls(qd, qg, Z, p):
    Abar = (1 - Z) * p["A_d"] + Z * p["A_gpp"]
    num = p["delta"] * (Abar + (1 - Z) / p["t_d"] + Z / p["t_g"])
    den = p["delta"] + (1 - Z) * p["G_d"] * qd + Z * p["G_g"] * qg
    c = num / den
    i_d = p["G_d"] * qd * c / p["delta"] - 1.0 / p["t_d"]
    i_g = p["G_g"] * qg * c / p["delta"] - 1.0 / p["t_g"]
    return i_d, i_g, c


def _central(v, axis, dx):
    g = np.zeros_like(v)
    lo = [slice(None)]*3; hi = [slice(None)]*3; md = [slice(None)]*3
    lo[axis] = slice(2, None); hi[axis] = slice(0, -2); md[axis] = slice(1, -1)
    g[tuple(md)] = (v[tuple(lo)] - v[tuple(hi)]) / (2*dx)
    e0 = [slice(None)]*3; e1 = [slice(None)]*3; en = [slice(None)]*3; en1 = [slice(None)]*3
    e0[axis] = 0; e1[axis] = 1; en[axis] = -1; en1[axis] = -2
    g[tuple(e0)] = (v[tuple(e1)] - v[tuple(e0)]) / dx
    g[tuple(en)] = (v[tuple(en)] - v[tuple(en1)]) / dx
    return g


def solve(lam3=1/6.0, xi=148.4, nK=21, nZ=25, nY=21, howard_max=60,
          kappa=0.4, tol=1e-6, verbose=True):
    p = P; delta = p["delta"]
    logK = np.linspace(4.0, 7.0, nK); dK = logK[1]-logK[0]
    Z = np.linspace(0.02, 0.98, nZ); dZ = Z[1]-Z[0]
    Y = np.linspace(0.0, 4.0, nY); dY = Y[1]-Y[0]
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    K = np.exp(LK); E = p["eta"]*p["A_d"]*(1-ZZ)*K
    sd2, sg2 = p["s_d"]**2, p["s_g"]**2
    lNy = p["l1"] + p["l2"]*YY + lam3*(YY - p["y_up"]); lNyy = p["l2"] + lam3
    inv_xi = 0.0 if not np.isfinite(xi) else 1.0/xi
    N = nK*nZ*nY
    ii, jj, kk = np.unravel_index(np.arange(N), (nK, nZ, nY))

    v = 0.5*LK + 1.0
    qd_s = np.maximum(_central(v, 0, dK) - ZZ*_central(v, 1, dZ), QFLOOR)
    qg_s = np.maximum(_central(v, 0, dK) + (1-ZZ)*_central(v, 1, dZ), QFLOOR)
    t0 = time.time(); it = 0; maxR = np.inf

    for it in range(howard_max):
        vlK = _central(v, 0, dK); vZ = _central(v, 1, dZ); vY = _central(v, 2, dY)
        qd_s = (1-kappa)*qd_s + kappa*np.maximum(vlK - ZZ*vZ, QFLOOR)
        qg_s = (1-kappa)*qg_s + kappa*np.maximum(vlK + (1-ZZ)*vZ, QFLOOR)
        i_d, i_g, c = controls(qd_s, qg_s, ZZ, p)
        phid = p["a_d"] + p["G_d"]*np.log(np.maximum(1+p["t_d"]*i_d, 1e-9))
        phig = p["a_g"] + p["G_g"]*np.log(np.maximum(1+p["t_g"]*i_g, 1e-9))
        Dc = sd2*(1-ZZ)**2 + sg2*ZZ**2
        a_lK = (1-ZZ)*phid + ZZ*phig - Dc/2.0;  b_lK = Dc/2.0
        a_Z = ZZ*(1-ZZ)*(phig-phid+(1-ZZ)*sd2-ZZ*sg2);  b_Z = 0.5*ZZ**2*(1-ZZ)**2*(sd2+sg2)
        a_Y = p["thbar"]*E;  b_Y = 0.5*p["vars"]**2*E**2
        cross = -ZZ*(1-ZZ)**2*sd2 + ZZ**2*(1-ZZ)*sg2
        E_d = (1-ZZ)*p["s_d"]*qd_s; E_g = ZZ*p["s_g"]*qg_s; E_y = p["vars"]*E*(vY-lNy)
        robust = -0.5*inv_xi*(E_d**2+E_g**2+E_y**2)
        damage = -(lNy*a_Y + lNyy*b_Y)
        flow = delta*(np.log(np.maximum(c, 1e-12)) + LK)
        # lagged cross term -> RHS
        vKZ = np.zeros_like(v)
        vKZ[1:-1,1:-1,:] = (v[2:,2:,:]-v[2:,:-2,:]-v[:-2,2:,:]+v[:-2,:-2,:])/(4*dK*dZ)
        rhs = (flow + robust + damage + cross*vKZ).ravel()

        # assemble A = delta*I - L  (upwind drift + central diffusion, M-matrix)
        rows = [np.arange(N)]; cols = [np.arange(N)]; data = [np.full(N, delta)]

        def add(axis, a, b, dx, npts, outflow_top=False):
            af = (a > 0).ravel(); a1 = np.abs(a).ravel()/dx; bd = b.ravel()/dx**2
            idx = [ii, jj, kk]
            top = idx[axis] == npts-1; bot = idx[axis] == 0
            # forward / backward neighbor indices (clamped)
            fwd = [ii, jj, kk]; fwd[axis] = np.minimum(idx[axis]+1, npts-1); nF = np.ravel_multi_index(fwd,(nK,nZ,nY))
            bwd = [ii, jj, kk]; bwd[axis] = np.maximum(idx[axis]-1, 0); nB = np.ravel_multi_index(bwd,(nK,nZ,nY))
            bd_eff = bd.copy()
            if outflow_top:
                bd_eff = np.where(top, 0.0, bd_eff)        # no diffusion at outflow face
            cF = -((af)*a1 + bd_eff)                        # M offdiag forward
            cB = -((~af)*a1 + bd_eff)                       # M offdiag backward
            if outflow_top:
                # at outflow top with a>0: force BACKWARD upwind (use k-1), no forward flux
                topf = top & af
                cF = np.where(topf, 0.0, cF)
                cB = np.where(topf, -(a1), cB)
            dC = a1 + 2.0*bd_eff
            if outflow_top:
                dC = np.where(top & af, a1, dC)             # only backward drift on diagonal at outflow
            rows.append(np.arange(N)); cols.append(nF); data.append(cF)
            rows.append(np.arange(N)); cols.append(nB); data.append(cB)
            rows.append(np.arange(N)); cols.append(np.arange(N)); data.append(dC)

        add(0, a_lK, b_lK, dK, nK)
        add(1, a_Z, b_Z, dZ, nZ)
        add(2, a_Y, b_Y, dY, nY, outflow_top=True)
        A = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(N, N)).tocsr()
        # RCM reorder + direct LU (robust, fast)
        perm = reverse_cuthill_mckee(A, symmetric_mode=False)
        iperm = np.argsort(perm)
        lu = splu(A[perm][:, perm].tocsc())
        v_new = lu.solve(rhs[perm])[iperm].reshape(nK, nZ, nY)

        # HJB residual (central) for monitoring
        R = (A @ v_new.ravel() - rhs)
        maxR = float(np.max(np.abs(R)))
        dv = np.max(np.abs(v_new - v)); v = v_new
        if verbose and (it % 5 == 0 or it == howard_max-1):
            ik = np.argmin(np.abs(logK-np.log(880))); jz = np.argmin(np.abs(Z-0.7)); ky = np.argmin(np.abs(Y-3.0))
            print(f"  [howard {it:3d}] linR={maxR:.2e} dv={dv:.2e} | i_d={i_d[ik,jz,ky]:+.4f} "
                  f"i_g={i_g[ik,jz,ky]:+.4f} vlK={vlK[ik,jz,ky]:.3f} c={c[ik,jz,ky]:.4f}", flush=True)
        if dv < tol and it > 3:
            break

    vlK = _central(v, 0, dK); vZ = _central(v, 1, dZ); vY = _central(v, 2, dY)
    qd = np.maximum(vlK - ZZ*vZ, QFLOOR); qg = np.maximum(vlK + (1-ZZ)*vZ, QFLOOR)
    i_d, i_g, c = controls(qd, qg, ZZ, p)
    # true central-difference HJB residual
    R = _hjb_residual(v, vlK, vZ, vY, i_d, i_g, c, qd, qg, ZZ, E, lNy, lNyy, inv_xi, LK,
                      dK, dZ, dY, p)
    return dict(logK=logK, Z=Z, Y=Y, v=v, i_d=i_d, i_g=i_g, c=c, vlK=vlK, vZ=vZ, vY=vY,
                iters=it+1, time=time.time()-t0, max_abs_residual=float(np.max(np.abs(R))))


def _hjb_residual(v, vlK, vZ, vY, i_d, i_g, c, qd, qg, ZZ, E, lNy, lNyy, inv_xi, LK, dK, dZ, dY, p):
    def d2(axis, dx):
        g = np.zeros_like(v); lo=[slice(None)]*3; hi=[slice(None)]*3; md=[slice(None)]*3
        lo[axis]=slice(2,None); hi[axis]=slice(0,-2); md[axis]=slice(1,-1)
        g[tuple(md)] = (v[tuple(lo)]-2*v[tuple(md)]+v[tuple(hi)])/dx**2
        return g
    sd2, sg2 = p["s_d"]**2, p["s_g"]**2
    vKK=d2(0,dK); vZZ=d2(1,dZ); vYY=d2(2,dY)
    vKZ=np.zeros_like(v); vKZ[1:-1,1:-1,:]=(v[2:,2:,:]-v[2:,:-2,:]-v[:-2,2:,:]+v[:-2,:-2,:])/(4*dK*dZ)
    phid=p["a_d"]+p["G_d"]*np.log(np.maximum(1+p["t_d"]*i_d,1e-9))
    phig=p["a_g"]+p["G_g"]*np.log(np.maximum(1+p["t_g"]*i_g,1e-9))
    Dc=sd2*(1-ZZ)**2+sg2*ZZ**2
    a_lK=(1-ZZ)*phid+ZZ*phig-Dc/2.0; a_Z=ZZ*(1-ZZ)*(phig-phid+(1-ZZ)*sd2-ZZ*sg2)
    a_Y=p["thbar"]*E; b_Y=0.5*p["vars"]**2*E**2
    cross=-ZZ*(1-ZZ)**2*sd2+ZZ**2*(1-ZZ)*sg2
    E_d=(1-ZZ)*p["s_d"]*qd; E_g=ZZ*p["s_g"]*qg; E_y=p["vars"]*E*(vY-lNy)
    robust=-0.5*inv_xi*(E_d**2+E_g**2+E_y**2); damage=-(lNy*a_Y+lNyy*b_Y)
    R=(p["delta"]*(np.log(np.maximum(c,1e-12))+LK)-p["delta"]*v
       +a_lK*vlK+(Dc/2.0)*vKK+a_Z*vZ+0.5*ZZ**2*(1-ZZ)**2*(sd2+sg2)*vZZ
       +cross*vKZ+a_Y*vY+b_Y*vYY+robust+damage)
    return R[2:-2,2:-2,1:-2]    # interior, excluding the outflow face


if __name__ == "__main__":
    out = solve(verbose=True)
    print(f"done in {out['iters']} Howard iters, {out['time']:.0f}s, max|resid|={out['max_abs_residual']:.3e}")
    OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"); os.makedirs(OD, exist_ok=True)
    np.savez(os.path.join(OD, "fd_pdpt_v3_lam3_0167_xi148.npz"),
             **{k: out[k] for k in ("logK","Z","Y","v","i_d","i_g","c","vlK","vZ","vY")})
    ik=np.argmin(np.abs(out["logK"]-np.log(880))); jz=np.argmin(np.abs(out["Z"]-0.7)); ky=np.argmin(np.abs(out["Y"]-3.0))
    lNy=0.00017675+2*0.0022*3.0+(1/6.0)*(3.0-2.5)
    print(f"[logK=6.78,Z=0.7,Y=3.0] i_d={out['i_d'][ik,jz,ky]:+.4f} i_g={out['i_g'][ik,jz,ky]:+.4f} "
          f"vlK={out['vlK'][ik,jz,ky]:.3f} c={out['c'][ik,jz,ky]:.4f} V_Y={out['vY'][ik,jz,ky]-lNy:+.4f}")
    print("TARGET (NN): i_d~0.040 i_g~0.104 vlK~0.54 c~0.064 V_Y~-0.16")
