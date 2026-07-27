"""
hessian_spectrum.py -- EIGENSPECTRUM of the linearized HJB operator for the TERMINAL
regime (PostDamagePostTech: states logK, Z, Y; NO jump terms).

CLAIM UNDER TEST
----------------
The worst-case damage-jump density collapses at low xi because the jump distortion
    g = exp(-(1/xi)(V^l - V))
depends on the ABSOLUTE value LEVEL, which the stationary HJB pins only weakly.

Rigorous statement: linearize the HJB residual R(V) about the converged solution.  With
the controls and the (frozen) robust distortion held fixed, the Jacobian dR/dV is the
LINEAR operator
    A = L - delta * I
where L is the (drift+diffusion) infinitesimal generator and delta = 0.01 the discount
rate.  A generator annihilates constants (L * 1 = 0), because every row of L is a
difference stencil whose weights sum to zero.  Hence
    A * 1 = (L - delta) * 1 = -delta * 1,
i.e. the CONSTANT (level / gauge) function is an exact eigenvector of A with eigenvalue
-delta = -0.01.  Because the drift+diffusion transport scales carry the O(drift/diff)
magnitude (>> delta here), -delta is the SMALLEST-magnitude eigenvalue and its eigenvector
is constant.  Then the least-squares (Gauss-Newton) loss Hessian ~ A^T A has smallest
singular value ~ delta, so the value LEVEL is determined only to ~ residual/delta; the
1/xi factor in g amplifies that into the density collapse.

METHOD
------
We ASSEMBLE the sparse generator L as a scipy.sparse matrix using the EXACT SAME
discretization the FD solver's policy_eval uses (READ from fd_terminal.py, reused here):
  * UPWIND first derivatives in each drift direction (a>=0 -> forward, a<0 -> backward),
  * CENTRAL second derivatives (diffusion),
  * CENTRAL x CENTRAL logK-Z cross term,
  * the solver's reflecting boundary convention (a neighbour beyond the face is replaced
    by the face node itself: Vkp[-1]=V[-1], Vkm[0]=V[0], etc.).
The drift/diffusion coefficients are evaluated at the CONVERGED controls (i_d,i_g) and the
FROZEN worst-case temperature drift a_Y_extra, exactly as the operator the solver inverts.
The robust drag and penalty are ADDITIVE SOURCES (constant in V under the frozen-distortion
linearization) -> they do NOT enter L, so we correctly omit them from the Jacobian.

With this construction L*1 = 0 up to floating-point (the reflecting BC keeps the row-sums
exactly zero at faces too), so A*1 = -delta*1 is verified to machine precision, and we then
confirm it is the smallest-magnitude eigenpair via sparse eigensolves.

numpy / scipy only -- no neural net, no TensorFlow.
"""
import os
import sys
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
FD_DIR = ("/home/kunjianli/claude-1983122066/"
          "-project-lhansen-Cap-damage-TwoStageTechJump-FOCIr-orignal/"
          "b1cc181d-78a1-47f3-9df3-aca58828ef92/scratchpad/prec_fd")
sys.path.insert(0, FD_DIR)
import fd_terminal as fd  # reuse coeffs(), robust_drag(), _grad(), solve(), P

DELTA = fd.P["delta"]  # 0.01


# ---------------------------------------------------------------------------
# Assemble the sparse generator L on the (nK,nZ,nY) grid, matching policy_eval.
# Row index r = ((i*nZ)+j)*nY+k  (numpy C-order flatten of shape (nK,nZ,nY)).
# ---------------------------------------------------------------------------
def assemble_generator(logK, Z, Y, i_d, i_g, aY_extra, lam3, xi, p=fd.P):
    nK, nZ, nY = len(logK), len(Z), len(Y)
    dK = logK[1] - logK[0]; dZ = Z[1] - Z[0]; dY = Y[1] - Y[0]
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")

    C = fd.coeffs(LK, ZZ, YY, i_d, i_g, lam3, p)
    a_lK, a_Z = C["a_lK"], C["a_Z"]
    aY = C["a_Y"] + (0.0 if aY_extra is None else aY_extra)   # frozen worst-case Y drift
    b_lK, b_Z, b_Y, b_KZ = C["b_lK"], C["b_Z"], C["b_Y"], C["b_KZ"]

    N = nK * nZ * nY
    idx = np.arange(N).reshape(nK, nZ, nY)

    rows = []; cols = []; vals = []

    def add(r_idx, c_idx, w):
        rows.append(np.asarray(r_idx).ravel())
        cols.append(np.asarray(c_idx).ravel())
        vals.append(np.asarray(w).ravel())

    # ---- upwind first derivatives (drift transport) ----
    # For drift a in direction with spacing dx:
    #   a>=0 : a*(V_{+1} - V)/dx    -> weight +a/dx on +1 neighbour, -a/dx on self
    #   a<0  : a*(V - V_{-1})/dx    -> weight -a/dx on -1 neighbour, +a/dx on self
    # Reflecting BC: if the neighbour is off-grid, its coefficient folds onto SELF
    #   (because Vkp[-1]=V[-1] etc.), which keeps the row-sum exactly zero.
    def upwind(a, axis, dx):
        ap = np.maximum(a, 0.0); am = np.minimum(a, 0.0)
        # forward neighbour (+1) for ap, backward (-1) for am
        sl_self = idx
        # ---- ap * (V_{+1}-V)/dx ----
        # self coefficient
        add(sl_self, sl_self, -ap / dx)
        # +1 neighbour (fold to self at the high face)
        nb = np.roll(idx, -1, axis=axis)
        w = ap / dx
        # at high face, np.roll wraps to index 0 -> WRONG; use reflecting: neighbour=self
        hi = [slice(None)] * 3; hi[axis] = -1
        nb_fixed = nb.copy(); nb_fixed[tuple(hi)] = idx[tuple(hi)]
        add(sl_self, nb_fixed, w)
        # ---- am * (V-V_{-1})/dx ----
        add(sl_self, sl_self, am / dx)
        nb2 = np.roll(idx, +1, axis=axis)
        w2 = -am / dx
        lo = [slice(None)] * 3; lo[axis] = 0
        nb2_fixed = nb2.copy(); nb2_fixed[tuple(lo)] = idx[tuple(lo)]
        add(sl_self, nb2_fixed, w2)

    upwind(a_lK, 0, dK)
    upwind(a_Z, 1, dZ)
    upwind(aY, 2, dY)

    # ---- central second derivatives (diffusion): b*(V_{+1}-2V+V_{-1})/dx^2 ----
    # policy_eval only applies the central Laplacian on the INTERIOR (via _lap): faces get 0.
    # Reproduce that: interior rows get (+b/dx^2, -2b/dx^2, +b/dx^2); face rows get nothing
    # (the diffusion vanishes at the face in the solver's stencil).  This keeps row-sum 0.
    def central2(b, axis, dx):
        bb = b / dx ** 2
        interior = [slice(None)] * 3
        interior[axis] = slice(1, -1)
        sl = idx[tuple(interior)]
        bsl = bb[tuple(interior)]
        # self
        add(sl, sl, -2.0 * bsl)
        # +1 and -1 neighbours (guaranteed on-grid because interior slice excludes faces)
        nb_p = np.roll(idx, -1, axis=axis)[tuple(interior)]
        nb_m = np.roll(idx, +1, axis=axis)[tuple(interior)]
        add(sl, nb_p, bsl)
        add(sl, nb_m, bsl)

    central2(b_lK, 0, dK)
    central2(b_Z, 1, dZ)
    central2(b_Y, 2, dY)

    # ---- cross term b_KZ * V_{logK,Z}, central x central on interior (matches _cross) ----
    # _cross: g[1:-1,1:-1,:] = (V[2:,2:]-V[2:,:-2]-V[:-2,2:]+V[:-2,:-2])/(4 dK dZ)
    bxz = b_KZ / (4.0 * dK * dZ)
    inter = (slice(1, -1), slice(1, -1), slice(None))
    sl = idx[inter]; bsl = bxz[inter]
    # ++ (i+1,j+1)
    add(sl, idx[2:, 2:, :], +bsl)
    # +- (i+1,j-1)
    add(sl, idx[2:, :-2, :], -bsl)
    # -+ (i-1,j+1)
    add(sl, idx[:-2, 2:, :], -bsl)
    # -- (i-1,j-1)
    add(sl, idx[:-2, :-2, :], +bsl)
    # NOTE: the cross stencil's four weights sum to zero per row, so it does NOT break L*1=0.

    R = np.concatenate(rows); Cc = np.concatenate(cols); Vv = np.concatenate(vals)
    L = sp.coo_matrix((Vv, (R, Cc)), shape=(N, N)).tocsr()
    L.sum_duplicates()
    return L, (nK, nZ, nY), (dK, dZ, dY)


def constancy_metrics(v, N):
    """How constant is eigenvector v?  Return std/|mean|, and cosine overlap with 1."""
    v = np.real_if_close(v).astype(float)
    v = v / (np.linalg.norm(v) + 1e-300)
    ones = np.ones_like(v) / np.sqrt(len(v))
    overlap = abs(float(np.dot(v, ones)))          # |<v,1>|/(||v|| ||1||)
    mean = float(np.mean(v)); std = float(np.std(v))
    rel_std = std / (abs(mean) + 1e-300)
    return dict(rel_std=rel_std, overlap_with_constant=overlap, mean=mean, std=std)


def spectrum_for_grid(xi, lam3, nK, nZ, nY, warm=None, k_small=8, label=""):
    print(f"\n=== assembling operator: grid {nK}x{nZ}x{nY}, xi={xi} {label} ===", flush=True)
    res = fd.solve(xi=xi, lam3=lam3, nK=nK, nZ=nZ, nY=nY,
                   howard_max=80, pe_sweeps=4000, verbose=False, warm=warm)
    print(f"    solved: econ maxR={res['max_abs_residual_econ']:.2e} "
          f"scheme maxR={res['max_abs_residual_scheme']:.2e}", flush=True)

    logK, Z, Y = res["logK"], res["Z"], res["Y"]
    LK, ZZ, YY = np.meshgrid(logK, Z, Y, indexing="ij")
    dK = logK[1] - logK[0]; dZ = Z[1] - Z[0]; dY = Y[1] - Y[0]

    # Reconstruct the FROZEN worst-case temperature drift from the converged V,
    # exactly as robust_from_V does (same cap).
    E_grid = fd.P["eta"] * fd.P["A_d"] * (1 - ZZ) * np.exp(LK)
    aYx_cap = 4.0 * fd.P["thbar"] * float(E_grid.max())
    VlK = fd._grad(res["V"], 0, dK); VZ = fd._grad(res["V"], 1, dZ); VY = fd._grad(res["V"], 2, dY)
    _, aY_extra, _ = fd.robust_drag(VlK, VZ, VY, ZZ, E_grid, xi, fd.P, aYx_cap=aYx_cap)

    L, shape, spacing = assemble_generator(logK, Z, Y, res["i_d"], res["i_g"],
                                           aY_extra, lam3, xi)
    N = L.shape[0]
    A = L - DELTA * sp.identity(N, format="csr")

    # ---- exact check: L*1 = 0, A*1 = -delta*1 ----
    ones = np.ones(N)
    L1 = L @ ones
    A1 = A @ ones
    l1_resid = float(np.max(np.abs(L1)))
    a1_plus_delta = float(np.max(np.abs(A1 + DELTA * ones)))
    # Rayleigh quotient of the CONSTANT direction: ||A*1||/||1|| = delta exactly
    # (the constant is squashed to magnitude delta by A -- it IS a near-null direction,
    #  even though, A being non-normal, the smallest SINGULAR vector is a different mix).
    rayleigh_const = float(np.linalg.norm(A1) / np.linalg.norm(ones))
    print(f"    ||L*1||_inf = {l1_resid:.3e}   ||A*1 + delta*1||_inf = {a1_plus_delta:.3e}"
          f"   ||A*1||/||1|| = {rayleigh_const:.6e}", flush=True)

    # ---- smallest-magnitude eigenvalues of A via shift-invert at sigma=0 ----
    # (A is real, nonsymmetric.)  which='LM', sigma=0 -> eigs closest to 0.
    try:
        eig_small, vec_small = spla.eigs(A.astype(np.float64), k=k_small, sigma=0.0,
                                         which='LM', maxiter=5000, tol=1e-10)
    except Exception as e:
        print(f"    shift-invert eigs failed ({e}); falling back to which='SM'", flush=True)
        eig_small, vec_small = spla.eigs(A.astype(np.float64), k=k_small, which='SM',
                                         maxiter=8000, tol=1e-9)
    order = np.argsort(np.abs(eig_small))
    eig_small = eig_small[order]; vec_small = vec_small[:, order]

    # ---- a few large-magnitude eigenvalues to bound the bulk scale / condition number ----
    eig_large = spla.eigs(A.astype(np.float64), k=6, which='LM', maxiter=5000,
                          tol=1e-8, return_eigenvectors=False)
    lam_max = float(np.max(np.abs(eig_large)))

    # ---- smallest singular values of A via the NORMAL operator AtA = A^T A ----
    # (svds which='SM' is unreliable; shift-invert eigsh on the SPD normal operator
    #  at sigma=0 robustly returns the smallest singular values = sqrt(smallest eig(AtA))
    #  and the corresponding RIGHT singular vectors.)
    Af = A.astype(np.float64)
    AtA = (Af.T @ Af).tocsc()
    try:
        mu_small, Wv = spla.eigsh(AtA, k=6, sigma=0.0, which='LM', maxiter=8000, tol=1e-10)
        mu_small = np.clip(mu_small, 0.0, None)
        s_small = np.sqrt(mu_small)
        s_order = np.argsort(s_small)
        s_small = s_small[s_order]; Wv = Wv[:, s_order]
        v_rightsing = Wv[:, 0]        # right singular vector of the smallest sing. val
    except Exception as e:
        print(f"    eigsh(AtA,SM) failed ({e}); using |eig| proxy for sing. vals", flush=True)
        s_small = np.sort(np.abs(eig_small))[:6]
        v_rightsing = np.real(vec_small[:, 0])

    mu_large = spla.eigsh(AtA, k=1, which='LM', maxiter=5000, tol=1e-8,
                          return_eigenvectors=False)
    smax = float(np.sqrt(max(mu_large[0], 0.0)))
    cond = smax / (s_small[0] + 1e-300)

    # constancy of the smallest eigenvector and smallest right singular vector
    cm_eig = constancy_metrics(vec_small[:, 0], N)
    cm_sing = constancy_metrics(v_rightsing, N)

    # a representative BULK eigenvector (largest |eig| among the small set that is NOT
    # the constant mode) for the contrast panel -- take the last of the small set.
    bulk_vec = np.real(vec_small[:, -1])
    bulk_cm = constancy_metrics(bulk_vec, N)

    print(f"    smallest |eig(A)| = {abs(eig_small[0]):.6e}  (target delta={DELTA})  "
          f"real={eig_small[0].real:.6e} imag={eig_small[0].imag:.2e}", flush=True)
    print(f"    eigvec constancy: rel_std={cm_eig['rel_std']:.3e} overlap|<v,1>|={cm_eig['overlap_with_constant']:.6f}",
          flush=True)
    print(f"    next |eig|: {np.abs(eig_small[1:5])}", flush=True)
    print(f"    smallest sing = {s_small[0]:.6e}  largest sing = {smax:.4e}  cond = {cond:.3e}",
          flush=True)

    return dict(
        res=res, logK=logK, Z=Z, Y=Y, shape=shape, xi=xi, lam3=lam3,
        L=L, A=A,
        l1_resid=l1_resid, a1_plus_delta=a1_plus_delta, rayleigh_const=rayleigh_const,
        eig_small=eig_small, vec_small=vec_small,
        eig_large=eig_large, lam_max=lam_max,
        s_small=s_small, smax=smax, cond=cond, v_rightsing=v_rightsing,
        cm_eig=cm_eig, cm_sing=cm_sing,
        bulk_vec=bulk_vec, bulk_cm=bulk_cm,
    )


if __name__ == "__main__":
    XI = 0.05          # well-converged terminal regime (scheme residual ~1e-6)
    LAM3 = 1 / 6.0

    # Primary grid + a coarser grid for the grid-independence check.
    out_fine = spectrum_for_grid(XI, LAM3, nK=25, nZ=31, nY=25, label="(fine)")
    out_coarse = spectrum_for_grid(XI, LAM3, nK=17, nZ=21, nY=17, label="(coarse)")

    np.savez_compressed(os.path.join(HERE, "hessian_spectrum_data.npz"),
                        eig_small_fine=out_fine["eig_small"],
                        eig_large_fine=out_fine["eig_large"],
                        s_small_fine=out_fine["s_small"],
                        vec_small0_fine=np.real(out_fine["vec_small"][:, 0]),
                        v_rightsing_fine=np.real(out_fine["v_rightsing"]),
                        bulk_vec_fine=out_fine["bulk_vec"],
                        shape_fine=np.array(out_fine["shape"]),
                        logK_fine=out_fine["logK"], Z_fine=out_fine["Z"], Y_fine=out_fine["Y"],
                        eig_small_coarse=out_coarse["eig_small"],
                        s_small_coarse=out_coarse["s_small"],
                        shape_coarse=np.array(out_coarse["shape"]),
                        smax_fine=out_fine["smax"], cond_fine=out_fine["cond"],
                        lam_max_fine=out_fine["lam_max"],
                        delta=DELTA, xi=XI)

    # persist a compact summary dict for the figure + txt builders
    summary = dict(
        xi=XI, lam3=LAM3, delta=DELTA,
        fine=dict(
            shape=list(out_fine["shape"]),
            l1_resid=out_fine["l1_resid"], a1_plus_delta=out_fine["a1_plus_delta"],
            rayleigh_const=out_fine["rayleigh_const"],
            smallest_abs_eig=float(abs(out_fine["eig_small"][0])),
            smallest_eig_real=float(out_fine["eig_small"][0].real),
            smallest_eig_imag=float(out_fine["eig_small"][0].imag),
            next_abs_eigs=[float(x) for x in np.abs(out_fine["eig_small"][1:6])],
            lam_max=out_fine["lam_max"],
            smallest_sing=float(out_fine["s_small"][0]),
            largest_sing=float(out_fine["smax"]),
            cond=float(out_fine["cond"]),
            eig_rel_std=out_fine["cm_eig"]["rel_std"],
            eig_overlap=out_fine["cm_eig"]["overlap_with_constant"],
            sing_rel_std=out_fine["cm_sing"]["rel_std"],
            sing_overlap=out_fine["cm_sing"]["overlap_with_constant"],
            bulk_rel_std=out_fine["bulk_cm"]["rel_std"],
            bulk_overlap=out_fine["bulk_cm"]["overlap_with_constant"],
            econ_maxR=out_fine["res"]["max_abs_residual_econ"],
            scheme_maxR=out_fine["res"]["max_abs_residual_scheme"],
        ),
        coarse=dict(
            shape=list(out_coarse["shape"]),
            smallest_abs_eig=float(abs(out_coarse["eig_small"][0])),
            smallest_eig_real=float(out_coarse["eig_small"][0].real),
            next_abs_eigs=[float(x) for x in np.abs(out_coarse["eig_small"][1:6])],
            smallest_sing=float(out_coarse["s_small"][0]),
            cond=float(out_coarse["cond"]),
            eig_rel_std=out_coarse["cm_eig"]["rel_std"],
            eig_overlap=out_coarse["cm_eig"]["overlap_with_constant"],
        ),
    )
    with open(os.path.join(HERE, "hessian_spectrum_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nWrote hessian_spectrum_data.npz + hessian_spectrum_summary.json")
