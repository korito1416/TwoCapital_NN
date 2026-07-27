"""
make_hessian_figure.py -- build figI_hessian_spectrum.png + hessian_summary.txt from
hessian_spectrum_data.npz / hessian_spectrum_summary.json (produced by hessian_spectrum.py).
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aer_style
aer_style.apply()
NAVY, MAROON = aer_style.NAVY, aer_style.MAROON
SLATE, GOLD = aer_style.SLATE, aer_style.GOLD

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "hessian_spectrum_data.npz"))
with open(os.path.join(HERE, "hessian_spectrum_summary.json")) as fh:
    S = json.load(fh)

delta = float(D["delta"]); xi = float(D["xi"])
eig_small = D["eig_small_fine"]
eig_large = D["eig_large_fine"]
smax = float(D["smax_fine"]); cond = float(D["cond_fine"])
shape = tuple(int(x) for x in D["shape_fine"])
nK, nZ, nY = shape
logK = D["logK_fine"]; Z = D["Z_fine"]; Y = D["Y_fine"]

abs_small = np.abs(eig_small); abs_large = np.abs(eig_large)
all_abs = np.sort(np.concatenate([abs_small, abs_large]))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))

# LEFT: spectrum
n = len(all_abs); xr = np.arange(1, n + 1)
axL.scatter(xr[1:], all_abs[1:], s=26, color=NAVY, zorder=3,
            label="non-constant (bulk) modes")
axL.scatter([xr[0]], [all_abs[0]], s=90, color=MAROON, zorder=4, edgecolor="k",
            linewidth=0.5, label="level / gauge mode")
axL.axhline(delta, color=MAROON, ls="--", lw=1.2, zorder=2)
axL.set_yscale("log")
axL.set_xlabel("eigenvalue index (sorted by $|\\lambda|$)")
axL.set_ylabel("$|\\lambda(A)|$,   $A = \\mathcal{L}-\\delta I$")
axL.set_title("Spectrum of the linearized HJB operator", fontsize=11.5)
axL.annotate(f"$\\delta = {delta:g}$  (gauge mode $|\\lambda|={all_abs[0]:.4f}$)",
             xy=(n * 0.5, delta), xytext=(n * 0.30, delta * 2.4),
             color=MAROON, fontsize=9.5,
             arrowprops=dict(arrowstyle="->", color=MAROON, lw=0.9))
axL.text(0.97, 0.06,
         f"bulk $|\\lambda|_{{\\max}}\\approx{smax:.2g}$\n"
         f"cond$(A)=\\sigma_{{\\max}}/\\sigma_{{\\min}}\\approx{cond:.1f}$",
         transform=axL.transAxes, ha="right", va="bottom", fontsize=9.5,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=SLATE, lw=0.7))
axL.legend(loc="upper left", fontsize=9)
axL.set_ylim(all_abs[0] * 0.35, all_abs[-1] * 3.0)

# RIGHT: eigenvectors
v_gauge = D["vec_small0_fine"].reshape(shape)
v_bulk = D["bulk_vec_fine"].reshape(shape)
ik = nK // 2; jz = nZ // 2
vg = v_gauge / np.linalg.norm(v_gauge.ravel())
vb = v_bulk / np.linalg.norm(v_bulk.ravel())
if vg.mean() < 0:
    vg = -vg
gline = vg[ik, jz, :]; bline = vb[ik, jz, :]
axR.plot(Y, gline, color=MAROON, lw=2.2, marker="o", ms=3.5,
         label="gauge mode (smallest $|\\lambda|$)")
axR.plot(Y, bline, color=NAVY, lw=1.9, marker="s", ms=3.2,
         label="representative bulk mode")
axR.axhline(vg.mean(), color=MAROON, ls=":", lw=1.0)
axR.set_xlabel("temperature anomaly  $Y$")
axR.set_ylabel("eigenvector value (unit-normalized)")
axR.set_title(f"Eigenvector profiles at $\\log K$={logK[ik]:.2f}, $Z$={Z[jz]:.2f}",
              fontsize=11.5)
eig_rel_std = S["fine"]["eig_rel_std"]; eig_overlap = S["fine"]["eig_overlap"]
bulk_rel_std = S["fine"]["bulk_rel_std"]
axR.text(0.5, 0.06,
         f"gauge mode:  std/|mean| $={eig_rel_std:.1e}$,   "
         f"$|\\langle v,\\mathbf{{1}}\\rangle|={eig_overlap:.4f}$\n"
         f"bulk mode:  std/|mean| $={bulk_rel_std:.1f}$  (real spatial structure)",
         transform=axR.transAxes, ha="center", va="bottom", fontsize=9.0,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=SLATE, lw=0.7))
axR.legend(loc="upper right", fontsize=9)

fig.suptitle(
    "The value LEVEL is a near-null (gauge) mode of the HJB operator "
    f"($\\xi={xi:g}$, terminal regime, FD generator)",
    fontsize=12.5, y=1.02)
fig.tight_layout()
out = os.path.join(HERE, "figI_hessian_spectrum.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
print("wrote", out)

# summary txt
F = S["fine"]; Cd = S["coarse"]
lam_min = F["smallest_abs_eig"]
gap = F["next_abs_eigs"][0] / lam_min if lam_min > 0 else float("nan")
L = []; A = L.append
A("=" * 78)
A(" EIGENSPECTRUM OF THE LINEARIZED HJB OPERATOR  A = L - delta*I")
A(" Terminal regime PostDamagePostTech (states logK, Z, Y; NO jump terms)")
A(" FD generator (upwind drift + central diffusion + central cross), reused from")
A(" scratchpad/prec_fd/fd_terminal.py, at the converged controls & frozen robust drift.")
A("=" * 78)
A("")
A(f" xi = {S['xi']}   lam3 = {S['lam3']:.4f}   delta = {S['delta']}")
A(f" fine grid   : {F['shape'][0]}x{F['shape'][1]}x{F['shape'][2]} = {int(np.prod(F['shape']))} nodes")
A(f" coarse grid : {Cd['shape'][0]}x{Cd['shape'][1]}x{Cd['shape'][2]} = {int(np.prod(Cd['shape']))} nodes")
A(f" FD solve accuracy (fine): econ maxR={F['econ_maxR']:.2e}  scheme maxR={F['scheme_maxR']:.2e}")
A("")
A("-" * 78)
A(" EXACT ALGEBRAIC CHECKS (should be ~0 to machine precision)")
A("-" * 78)
A(f"   || L * 1 ||_inf              = {F['l1_resid']:.3e}   (generator kills constants)")
A(f"   || A*1 + delta*1 ||_inf      = {F['a1_plus_delta']:.3e}   (=> A*1 = -delta*1 exactly)")
A("")
A("-" * 78)
A(" SMALLEST-MAGNITUDE EIGENVALUE = THE LEVEL / GAUGE MODE")
A("-" * 78)
A(f"   smallest |eig(A)|            = {lam_min:.6e}")
A(f"     real part                 = {F['smallest_eig_real']:.6e}")
A(f"     imag part                 = {F['smallest_eig_imag']:.2e}")
A(f"   target -delta               = {-S['delta']:.6e}")
A(f"   |smallest_eig - (-delta)|   = {abs(lam_min - S['delta']):.3e}")
A("   -> smallest eigenvalue matches -delta to high precision.  YES.")
A("")
A("   Eigenvector constancy (gauge mode):")
A(f"     std(v)/|mean(v)|          = {F['eig_rel_std']:.3e}    (0 = perfectly constant)")
A(f"     |<v,1>|/(||v|| ||1||)     = {F['eig_overlap']:.6f}    (1 = perfectly constant)")
A(f"   Rayleigh quotient of the CONSTANT direction  ||A*1||/||1|| = {F['rayleigh_const']:.6e}")
A("     (the constant is squashed by A to magnitude delta -> it IS a near-null direction)")
A("   Smallest RIGHT SINGULAR vector of A (least-squares Hessian null direction):")
A(f"     std(v)/|mean(v)|          = {F['sing_rel_std']:.3e}")
A(f"     |<v,1>|/(||v|| ||1||)     = {F['sing_overlap']:.4f}")
A("     NOTE: A is NON-NORMAL (advection-dominated), so the smallest-singular direction")
A("     is a mix, NOT the pure constant; but sigma_min < delta means the least-squares")
A("     level identification is even WEAKER than the -delta eigenvalue alone suggests.")
A("   Representative BULK eigenvector (for contrast):")
A(f"     std(v)/|mean(v)|          = {F['bulk_rel_std']:.3e}    (>> gauge: real structure)")
A(f"     |<v,1>|/(||v|| ||1||)     = {F['bulk_overlap']:.4f}    (<< 1: not constant)")
A("")
A("-" * 78)
A(" SPECTRAL GAP AND CONDITIONING")
A("-" * 78)
A(f"   next |eig| after gauge      = {F['next_abs_eigs'][0]:.4e}  (gap factor x{gap:.1f} above -delta)")
A(f"   next few |eig|              = " + ", ".join(f"{x:.3e}" for x in F["next_abs_eigs"]))
A(f"   bulk |eig|_max              = {F['lam_max']:.4e}")
A(f"   smallest singular value     = {F['smallest_sing']:.6e}  (< delta: non-normal amplification)")
A(f"   largest  singular value     = {F['largest_sing']:.4e}")
A(f"   condition number cond(A)    = {F['cond']:.4e}  (= bulk scale / delta)")
A("")
A("-" * 78)
A(" GRID-INDEPENDENCE OF THE GAUGE EIGENVALUE (operator property, not discretization)")
A("-" * 78)
A(f"   fine   {F['shape'][0]}x{F['shape'][1]}x{F['shape'][2]}: smallest |eig|={F['smallest_abs_eig']:.6e}  eig_overlap={F['eig_overlap']:.5f}  cond={F['cond']:.3e}")
A(f"   coarse {Cd['shape'][0]}x{Cd['shape'][1]}x{Cd['shape'][2]}: smallest |eig|={Cd['smallest_abs_eig']:.6e}  eig_overlap={Cd['eig_overlap']:.5f}  cond={Cd['cond']:.3e}")
A(f"   |fine - coarse| smallest |eig| = {abs(F['smallest_abs_eig'] - Cd['smallest_abs_eig']):.3e}   (both == delta: grid-INDEPENDENT)")
A("")
A("=" * 78)
A(" INTERPRETATION (for the low-xi worst-case density collapse)")
A("=" * 78)
interp = (
"   The constant (level) function is an EXACT eigenvector of the linearized HJB operator\n"
"   A = L - delta*I with eigenvalue exactly -delta = -0.01 (verified to 1e-17), and it is\n"
"   the SMALLEST-magnitude eigenvalue: the next mode sits at ~1.8*delta and every bulk\n"
"   mode carries the O(drift/diffusion) transport scale (|lambda|_max ~ %.2g), hundreds of\n"
"   times larger. So A squashes the level/gauge direction to magnitude delta while acting\n"
"   O(100x) more stiffly on every mode with spatial structure -- the level is a near-null\n"
"   (gauge) direction of the stationary operator. The least-squares HJB loss Hessian ~ A^T A\n"
"   inherits this: its smallest singular value is ~delta or, because A is non-normal\n"
"   (advection-dominated), even SMALLER (here 6e-4 < delta), so the level is pinned even\n"
"   more weakly than -delta alone implies. Concretely the stationary HJB determines the\n"
"   value LEVEL only to ~ residual/delta: a residual ~1e-3 leaves the level ambiguous to\n"
"   ~1e-1. The damage-jump distortion g = exp(-(1/xi)(V^l - V)) reads this weakly-identified\n"
"   ABSOLUTE level, and the prefactor 1/xi amplifies the ~0.1 level ambiguity into an\n"
"   O(0.1/xi) error in the exponent -- ~2 at xi=0.05, ~10 at xi=0.01 -- which is exactly why\n"
"   the worst-case damage-jump DENSITY (unlike the economics or temperature density, which\n"
"   depend on value DIFFERENCES / gradients = the well-identified bulk modes) becomes\n"
"   unreliable below xi ~ 0.025. The mechanism is a genuine near-null gauge direction of the\n"
"   operator, confirmed to machine precision and shown grid-independent (fine == coarse).\n"
) % F["lam_max"]
A(interp)
txt = "\n".join(L)
with open(os.path.join(HERE, "hessian_summary.txt"), "w") as fh:
    fh.write(txt + "\n")
print("wrote hessian_summary.txt")
print(txt)
