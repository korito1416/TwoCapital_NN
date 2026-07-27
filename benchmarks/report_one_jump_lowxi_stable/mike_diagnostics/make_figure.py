"""Figure: per-xi HJB residual vs 1/xi (two regimes) + the g-distortion exponent amplification."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/report_one_jump_lowxi_stable/mike_diagnostics"

res = json.load(open(os.path.join(OUT, "per_xi_residual.json")))
gd  = json.load(open(os.path.join(OUT, "per_xi_g_distortion.json")))

# order by xi ascending -> 1/xi descending; use grid order
labels = ["inf(logxi=5)","0.1","0.05","0.04","0.03","0.025","0.02","0.01","0.005"]
def get(regime, label):
    return res[regime][label]
xis   = np.array([res["PreDamagePreTech"][l]["xi"] for l in labels])
inv_xi= 1.0/xis

pdpt_box  = np.array([get("PreDamagePreTech", l)["full_box"]["loss_v_rms"] for l in labels])
podt_box  = np.array([get("PostDamagePostTech", l)["full_box"]["loss_v_rms"] for l in labels])
pdpt_econ = np.array([get("PreDamagePreTech", l).get("econ_region",{}).get("loss_v_rms", np.nan) for l in labels])
foc_pdpt  = np.array([get("PreDamagePreTech", l)["full_box"]["FOC_max"] for l in labels])
foc_podt  = np.array([get("PostDamagePostTech", l)["full_box"]["FOC_max"] for l in labels])
exp_absmax= np.array([gd[l]["dmg_exp_absmax"] for l in labels])
dV_rms    = np.array([gd[l]["dmg_dV_rms"] for l in labels])

fig, axes = plt.subplots(1, 2, figsize=(13,5))

ax = axes[0]
ax.plot(inv_xi, pdpt_box, "o-", color="C0", label="PreDamagePreTech HJB residual (full box)")
ax.plot(inv_xi, pdpt_econ,"s--",color="C0", alpha=0.6, label="PreDamagePreTech (econ region)")
ax.plot(inv_xi, podt_box, "o-", color="C1", label="PostDamagePostTech HJB residual (full box)")
ax.plot(inv_xi, foc_pdpt, "^:", color="C2", label="PreDamagePreTech FOC error (max)")
ax.plot(inv_xi, foc_podt, "v:", color="C3", label="PostDamagePostTech FOC error (max)")
ax.axhline(1e-3, color="k", ls="--", lw=1, label="paper accuracy 1e-3")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$1/\xi$ (uncertainty aversion)")
ax.set_ylabel("RMS residual / FOC error")
ax.set_title("HJB residual & FOC error vs $1/\\xi$\n(raw, un-rescaled; float64 one-jump $\\pi=1$)")
ax.grid(True, which="both", alpha=0.3)
for x, l in zip(inv_xi, labels):
    xi = float(l) if l[0].isdigit() else np.exp(5.0)
ax.legend(fontsize=8, loc="upper left")
# annotate xi values on top axis
ax2 = ax.secondary_xaxis("top")
ax2.set_xticks(inv_xi)
ax2.set_xticklabels([f"{x:.3g}" for x in xis], rotation=45, fontsize=7)
ax2.set_xlabel(r"$\xi$", fontsize=8)

ax = axes[1]
ax.plot(inv_xi, exp_absmax, "o-", color="C4",
        label=r"max $|{-}\frac{1}{\xi}(V^{\ell}-V)|$ (g exponent)")
ax.plot(inv_xi, dV_rms, "s-", color="C5",
        label=r"RMS $|V^{\ell}-V|$ (value-LEVEL gap, damage jump)")
ax.axhline(88.7, color="r", ls=":", lw=1, label="float32 exp overflow (~88.7)")
ax.axhline(350.0, color="darkred", ls="--", lw=1, label="production clamp cap (350)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$1/\xi$")
ax.set_ylabel("magnitude")
ax.set_title("Why $g=\\exp(-\\frac{1}{\\xi}(V^{\\ell}{-}V))$ becomes fragile:\n"
             "value gap is ~flat, but exponent scales with $1/\\xi$")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
ax3 = ax.secondary_xaxis("top")
ax3.set_xticks(inv_xi); ax3.set_xticklabels([f"{x:.3g}" for x in xis], rotation=45, fontsize=7)
ax3.set_xlabel(r"$\xi$", fontsize=8)

plt.tight_layout()
out = os.path.join(OUT, "figF_per_xi_residual.png")
plt.savefig(out, dpi=150)
print("WROTE", out)
