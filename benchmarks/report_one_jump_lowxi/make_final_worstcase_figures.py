"""FINAL worst-case (distorted) jump figures for Lars -- one-jump (pi=1) climate NN, float64.

Produces publication-grade figures with the RIGOROUS admissibility test baked in:
  a worst-case damage-jump cumulative probability P_wc must satisfy
        P_undistorted  <=  P_wc  <=  P_floor(xi->0)
  The lower bound P_wc >= P_undistorted is EQUIVALENT to  mean_l g_l >= 1  (proven, Jensen);
  the upper bound P_floor is the xi->0 limit (g->inf inside the damage region => jump fires the
  instant Y enters [y_lower, y_upper]).

Reliable thresholds xi* (below which the raw NN worst case violates the lower bound and is
therefore PROVABLY wrong), determined from admiss_{full,half}.txt:
  FULL : reliable for xi >= 0.025  (already inadmissible at xi = 0.020)
  HALF : reliable for xi >= 0.020  (already inadmissible at xi = 0.015)

Figures:
  F1  damage-jump CDF (both calibrations): reliable solid, unreliable dashed/grey, floor dotted
  F2  ADMISSIBILITY PROOF: terminal P_wc / P_undistorted / P_floor vs xi, with mean_l g_l panel
  F3  tech-jump CDF (both calibrations) -- FULL reliable all xi (monotone down); HALF reliable
      only xi >= xi* (worst-case g>1 reversal below; shown dashed)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 19, "axes.labelsize": 22, "axes.titlesize": 22,
    "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 16,
    "lines.linewidth": 2.8, "figure.dpi": 130, "savefig.bbox": "tight",
})

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_lowxi_float64"
FULL = ROOT + "/OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostFull_logximin_m5p30_LR_warmup_cosine_10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations50000/SimulationDeterministic"
HALF = ROOT + "/OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostHalf_Gamma0p12_Theta8p35_logximin_m5p30_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations50000/SimulationDeterministic"
OUT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/report_one_jump_lowxi/figures"
SCRATCH = "/home/kunjianli/claude-1983122066/-project-lhansen-Cap-damage-TwoStageTechJump-FOCIr-orignal/b1cc181d-78a1-47f3-9df3-aca58828ef92/scratchpad"
os.makedirs(OUT, exist_ok=True)

R1, R2, Y_LOWER, Y_UPPER = 1.5, 0.36, 1.5, 2.5
XISTAR = {"FULL": 0.025, "HALF": 0.020}   # last reliable xi (admissibility holds)

# (folder-label, pretty-label, xi-float)   -- folder uses %.3f so 0.0125 -> "0.013"
XIS = [
    ("148.600", r"$\xi=\infty$ (neutral)",  148.6),
    ("0.050",   r"$\xi=0.05$",              0.050),
    ("0.025",   r"$\xi=0.025$",             0.025),
    ("0.020",   r"$\xi=0.02$",              0.020),
    ("0.015",   r"$\xi=0.015$",             0.015),
    ("0.013",   r"$\xi=0.0125$",            0.0125),
    ("0.010",   r"$\xi=0.01$",              0.010),
    ("0.005",   r"$\xi=0.005$",             0.005),
]
# blue (neutral) -> deep red (most averse)
CMAP = plt.get_cmap("viridis")
COLORS = ["0.45", "#2c7fb8", "#41b6c4", "#7fcdbb", "#fdae61", "#f46d43", "#d73027", "#a50026"]


def load(base, lab, name):
    p = os.path.join(base, f"SimulationOutputs_ξ_{lab}", name + ".txt")
    return np.loadtxt(p) if os.path.exists(p) else None


def reliable(xi, calib):
    return xi >= XISTAR[calib] - 1e-12


def analytic_floor_cdf(base, lab):
    """xi->0 worst-case limit: g->inf inside (y_lower, y_upper) so the damage hazard saturates.
    Competing-risk UNCONDITIONAL cumulative damage-jump probability with the same survival
    accounting as the pipeline (exclusive cumsum), using the saved tech intensity as competing risk."""
    Y = load(base, lab, "Y"); t = load(base, lab, "t")
    if Y is None:
        return None, None
    dt = float(t[1] - t[0])
    tech = load(base, lab, "tech_jump_intensity")
    if tech is None:
        tech = np.zeros_like(Y)
    BIG = 1e6
    lam_d = np.where(Y > Y_LOWER, BIG, 0.0)
    lam = lam_d + tech
    cumhaz = np.concatenate([[0.0], np.cumsum(lam * dt)[:-1]])   # exclusive cumsum
    surv = np.exp(-cumhaz)
    interval = surv * (1.0 - np.exp(-lam * dt))
    interval[-1] = 0.0                                            # horizon mask (no [H,H+dt])
    share = np.where(lam > 0, lam_d / lam, 0.0)
    dmg_interval = share * interval
    cum = np.concatenate([[0.0], np.cumsum(dmg_interval)[:-1]])
    return t, cum


# ============================================================================
# FIGURE 1 -- worst-case DAMAGE-jump cumulative probability
# ============================================================================
fig, ax = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
for c, (base, ctitle, calib) in enumerate([
        (FULL, "Full adjustment cost  ($\\Gamma=0.060,\\ \\theta=16.7$)", "FULL"),
        (HALF, "Half adjustment cost  ($\\Gamma=0.12,\\ \\theta=8.35$)",  "HALF")]):
    a = ax[c]
    for (lab, plab, xi), col in zip(XIS, COLORS):
        cum = load(base, lab, "dmg_jump_prob")
        if cum is None:
            continue
        t = load(base, lab, "t")
        ok = reliable(xi, calib)
        a.plot(t, cum, "-" if ok else "--", color=col, label=plab,
               alpha=1.0 if ok else 0.5, zorder=3 if ok else 2)
    t, cum_lim = analytic_floor_cdf(base, "0.005")
    if t is not None:
        a.plot(t, cum_lim, ":", color="k", linewidth=3.4,
               label=r"analytic $\xi\!\to\!0$ floor")
    a.set_title(ctitle, fontsize=19, pad=10)
    a.set_xlabel("Year")
    if c == 0:
        a.set_ylabel("Cumulative worst-case\ndamage-jump probability")
    a.set_ylim(-0.02, 1.02); a.set_xlim(0, 60)
    a.grid(alpha=0.25)
    a.legend(loc="upper left", framealpha=0.93, ncol=1)
    a.text(0.98, 0.03, f"reliable: $\\xi\\geq{XISTAR[calib]:g}$", transform=a.transAxes,
           ha="right", va="bottom", fontsize=15, color="0.25",
           bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
fig.suptitle("Worst-case (distorted) damage-jump cumulative probability — one-jump ($\\pi=1$) climate model\n"
             "solid = reliable NN;  dashed = NN inadmissible ($P_{\\rm wc}<P_{\\rm undistorted}$, proven wrong);  dotted = analytic $\\xi\\!\\to\\!0$ floor ($P\\!\\to\\!1$)",
             fontsize=17, y=1.01)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "final_fig1_damage_cdf.png"))
print("saved final_fig1_damage_cdf.png")
plt.close(fig)

# ============================================================================
# FIGURE 2 -- ADMISSIBILITY PROOF panel
#   top: terminal P_wc, P_undistorted, P_floor vs xi  (the band [P_undist, P_floor])
#   bottom: mean_l g_l vs xi (crosses 1 at xi*)
# ============================================================================
fig, ax = plt.subplots(2, 2, figsize=(18, 13), sharex="col")
for c, (calib, tag) in enumerate([("FULL", "Full adjustment cost"), ("HALF", "Half adjustment cost")]):
    d = np.loadtxt(os.path.join(SCRATCH, f"admiss_{calib.lower()}.txt"))
    # cols: xi P_undist P_NN P_trueA P_floor abar gmin gmax Ymax admiss
    order = np.argsort(-d[:, 0])
    d = d[order]
    xi = d[:, 0]; P_undist = d[:, 1]; P_NN = d[:, 2]; P_floor = d[:, 4]; abar = d[:, 5]
    rel = xi >= XISTAR[calib] - 1e-12

    a = ax[0][c]
    a.fill_between(xi, P_undist, P_floor, color="0.85", alpha=0.6, zorder=0,
                   label="admissible band\n$[P_{\\rm undist},\\,P_{\\rm floor}]$")
    a.plot(xi, P_floor, ":", color="k", lw=2.6, label=r"$P_{\rm floor}$ ($\xi\!\to\!0$)")
    a.plot(xi, P_undist, "-", color="#1a9850", lw=2.6, marker="^", ms=8, label=r"$P_{\rm undistorted}$ ($g\equiv1$)")
    a.plot(xi, P_NN, "-", color="0.6", lw=1.6, alpha=0.6, zorder=1)
    a.scatter(xi[rel], P_NN[rel], s=150, color="#d73027", marker="o", zorder=5, label=r"$P_{\rm wc}$ NN (reliable)")
    a.scatter(xi[~rel], P_NN[~rel], s=150, facecolors="none", edgecolors="#d73027",
              linewidths=2.6, marker="o", zorder=5, label=r"$P_{\rm wc}$ NN (inadmissible)")
    a.axvspan(xi.min(), XISTAR[calib], color="#fdded7", alpha=0.5, zorder=0)
    a.axvline(XISTAR[calib], ls="--", color="#d73027", lw=1.8, alpha=0.7)
    a.set_xscale("log"); a.invert_xaxis()
    a.set_title(tag, fontsize=20, pad=10)
    a.set_ylim(-0.02, 1.05)
    if c == 0:
        a.set_ylabel("Cumulative worst-case\ndamage-jump prob. by 60 yr")
    a.grid(alpha=0.25)
    a.legend(loc="center left", framealpha=0.93, fontsize=14)

    b = ax[1][c]
    b.axhline(1.0, ls="-", color="k", lw=1.6)
    b.plot(xi, abar, "-", color="0.6", lw=1.6, alpha=0.6, zorder=1)
    b.scatter(xi[rel], abar[rel], s=150, color="#2c7fb8", marker="s", zorder=5, label=r"$\overline{g}\geq1$ (admissible)")
    b.scatter(xi[~rel], abar[~rel], s=150, facecolors="none", edgecolors="#2c7fb8",
              linewidths=2.6, marker="s", zorder=5, label=r"$\overline{g}<1$ (inadmissible)")
    b.axvspan(xi.min(), XISTAR[calib], color="#fdded7", alpha=0.5, zorder=0)
    b.axvline(XISTAR[calib], ls="--", color="#d73027", lw=1.8, alpha=0.7)
    b.text(XISTAR[calib], 3.2, f"$\\xi^*\\!\\approx\\!{XISTAR[calib]:g}$", color="#d73027",
           ha="center", va="bottom", fontsize=16,
           bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.85))
    b.set_xscale("log"); b.invert_xaxis(); b.set_yscale("log")
    b.set_ylim(top=8.0)
    b.set_xlabel(r"$\xi$  (smaller $\rightarrow$ more robustness aversion)")
    if c == 0:
        b.set_ylabel(r"$\overline{g}=\mathrm{mean}_\ell\,g^\ell$" + "\n(region-avg, log scale)")
    b.grid(alpha=0.25, which="both")
    b.legend(loc="lower left", framealpha=0.93, fontsize=14)
fig.suptitle("Admissibility of the worst-case damage-jump density:  $P_{\\rm undistorted}\\leq P_{\\rm wc}\\leq P_{\\rm floor}$, equivalently $\\overline{g}\\geq1$\n"
             "Open markers / shaded band = NN inadmissible ($P_{\\rm wc}<P_{\\rm undistorted}$, $\\overline{g}<1$): the raw NN worst case is PROVABLY wrong there",
             fontsize=17, y=1.005)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(os.path.join(OUT, "final_fig2_admissibility.png"))
print("saved final_fig2_admissibility.png")
plt.close(fig)

# ============================================================================
# FIGURE 3 -- worst-case TECH-jump cumulative probability
# ============================================================================
fig, ax = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
for c, (base, ctitle, calib) in enumerate([
        (FULL, "Full adjustment cost", "FULL"),
        (HALF, "Half adjustment cost", "HALF")]):
    a = ax[c]
    for (lab, plab, xi), col in zip(XIS, COLORS):
        cum = load(base, lab, "tech_jump_prob")
        if cum is None:
            continue
        t = load(base, lab, "t")
        # FULL tech is monotone-down and reliable at ALL xi (g_post<=1 everywhere); HALF tech
        # reverses below xi* (worst-case g_post>1 at high-Y states) -> unreliable there.
        ok = True if calib == "FULL" else reliable(xi, calib)
        a.plot(t, cum, "-" if ok else "--", color=col, label=plab,
               alpha=1.0 if ok else 0.5, zorder=3 if ok else 2)
    a.set_title(ctitle, fontsize=19, pad=10)
    a.set_xlabel("Year")
    if c == 0:
        a.set_ylabel("Cumulative worst-case\ntechnology-jump probability")
    a.set_ylim(-0.02, 1.02); a.set_xlim(0, 60)
    a.grid(alpha=0.25)
    a.legend(loc="center right", framealpha=0.93)
    note = ("reliable: all $\\xi$\n(worst case down-weights\nthe good tech jump,\n$g\\leq1$, monotone)" if calib == "FULL"
            else f"reliable: $\\xi\\geq{XISTAR[calib]:g}$\n(below, $g>1$ at high $Y$:\nCDF reverses upward)")
    a.text(0.02, 0.97, note, transform=a.transAxes, ha="left", va="top", fontsize=14, color="0.25",
           bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
fig.suptitle("Worst-case (distorted) technology-jump cumulative probability — one-jump ($\\pi=1$) climate model\n"
             "Full: reliable at all $\\xi$ (monotonically down — the worst case fears the good tech jump less);  "
             "Half: reliable only $\\xi\\geq0.02$ (reverses below)",
             fontsize=16, y=1.01)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "final_fig3_tech_cdf.png"))
print("saved final_fig3_tech_cdf.png")
plt.close(fig)

print("\nALL FINAL FIGURES SAVED to", OUT)
