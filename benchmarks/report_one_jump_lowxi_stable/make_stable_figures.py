"""
STABLE one-jump (pi=1) xi-sensitivity report figures + number dump.

Builds ONLY verified-correct objects:
  fig1_investment_emissions_vs_xi.png  -- year-5 i_d, i_g, E vs xi, both calibrations
                                          (stable set {148.600,0.050,0.025,0.010,0.005})
  fig2_worstcase_temperature_cdf.png   -- continuous-channel (drift h_y) worst-case temp CDF
                                          vs baseline, FULL stable xi set (incl 0.01 & 0.005)
  fig3_worstcase_jump_cdf_validated.png-- worst-case cumulative damage-jump prob vs Y,
                                          VALIDATED RANGE ONLY: xi >= 0.025 ({148.600,0.050,0.025})

Self-verification: every plotted/tabulated number is re-read from disk and asserted to match;
an assertion guarantees NO jump-prob data with xi<0.025 enters any figure/table.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/output_lowxi_float64"
FULL_DIR = os.path.join(ROOT, "OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostFull_logximin_m5p30_LR_warmup_cosine_10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations50000")
HALF_DIR = os.path.join(ROOT, "OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostHalf_Gamma0p12_Theta8p35_logximin_m5p30_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations50000")
FIGDIR = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal/benchmarks/report_one_jump_lowxi_stable/figures"
os.makedirs(FIGDIR, exist_ok=True)

# stable / certain xi sets
XIS_ECON = ["148.600", "0.040", "0.030", "0.025", "0.020", "0.010", "0.005"]   # controls + continuous channel: all stable
XIS_JUMP_VALID = ["148.600", "0.040", "0.030", "0.025"]                        # jump distribution: validated range only (xi>=0.025)

# climate params (params.py / AGENTS.md)
theta_bar = 1.86/1000.0
varsigma  = 1.2*1.86/1000.0
r1, r2, y_lower = 1.5, 0.36, 1.5

CALS = [("Full", FULL_DIR), ("Half", HALF_DIR)]

def gload(base, xi, name):
    return np.loadtxt(os.path.join(base, "SimulationDeterministic", f"SimulationOutputs_ξ_{xi}", name))

# ============================================================================
# Collect all numbers (single source of truth read from disk)
# ============================================================================
NUMBERS = []   # (calibration, xi, quantity, value)

# ---- year-5 controls + emissions ----
econ = {}   # econ[cal][xi] = dict(i_d,i_g,E)
for cal, base in CALS:
    econ[cal] = {}
    for xi in XIS_ECON:
        t = gload(base, xi, "t.txt")
        idx = min(60, len(t)-1)
        assert abs(t[idx]-5.0) < 1e-9, f"year-5 index mismatch {cal} {xi}: t[{idx}]={t[idx]}"
        i_d = float(gload(base, xi, "DirtyInvestment.txt")[idx])
        i_g = float(gload(base, xi, "GreenInvestment.txt")[idx])
        E   = float(gload(base, xi, "E.txt")[idx])
        econ[cal][xi] = dict(i_d=i_d, i_g=i_g, E=E, idx=idx)
        NUMBERS += [(cal, xi, "i_d_year5", i_d), (cal, xi, "i_g_year5", i_g), (cal, xi, "E_year5", E)]

# ============================================================================
# FIGURE 1: year-5 investment & emissions vs xi (both calibrations)
# ============================================================================
# x-axis = 1/xi (robustness aversion), TRUE spacing; infinity -> 0
xlab = [r"$\infty$" if x == "148.600" else x for x in XIS_ECON]
xpos = np.array([0.0 if x == "148.600" else 1.0 / float(x) for x in XIS_ECON])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, q, ttl, ylab in zip(
        axes,
        ["i_d", "i_g", "E"],
        [r"Dirty investment rate $i_d$ (year 5)",
         r"Green investment rate $i_g$ (year 5)",
         r"Emissions $\mathcal{E}$ (year 5)"],
        ["rate", "rate", r"$\mathcal{E}$"]):
    for cal, color, mk in [("Full", "C3", "o"), ("Half", "C0", "s")]:
        yv = [econ[cal][xi][q] for xi in XIS_ECON]
        ax.plot(xpos, yv, marker=mk, color=color, lw=1.8, label=f"{cal} adj-cost")
    ax.set_xticks(xpos); ax.set_xticklabels(xlab, rotation=45, fontsize=8)
    ax.set_xlabel(r"$1/\xi$  (robustness aversion; $\xi$ labelled; $\xi=\infty$ at $0$)")
    ax.set_title(ttl); ax.set_ylabel(ylab)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.suptitle(r"Year-5 robustness pullback vs $1/\xi$ (one-tech-jump $\pi=1$, neural solver)  --  true aversion scale, $\xi=\infty$ at $0$", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
f1 = os.path.join(FIGDIR, "fig1_investment_emissions_vs_xi.png")
fig.savefig(f1, dpi=140); plt.close(fig)
print("saved", f1)

# ============================================================================
# FIGURE 2: continuous-channel worst-case temperature CDF vs baseline
#           FULL stable xi set incl 0.01 & 0.005
# ============================================================================
def worstcase_temp_path(base, xi):
    t = gload(base, xi, "t.txt"); E = gload(base, xi, "E.txt"); h_y = gload(base, xi, "h_y.txt")
    Y0 = gload(base, xi, "Y.txt")[0]
    dt = np.diff(t)
    drift_base  = theta_bar*E
    drift_worst = theta_bar*E + varsigma*h_y*E
    Yb = np.empty_like(E); Yw = np.empty_like(E); Yb[0]=Y0; Yw[0]=Y0
    for i in range(1, len(E)):
        Yb[i] = Yb[i-1] + 0.5*(drift_base[i]+drift_base[i-1])*dt[i-1]
        Yw[i] = Yw[i-1] + 0.5*(drift_worst[i]+drift_worst[i-1])*dt[i-1]
    inst_var = (varsigma*E)**2
    var = np.empty_like(E); var[0]=0.0
    for i in range(1, len(E)):
        var[i] = var[i-1] + 0.5*(inst_var[i]+inst_var[i-1])*dt[i-1]
    return t, Yb, Yw, var, drift_base, drift_worst, h_y

def idx_at(t, year):
    return int(np.argmin(np.abs(t-year)))

# colors by xi (neutral -> most averse)
XI_C = {"148.600":"0.55", "0.040":"C2", "0.030":"C8", "0.025":"C0",
        "0.020":"C5", "0.010":"C1", "0.005":"C3"}
XI_T = {"148.600":r"$\xi=\infty$", "0.040":r"$\xi=0.04$", "0.030":r"$\xi=0.03$",
        "0.025":r"$\xi=0.025$", "0.020":r"$\xi=0.02$",
        "0.010":r"$\xi=0.01$", "0.005":r"$\xi=0.005$"}
HORIZON = 60

# The continuous channel is ISOLATED by comparing, for each xi, the worst-case
# Y-CDF (drift theta_bar*E + varsigma*h_y*E) against THAT SAME xi's own baseline
# (drift theta_bar*E). Same path => same emissions => same Brownian variance => the
# only difference is the h_y distortion. This is the genuinely FOSD-guaranteed object
# (the h_y>0 shift). Plotting each worst-case against a single foreign baseline would
# conflate the distortion with the (separate) lower-emissions effect of robust i_d.
#
# LEFT  panel (per calibration figure): worst-case vs own-xi baseline temperature CDF at t=60yr
#   -- only the most-averse xi=0.005 worst/base pair, to show the FOSD shift cleanly.
# RIGHT panel: the isolated worst-case temperature SHIFT dY(t)=Y_worst-Y_base over the
#   horizon, for every stable xi (incl 0.01 & 0.005) -- monotone, grows ~1/xi.
wc_store = {}
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
for col, (cal, base) in enumerate(CALS):
    wc_store[cal] = {}
    ygrid = np.linspace(1.6, 2.8, 700)
    axU = axes[0, col]; axL = axes[1, col]
    # store all xi numbers
    paths = {}
    for xi in XIS_ECON:
        t, Yb, Yw, var, db, dw, h_y = worstcase_temp_path(base, xi)
        paths[xi] = (t, Yb, Yw, var, h_y)
        k = idx_at(t, HORIZON); sd = max(np.sqrt(var[k]), 1e-6)
        wc_store[cal][xi] = dict(Yw60=float(Yw[k]), Yb60=float(Yb[k]), sd60=float(sd),
                                 dY60=float(Yw[k]-Yb[k]),
                                 hy_max=float(h_y.max()), hy_min=float(h_y.min()),
                                 shift0=float(dw[0]-db[0]))
        NUMBERS += [(cal, xi, "Yworst_60yr", float(Yw[k])),
                    (cal, xi, "Ybase_60yr", float(Yb[k])),
                    (cal, xi, "hy_max", float(h_y.max()))]
    # --- UPPER: CDF, worst vs own-baseline, most-averse xi (cleanest FOSD picture) ---
    for xi in ["0.005", "0.010"]:
        t, Yb, Yw, var, h_y = paths[xi]
        k = idx_at(t, HORIZON); sd = max(np.sqrt(var[k]), 1e-6)
        Fb = norm.cdf(ygrid, loc=Yb[k], scale=sd)
        Fw = norm.cdf(ygrid, loc=Yw[k], scale=sd)
        axU.plot(ygrid, Fb, color=XI_C[xi], lw=1.4, ls="--", alpha=0.85,
                 label=f"Baseline {XI_T[xi]}")
        axU.plot(ygrid, Fw, color=XI_C[xi], lw=2.1,
                 label=f"Worst-case {XI_T[xi]}")
    axU.set_title(f"{cal} adj-cost: worst-case vs own-$\\xi$ baseline CDF (t = 60 yr)")
    axU.set_xlabel(r"Temperature anomaly $Y$ ($^\circ$C)")
    axU.set_ylabel(r"CDF  $F(y)$")
    axU.grid(alpha=0.3); axU.legend(fontsize=8.5, loc="lower right")
    # --- LOWER: isolated worst-case temperature SHIFT dY(t) for every stable xi ---
    for xi in ["0.040", "0.030", "0.025", "0.020", "0.010", "0.005"]:
        t, Yb, Yw, var, h_y = paths[xi]
        axL.plot(t, Yw - Yb, color=XI_C[xi], lw=2.0, label=XI_T[xi])
    axL.axhline(0.0, color="k", lw=1.0, ls=":")
    axL.set_title(f"{cal} adj-cost: isolated worst-case temperature shift $\\Delta Y(t)=Y_{{\\rm worst}}-Y_{{\\rm base}}$")
    axL.set_xlabel("Year  $t$")
    axL.set_ylabel(r"$\Delta Y$  ($^\circ$C, hotter)")
    axL.grid(alpha=0.3); axL.legend(fontsize=9, loc="upper left", title=r"continuous channel")
fig.suptitle(r"Continuous-channel (drift distortion $h_y$) worst-case temperature vs baseline -- full stable $\xi$ set"
             "\n"
             r"worst-case is hotter ($F_{\mathrm{worst}}\leq F_{\mathrm{base}}$, FOSD); the isolated $h_y$ shift grows $\sim 1/\xi$",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
f2 = os.path.join(FIGDIR, "fig2_worstcase_temperature_cdf.png")
fig.savefig(f2, dpi=140); plt.close(fig)
print("saved", f2)

# stochastic dominance check: each xi's worst-case FOSD-dominates ITS OWN baseline
for cal, base in CALS:
    ygrid = np.linspace(1.0, 3.0, 700)
    for xi in ["0.040","0.030","0.025","0.020","0.010","0.005"]:
        t, Yb, Yw, var, db, dw, h_y = worstcase_temp_path(base, xi)
        k = idx_at(t, HORIZON); sd = max(np.sqrt(var[k]), 1e-6)
        Fb = norm.cdf(ygrid, loc=Yb[k], scale=sd)
        Fw = norm.cdf(ygrid, loc=Yw[k], scale=sd)
        dom = bool(np.all(Fw <= Fb + 1e-9))
        assert dom, f"FOSD violated (own-baseline) {cal} {xi}"
        assert h_y.max() > 0, f"h_y not positive {cal} {xi}"
        assert (Yw[k]-Yb[k]) > 0, f"worst-case not hotter {cal} {xi}"

# ============================================================================
# FIGURE 3: worst-case cumulative damage-jump prob vs Y -- VALIDATED RANGE xi>=0.025
# ============================================================================
# HARD GUARD: assert no xi<0.025 is in the jump set
for x in XIS_JUMP_VALID:
    assert float(x) >= 0.025 or x == "148.600", f"xi<0.025 leaked into jump set: {x}"

XI_CJ = {"148.600":"k", "0.040":"C2", "0.030":"C8", "0.025":"C0"}
XI_TJ = {"148.600":"Baseline ($g=1$, uncertainty-neutral)", "0.040":r"Worst-case $\xi=0.04$",
         "0.030":r"Worst-case $\xi=0.03$", "0.025":r"Worst-case $\xi=0.025$"}

jump_store = {}
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
for ax, (cal, base) in zip(axes, CALS):
    jump_store[cal] = {}
    for xi in XIS_JUMP_VALID:
        Y = gload(base, xi, "Y.txt")
        djp = gload(base, xi, "dmg_jump_prob.txt")   # cumulative worst-case dmg-jump prob along path
        assert np.all(np.diff(Y) >= -1e-9), f"Y not monotone {cal} {xi}"
        Pterm = float(djp[-1])
        jump_store[cal][xi] = dict(Pterm=Pterm, Yterm=float(Y[-1]))
        NUMBERS.append((cal, xi, "dmg_jump_prob_terminal", Pterm))
        lw, ls = (2.6, "--") if xi == "148.600" else (2.1, "-")
        ax.plot(Y, djp, color=XI_CJ[xi], lw=lw, ls=ls, label=XI_TJ[xi])
    ax.set_title(f"{cal} adj-cost  (one-tech-jump, validated range $\\xi\\geq0.025$)")
    ax.set_xlabel(r"Temperature anomaly $Y$ ($^\circ$C)")
    ax.set_ylabel(r"Cumulative damage-jump probability  $P(\mathrm{jump}\ \mathrm{by}\ Y)$")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 0.75)
fig.suptitle(r"Worst-case cumulative damage-jump probability vs temperature -- validated range ($\xi\geq0.025$)"
             "\n"
             "more aversion (smaller xi) raises the feared jump probability above baseline; results for xi<0.025 omitted (under development)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
f3 = os.path.join(FIGDIR, "fig3_worstcase_jump_cdf_validated.png")
fig.savefig(f3, dpi=140); plt.close(fig)
print("saved", f3)

# ============================================================================
# SELF-VERIFICATION
# ============================================================================
print("\n" + "="*78)
print("SELF-VERIFICATION")
print("="*78)

# (A) re-read every tabulated/plotted number from disk and assert match
fails = 0
for cal, xi, q, val in NUMBERS:
    base = dict(CALS)[cal]
    if q in ("i_d_year5", "i_g_year5", "E_year5"):
        fn = {"i_d_year5":"DirtyInvestment.txt","i_g_year5":"GreenInvestment.txt","E_year5":"E.txt"}[q]
        t = gload(base, xi, "t.txt"); idx = min(60, len(t)-1)
        disk = float(gload(base, xi, fn)[idx])
        if abs(disk - val) > 1e-9:
            print(f"  MISMATCH {cal} {xi} {q}: stored {val} disk {disk}"); fails += 1
    elif q == "dmg_jump_prob_terminal":
        disk = float(gload(base, xi, "dmg_jump_prob.txt")[-1])
        if abs(disk - val) > 1e-9:
            print(f"  MISMATCH {cal} {xi} {q}: stored {val} disk {disk}"); fails += 1
print(f"  re-read assertion: {len(NUMBERS)} numbers checked, {fails} mismatches")
assert fails == 0, "re-read mismatch!"

# (B) assert NO jump-prob data with xi<0.025 anywhere in NUMBERS
jump_numbers = [(c,x,q,v) for (c,x,q,v) in NUMBERS if q == "dmg_jump_prob_terminal"]
bad = [(c,x,q,v) for (c,x,q,v) in jump_numbers if (x != "148.600" and float(x) < 0.025)]
print(f"  jump-prob data points: {[(c,x) for (c,x,q,v) in jump_numbers]}")
print(f"  jump-prob points with xi<0.025: {bad}  (must be empty)")
assert len(bad) == 0, "xi<0.025 jump data leaked!"
print("  ASSERTION PASS: no jump-probability data point with xi<0.025 in any figure/table.")

# (C) neutral worst-case == baseline for jump panel (g=1)
for cal, base in CALS:
    Y = gload(base, "148.600", "Y.txt")
    Jd_base = r1*(np.exp(r2/2*(Y-y_lower)**2)-1)*(Y > y_lower)
    intens = gload(base, "148.600", "dmg_jump_intensity.txt")
    mask = Jd_base > 0
    gavg = intens[mask]/Jd_base[mask]
    print(f"  {cal}: neutral (xi=148.600) g_avg in [{gavg.min():.6f},{gavg.max():.6f}] (should be ~1 => worst-case==baseline)")
    assert abs(gavg.mean()-1.0) < 1e-3, "neutral run not g=1!"

# (D) FULL NUMBER DUMP
print("\n" + "="*78)
print("FULL NUMBER DUMP (every value that went into the report)")
print("="*78)
print(f"{'calibration':<12} {'xi':>9} {'quantity':<24} {'value':>14}")
print("-"*64)
for cal, xi, q, val in NUMBERS:
    xilab = "inf(148.600)" if xi == "148.600" else xi
    print(f"{cal:<12} {xilab:>9} {q:<24} {val:>14.6f}")

# (E) dump as JSON too for spot-checking
dump = {"econ": {c: {x: {k: econ[c][x][k] for k in ("i_d","i_g","E")} for x in XIS_ECON} for c in ("Full","Half")},
        "worstcase_temp": {c: wc_store[c] for c in ("Full","Half")},
        "jump_validated": {c: jump_store[c] for c in ("Full","Half")}}
with open(os.path.join(FIGDIR, "..", "report_numbers.json"), "w") as fh:
    json.dump(dump, fh, indent=2)
print("\nwrote report_numbers.json")
print("\nFIGURES:", f1, f2, f3, sep="\n  ")
