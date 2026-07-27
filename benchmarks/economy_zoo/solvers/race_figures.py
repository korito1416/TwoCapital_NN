"""Figures for RACE (design_race.json plots field + task-required hazard curve).

Reads outputs/race.npz (+ exact evaluation via race.py where a dense curve is
cleaner). Writes benchmarks/economy_zoo/figures/race_*.png:
  race_robustness_chokes_rd.png  Delta*(xi), P(xi) | i_r*(xi) with E[time-to-
                                 breakthrough] annotations + xibar | i_g/i_d policies
                                 (three stacked panels — no dual axes)
  race_value_slices.png          v vs Y at (logK,Z)=(6.1,0.7), pre/post tech per
                                 damage regime (pre-damage + l=3, l=5), xi=0.05,
                                 with the level-damage slope (-0.008) dashed reference
  race_ir_net_heatmap.png        lifted i_r-net = Z * i_r*(xi) over (Z, logxi),
                                 PreDamagePreTech, corner region shaded, production
                                 lx box marked
  race_hazard_curve.png          Lambda(i_r) = rho_b log(1+theta_r i_r) with the
                                 solved (i_r*, Lambda*) at xi = 0.05 / 0.1 / neutral

Design deviations (documented): the ABSORB w(Y) overlay in the value-slice plot is
replaced by a dashed level-damage-slope reference (ABSORB not yet built at RACE
build time); the reference-run i_r contour overlay on the heatmap is omitted
(needs production checkpoints — out of scope for the analytic build).
"""
import os, json, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("race_solver", os.path.join(HERE, "race.py"))
race = importlib.util.module_from_spec(spec)
spec.loader.exec_module(race)
PAR = race.PAR
FIG = race.FIG_DIR
os.makedirs(FIG, exist_ok=True)

npz = np.load(os.path.join(race.OUT_DIR, "race.npz"))
lx, xi = npz["logxi_grid"], npz["xi_grid"]
xibar = float(npz["xibar"])

# Okabe-Ito categorical order (colorblind-safe, fixed assignment)
C_BLUE, C_ORANGE, C_GREEN, C_VERM, C_PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
GRAY, INK = "#999999", "#222222"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})

XT = [0.01, 0.05, 0.1, 0.5, 1, 10, 150]


def xi_axis(ax):
    ax.set_xticks(np.log(XT))
    ax.set_xticklabels([str(v) for v in XT])
    ax.set_xlim(lx[0], lx[-1])


# ------------------------------------------------------------------ fig 1: robustness chokes R&D
fig, axes = plt.subplots(3, 1, figsize=(6.4, 8.2), sharex=True)
a = axes[0]
a.plot(lx, npz["pre_Delta"], color=C_BLUE, lw=2, label=r"$\Delta^*(\xi)$ (tech-jump value gap)")
a.plot(lx, npz["pre_P"], color=C_ORANGE, lw=2,
       label=r"$P(\xi)=\xi(1-e^{-\Delta^*/\xi})$ (robust jump premium)")
a.axvline(np.log(xibar), color=GRAY, lw=1, ls=":")
a.set_ylabel("value units")
a.legend(frameon=False, loc="upper right")
a.set_title("RACE: robustness chokes R&D (pre-damage, $\\kappa=0.001$)", fontsize=10)

a = axes[1]
a.plot(lx, npz["pre_i_r"], color=C_GREEN, lw=2)
a.axvline(np.log(xibar), color=GRAY, lw=1, ls=":")
a.text(np.log(xibar) + 0.06, 0.0455, r"corner $\bar\xi=%.4f$" % xibar,
       color=INK, fontsize=9, ha="left", va="bottom", rotation=90)
a.fill_betweenx([0, 0.06], lx[0], np.log(xibar), color=GRAY, alpha=0.15, lw=0)
for x0, off, ha in [(0.05, (10, 4), "left"), (0.1, (8, -12), "left"),
                    (148.4, (-8, -16), "right")]:
    s = race.solve(np.array([x0]), np.array([PAR["kappa0"]]))
    irv, Lv = float(s["i_r"][0]), float(s["Lambda"][0])
    a.plot(np.log(x0), irv, "o", color=C_GREEN, ms=6, mec="white", mew=1)
    a.annotate(r"$\xi$=%s: $i_r^*$=%.4f, E[T]=%.0fy" % (("%g" % x0), irv, 1.0 / Lv),
               (np.log(x0), irv), textcoords="offset points", xytext=off, ha=ha,
               fontsize=8.5)
a.set_ylim(0, 0.06)
a.set_ylabel(r"$i_r^*$  ($I_r/K_g$)")

a = axes[2]
a.plot(lx, npz["pre_i_g"], color=C_BLUE, lw=2, label=r"$i_g^*(\xi)$ pre-tech")
a.axhline(float(npz["i_g_pp"]), color=C_ORANGE, lw=2, label=r"$i_g''$ post-tech = 0.1258")
a.axhline(float(npz["i_d_star"]), color=C_VERM, lw=2, ls="--", label=r"$i_d^*$ = 0.1031 (all regimes)")
a.axvline(np.log(xibar), color=GRAY, lw=1, ls=":")
a.set_ylabel("investment rate")
a.set_xlabel(r"$\xi$ (log scale); more uncertainty-averse $\rightarrow$ left")
a.legend(frameon=False, loc="center right", fontsize=9)
xi_axis(a)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "race_robustness_chokes_rd.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ fig 2: value slices v vs Y
XI0 = 0.05
lk0, Z0 = 6.1, 0.7
base = lk0 + PAR["beta_d"] * np.log(1 - Z0) + PAR["beta_g"] * np.log(Z0)
fig, a = plt.subplots(figsize=(6.4, 4.6))
cases = [("pre-damage (l=1)", PAR["kappa0"], C_BLUE, np.linspace(0, 4, 50)),
         ("post-damage l=3", float(race.kappa_of(1.0 / 6.0)), C_ORANGE, np.linspace(2.5, 4, 30)),
         ("post-damage l=5", float(race.kappa_of(1.0 / 3.0)), C_VERM, np.linspace(2.5, 4, 30))]
for lab, kapv, col, Yg in cases:
    s = race.solve(np.array([XI0]), np.array([kapv]))
    q1 = float(s["q1"][0])
    a.plot(Yg, base + q1 * Yg + float(s["V0pre"][0]), color=col, lw=2,
           label="%s, pre-tech ($q_1$=%.2f)" % (lab, q1))
    a.plot(Yg, base + q1 * Yg + float(s["V0post"][0]), color=col, lw=2, ls="--")
s = race.solve(np.array([XI0]), np.array([PAR["kappa0"]]))
v0 = base + float(s["V0pre"][0])
Yg = np.linspace(0, 4, 50)
a.plot(Yg, v0 - 0.008 * Yg, color=GRAY, lw=1.5, ls=":",
       label="level-damage slope $-0.008$ (ABSORB-class, ~12x flatter)")
a.annotate("dashed = post-tech (gap $\\Delta^*$)", (0.55, 0.93), xycoords="axes fraction",
           fontsize=9, color=INK)
a.set_xlabel("temperature anomaly Y")
a.set_ylabel("v (production states)")
a.set_title(r"RACE value slices at $(\log K, Z)=(6.1, 0.7)$, $\xi=0.05$: growth damages"
            "\n" r"give the steep common slope $q_1=-\kappa_l/\delta$")
a.legend(frameon=False, fontsize=8.5, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "race_value_slices.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ fig 3: lifted i_r-net heatmap
Zg = np.linspace(0.01, 0.99, 99)
LX, ZZ = np.meshgrid(lx, Zg, indexing="ij")
IR = np.maximum(np.interp(LX, lx, npz["pre_i_r"]), 0.0) * ZZ   # i_r-net = Z * i_r*(xi)
fig, a = plt.subplots(figsize=(6.8, 4.4))
pc = a.pcolormesh(LX, ZZ, IR, cmap="Blues", shading="auto")
cb = fig.colorbar(pc, ax=a)
cb.set_label(r"lifted $i_r$-net $= Z\, i_r^*(\xi)$   ($I_r/K$)")
a.axvline(np.log(xibar), color=C_VERM, lw=1.5)
a.fill_betweenx([0.01, 0.99], lx[0], np.log(xibar), color=GRAY, alpha=0.35, lw=0)
a.text(0.5 * (lx[0] + np.log(xibar)), 0.5, "corner\n$i_r^*=0$", ha="center", va="center",
       fontsize=9, color=INK)
for b in (np.log(0.05), np.log(148.6)):
    a.axvline(b, color=INK, lw=1, ls="--")
a.text(0.5 * (np.log(0.05) + np.log(148.6)), 0.05,
       "production $\\log\\xi$ box (between dashed lines)", fontsize=8.5, color=INK,
       ha="center", va="bottom")
a.set_xlabel(r"$\xi$ (log scale)")
a.set_ylabel("Z (green capital share)")
a.set_title("RACE lifted PreDamagePreTech $i_r$-net field (flat in $\\log R$ by design)")
xi_axis(a)
a.grid(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "race_ir_net_heatmap.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ fig 4: hazard curve
irg = np.linspace(0, 0.08, 200)
fig, a = plt.subplots(figsize=(6.0, 4.2))
a.plot(irg, race.Lam(irg), color=C_BLUE, lw=2,
       label=r"$\Lambda(i_r)=\rho_b\log(1+\theta_r i_r)$, $(\rho_b,\theta_r)=(0.40,16.7)$")
for x0, col in [(0.05, C_VERM), (0.1, C_ORANGE), (np.inf, C_GREEN)]:
    s = race.solve(np.array([x0]), np.array([PAR["kappa0"]]))
    irv, Lv = float(s["i_r"][0]), float(s["Lambda"][0])
    a.plot(irv, Lv, "o", color=col, ms=7, mec="white", mew=1)
    a.annotate(r"$\xi$=%s: (%.4f, %.3f), E[T]=%.1fy"
               % ("$\\infty$" if np.isinf(x0) else "%g" % x0, irv, Lv, 1.0 / Lv),
               (irv, Lv), textcoords="offset points", xytext=(10, -4), fontsize=8.5, color=col)
a.set_xlabel(r"R&D rate $i_r$  ($I_r/K_g$)")
a.set_ylabel(r"breakthrough hazard $\Lambda$ (1/yr)")
a.set_title("RACE patent-race hazard and the solved operating points")
a.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "race_hazard_curve.png"), dpi=150)
plt.close(fig)

print("wrote figures:", sorted(f for f in os.listdir(FIG) if f.startswith("race_")))
