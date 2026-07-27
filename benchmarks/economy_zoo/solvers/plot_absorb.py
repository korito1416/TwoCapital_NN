"""Figures for the ABSORB economy (design_absorb.json plots spec).

  figures/absorb_catchup.png   - A_g(s), F(s)+F'(s), i_g(s), i_r(s) on s in [-7,3],
                                 s0 and the A_g = A_d crossing and the stationary
                                 point s* (mu_s = 0) marked.
  figures/absorb_climate.png   - w_pre(Y) and the five closed-form w^l(Y), jump
                                 window [1.5, 2.5] shaded.
  figures/absorb_lifted_ir.png - FOC-consistent lifted fields consumed by the
                                 anchor fit: log10 i_r and i_d over (logR, Z) at
                                 logK = 6.78 (PreDamagePreTech).

Run: python plot_absorb.py   (after absorb.py)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
D = np.load(os.path.join(HERE, "..", "outputs", "absorb.npz"))
sys.path.insert(0, os.path.join(HERE, "..", "maps"))
import absorb as amap  # noqa: E402  (the map module, for the lifted fields)

s, F, Fp = D["s_grid"], D["F"], D["Fp"]
ig_s, ir_s, Ag_s, mu_s = D["i_g_s"], D["i_r_s"], D["A_g_s"], D["mu_s"]
Yg, w_pre = D["Y_grid"], D["w_pre"]
lam3_levels = D["lam3_levels"]
a_l, b_l, c_l = D["w_post_a"], D["w_post_b"], D["w_post_c"]
s0 = float(np.log(11.2 / 616.0))
A_d = 0.1303

GRID = dict(color="0.85", lw=0.6)
ACC = "#2166ac"      # primary series hue
ACC2 = "#b2182b"     # secondary (distinct) hue

# --------------------------------------------------------------- 1: catch-up
m = (s >= -7) & (s <= 3)
i_star = np.argmin(np.abs(mu_s))                     # stationary knowledge intensity
s_star = s[i_star]
cross = s[np.argmin(np.abs(Ag_s - A_d))]
fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
panels = [
    (ax[0, 0], Ag_s[m], r"$A_g(s)$ green productivity ladder", None),
    (ax[0, 1], F[m], r"$F(s)$ knowledge-block value", None),
    (ax[1, 0], ig_s[m], r"$i_g(s)$ green investment (economy units)", None),
    (ax[1, 1], ir_s[m], r"$i_r(s)$ R&D (economy units, $I_r/K_g$)", None),
]
for a, y, title, _ in panels:
    a.plot(s[m], y, color=ACC, lw=2)
    a.axvline(s0, color="0.4", lw=1, ls="--")
    a.axvline(s_star, color=ACC2, lw=1, ls=":")
    a.grid(**GRID)
    a.set_title(title, fontsize=10)
    a.set_xlabel(r"$s = \log R - \log K_g$")
ax[0, 0].axhline(A_d, color="0.4", lw=1)
ax[0, 0].axvline(cross, color="0.6", lw=1, ls="-.")
ax[0, 0].annotate(f"$A_g=A_d$ at s={cross:.2f}", (cross, A_d), xytext=(cross + 0.3, A_d - 0.006), fontsize=8)
ax[0, 0].annotate(f"$s_0$={s0:.2f}", (s0, Ag_s[m][0]), xytext=(s0 + 0.15, 0.112), fontsize=8)
ax[0, 0].annotate(f"$s^*$={s_star:.2f} (drift=0)", (s_star, 0.135), fontsize=8, color=ACC2)
axFp = ax[0, 1].twinx()  # noqa -- second panel content, same x
# avoid dual axis: plot F' in its own inset instead
axFp.remove()
ins = ax[0, 1].inset_axes([0.52, 0.14, 0.44, 0.38])
ins.plot(s[m], Fp[m], color=ACC2, lw=1.5)
ins.axhline(0.66, color="0.4", lw=0.8, ls="--")
ins.set_title(r"$F'(s)$ (gate 0.66 dashed)", fontsize=7)
ins.tick_params(labelsize=6)
ins.grid(**GRID)
fig.suptitle("ABSORB catch-up diagram: ladder, value, policies "
             f"(s0={s0:.3f} dashed, stationary s*={s_star:.2f} dotted)", fontsize=11)
fig.savefig(os.path.join(FIG, "absorb_catchup.png"), dpi=150)
plt.close(fig)

# --------------------------------------------------------------- 2: climate
fig, a = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
mY = Yg <= 4.0
blues = plt.cm.Blues(np.linspace(0.35, 0.95, 5))     # sequential: lambda3 magnitude
for k in range(5):
    wl = -(a_l[k] * Yg**2 + b_l[k] * Yg + c_l[k])
    a.plot(Yg[mY], wl[mY], color=blues[k], lw=1.8)
    a.annotate(rf"$\ell$={k+1}", (4.02, wl[mY][-1]), fontsize=8, color=blues[k],
               va="center", annotation_clip=False)
a.plot(Yg[mY], w_pre[mY], color=ACC2, lw=2.4)
a.annotate("$w_{pre}$ (with jump anticipation)", (2.0, np.interp(2.0, Yg, w_pre) + 0.06),
           fontsize=9, color=ACC2)
a.axvspan(1.5, 2.5, color="0.92", zorder=0)
a.annotate("jump window", (1.55, a.get_ylim()[0] * 0.05), fontsize=8, color="0.4")
a.grid(**GRID)
a.set_xlabel("temperature anomaly $Y$")
a.set_ylabel("climate account $w(Y)$")
a.set_title("ABSORB climate accounts: pre-damage BVP vs post-damage quadratics\n"
            r"$w^\ell(Y)$, $\ell$=1..5 with $\lambda_3$ = 0 .. 1/3", fontsize=10)
fig.savefig(os.path.join(FIG, "absorb_climate.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------ 3: lifted fields (map view)
lr_g = np.linspace(1, 6, 121)
Z_g = np.linspace(0.02, 0.98, 121)
LR, ZZ = np.meshgrid(lr_g, Z_g)
lk0 = 6.78
n = LR.size
FLD = amap.fields("PreDamagePreTech",
                  np.full(n, lk0), ZZ.ravel(), np.full(n, 1.1),
                  LR.ravel(), np.zeros(n), np.zeros(n))
IR = FLD["i_r"].reshape(LR.shape)
IDf = FLD["i_d"].reshape(LR.shape)
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
p0 = ax[0].pcolormesh(lr_g, Z_g, np.log10(np.maximum(IR, 1e-12)), cmap="viridis",
                      shading="auto")
fig.colorbar(p0, ax=ax[0], label=r"$\log_{10}\, i_r$  (production rate $I_r/K$)")
ax[0].set_title(r"lifted FOC-consistent $i_r$ at $\log K=6.78$", fontsize=10)
p1 = ax[1].pcolormesh(lr_g, Z_g, IDf, cmap="magma", shading="auto")
fig.colorbar(p1, ax=ax[1], label=r"$i_d$")
ax[1].set_title(r"lifted FOC-consistent $i_d$ at $\log K=6.78$", fontsize=10)
for a in ax:
    a.set_xlabel(r"$\log R$")
    a.set_ylabel(r"$Z$")
    a.plot([np.log(11.2)], [0.7], marker="*", ms=12, color="w", mec="k")
fig.suptitle("ABSORB lifted policy fields consumed by the anchor fit "
             "(star = production x0)", fontsize=11)
fig.savefig(os.path.join(FIG, "absorb_lifted_ir.png"), dpi=150)
plt.close(fig)
print("wrote figures:", sorted(os.listdir(FIG)))
