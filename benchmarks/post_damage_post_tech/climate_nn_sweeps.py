"""
Climate analysis from the validated post-damage-post-tech DGM network: how the optimal
response to temperature Y changes with (a) the robustness multiplier xi and (b) the damage
curvature lambda3. Uses the FOC-consistent large-batch checkpoint (validated: RMS FOC ~6e-5).

Two figures (benchmark format): controls i_d,i_g and the economic marginal value of
temperature V_Y = v_Y-(logN)_Y, plotted vs Y (top) and vs Z (bottom), with curves indexed
by xi (robustness sweep) and by lambda3 (damage sweep).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_pretrained_climate as C   # build_and_load, evaluate, fixed logK/lam3/logxi, OD

OD = C.OD
y_up = C.y_up


def _panels(net, sweep, fixed_label, title, fname):
    v_nn, i_d_nn, i_g_nn = net
    Zg = np.linspace(0.1, 0.9, 41)
    Yg = np.linspace(0.0, 4.0, 41)
    Yfix, Zfix = 2.5, 0.5
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    cols = plt.cm.viridis(np.linspace(0.12, 0.85, len(sweep)))
    for col, (lab, kw) in zip(cols, sweep):
        oY = C.evaluate(v_nn, i_d_nn, i_g_nn, np.full_like(Yg, Zfix), Yg, **kw)
        oZ = C.evaluate(v_nn, i_d_nn, i_g_nn, Zg, np.full_like(Zg, Yfix), **kw)
        ax[0, 0].plot(Yg, oY["i_d"], color=col, lw=1.9, label=lab)
        ax[0, 1].plot(Yg, oY["i_g"], color=col, lw=1.9, label=lab)
        ax[0, 2].plot(Yg, oY["V_Y"], color=col, lw=1.9, label=lab)
        ax[1, 0].plot(Zg, oZ["i_d"], color=col, lw=1.9, label=lab)
        ax[1, 1].plot(Zg, oZ["i_g"], color=col, lw=1.9, label=lab)
        ax[1, 2].plot(Zg, oZ["q_d"], color=col, lw=1.9, label=lab)
    for a in ax[0]:
        a.axvline(y_up, color="grey", ls=":", lw=1.0); a.set_xlabel("Y (temperature)")
        a.grid(alpha=0.3); a.legend(fontsize=8)
    for a in ax[1]:
        a.set_xlabel("Z (green capital share)"); a.grid(alpha=0.3); a.legend(fontsize=8)
    ax[0, 0].set_title(fr"$i^d$ vs temperature  ($Z={Zfix}$)"); ax[0, 0].set_ylabel(r"$i^d$")
    ax[0, 1].set_title(fr"$i^g$ vs temperature  ($Z={Zfix}$)"); ax[0, 1].set_ylabel(r"$i^g$")
    ax[0, 2].set_title(fr"$V_Y$ vs temperature  ($Z={Zfix}$)"); ax[0, 2].set_ylabel(r"$V_Y$")
    ax[0, 2].axhline(0, color="r", ls="--", lw=0.8)
    ax[1, 0].set_title(fr"$i^d$ vs share  ($Y={Yfix}$)"); ax[1, 0].set_ylabel(r"$i^d$")
    ax[1, 1].set_title(fr"$i^g$ vs share  ($Y={Yfix}$)"); ax[1, 1].set_ylabel(r"$i^g$")
    ax[1, 2].set_title(fr"$\tilde q_d$ vs share  ($Y={Yfix}$)"); ax[1, 2].set_ylabel(r"$\tilde q_d$")
    fig.suptitle(f"{title}  [{fixed_label}, logK={C.logK_fix:.2f}]", fontsize=13)
    fig.tight_layout()
    p = os.path.join(OD, fname); fig.savefig(p, dpi=150); print("saved", p, flush=True)


def main():
    net = C.build_and_load()
    print("[loaded validated checkpoint]", flush=True)

    # (1) robustness sweep: xi in {148.4 ~ none, 0.1, 0.05 strong}, lambda3 fixed
    xis = [(148.4, 5.0), (0.1, np.log(0.1)), (0.05, np.log(0.05))]
    sweep_xi = [(fr"$\xi$={xi:g}", dict(logxi=lx, lam3=1/6.0)) for xi, lx in xis]
    _panels(net, sweep_xi, r"$\lambda_3$=0.167",
            "Robustness sweep: response to temperature vs uncertainty aversion $\\xi$",
            "climate_robustness_sweep.png")

    # (2) damage sweep: lambda3 in {0, 1/6, 1/3}, xi fixed ~ no robustness
    l3s = [0.0, 1/6.0, 1/3.0]
    sweep_l3 = [(fr"$\lambda_3$={l3:.3f}", dict(lam3=l3, logxi=5.0)) for l3 in l3s]
    _panels(net, sweep_l3, r"$\xi$=148.4",
            "Damage sweep: response to temperature vs damage curvature $\\lambda_3$",
            "climate_damage_sweep.png")

    # numeric readouts at Z=0.5, Y=3
    for xi, lx in xis:
        o = C.evaluate(*net, np.array([0.5]), np.array([3.0]), logxi=lx, lam3=1/6.0)
        print(f"  xi={xi}: i_d={o['i_d'][0]:+.4f} i_g={o['i_g'][0]:+.4f} V_Y={o['V_Y'][0]:+.4f}", flush=True)


if __name__ == "__main__":
    main()
