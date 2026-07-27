"""By-year test of the damage-belief defect, in the report's usual simulated-path
format. PREDICT the temperature the economy reaches from the model (dY = E*thetabar
along the reference's own emission path), then confront it with the temperature the
reference's damage belief prices (Y_imp = yhat + sqrt(6*Delta), backed out from the
delivered belief). The belief prices a temperature the economy never reaches.
No text on the plot; caption carries the message."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(ROOT, "output_xiprofile_ref_20260714",
                 "SimulationDeterministic", "SimulationOutputs_ξ_0.050")
THBAR, YHAT, YBAR = 1.86e-3, 2.5, 4.0
FIG = os.path.join(os.path.dirname(__file__), "figures")

t = np.loadtxt(f"{D}/t.txt"); Y = np.loadtxt(f"{D}/Y.txt"); E = np.loadtxt(f"{D}/E.txt")
n = min(len(t), len(Y), len(E)); m = t[:n] <= 60.0
t, Y, E = t[:n][m], Y[:n][m], E[:n][m]

# PREDICTION: if the damage jump fired now, forward temperature the economy's own
# emissions produce, dY = E*thetabar, from Y=yhat.
Yfwd = YHAT + THBAR * np.concatenate([[0.0], np.cumsum(0.5 * (E[1:] + E[:-1]) * np.diff(t))])

# backed out from the delivered belief: Delta = xi*log(w_L/w_1) -> Y_imp
xi = 0.05
w = np.loadtxt(f"{D}/lambda3_weights_distorted.txt")
Delta = xi * np.log(w[-1] / w[0]); Yimp = YHAT + np.sqrt(6.0 * Delta)

plt.rcParams.update({"font.size": 14})
fig, ax = plt.subplots(figsize=(8.2, 4.2))
ax.axhline(YBAR, ls="--", color="0.45", lw=1.7, label=r"model bound ($4^\circ$C)")
ax.axhline(Yimp, color="#D55E00", lw=3.0, label="temperature the damage belief prices")
ax.plot(t, Yfwd, color="#0072B2", lw=3.0, label="if damaged: temperature the economy reaches")
ax.plot(t, Y, color="#56B4E9", lw=2.4, ls=(0, (5, 2)), label="actual (no-shock) temperature")
ax.axhline(YHAT, color="0.7", lw=1.0, ls=":")
ax.set_xlabel("year"); ax.set_ylabel(r"temperature ($^\circ$C)")
ax.set_xlim(0, 60); ax.set_ylim(1.0, 4.2); ax.grid(alpha=.2)
ax.legend(frameon=False, loc="center left", fontsize=12.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "byyear.png"), dpi=160, bbox_inches="tight")
print(f"byyear.png | Delta={Delta:.3f} -> Y_imp={Yimp:.2f}C ; forward reaches {Yfwd[-1]:.2f}C ; actual {Y[-1]:.2f}C")
