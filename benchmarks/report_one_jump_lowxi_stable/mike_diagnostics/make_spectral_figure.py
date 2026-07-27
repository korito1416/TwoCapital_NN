"""figH — clean spectral view of the terminal value error E = V_NN - V_FD (from spectral_fields.npz).
Left: DCT spectrum along Y (DC / low-frequency dominance). Right: E profile vs Y (smooth level offset,
no structure at the damage threshold y=1.5). No text on the figure; large; project colours."""
import os
import numpy as np
from scipy.fft import dct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aer_style; aer_style.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "spectral_fields.npz"), allow_pickle=True)
XIS = [("0.05", aer_style.DISTORTED), ("0.01", aer_style.BASELINE)]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 7))
for xi, col in XIS:
    E = d[f"xi_{xi}_E"]              # (lK, Z, Y) value error
    Y = d[f"xi_{xi}_Y"]
    # spectrum along Y: DCT per (lK,Z) line, RMS magnitude across lines per mode
    C = dct(E, axis=2, type=2, norm="ortho")
    spec = np.sqrt((C ** 2).mean(axis=(0, 1)))
    axL.plot(np.arange(len(spec)), spec, "o-", color=col, label=rf"$\xi={xi}$")
    # profile: mean error vs Y (a slowly-varying level offset)
    axR.plot(Y, E.mean(axis=(0, 1)), "-", color=col, label=rf"$\xi={xi}$")

axL.set_yscale("log")
axL.set_xlabel(r"DCT mode along $Y$ (wavenumber)")
axL.set_ylabel(r"error spectrum $|\hat E|$")
axL.legend()

axR.axvline(1.5, ls="--", color="0.4")   # damage threshold y_lower, no label text
axR.set_xlabel(r"$Y$ (temperature anomaly)")
axR.set_ylabel(r"mean value error $V_{\rm NN}-V_{\rm FD}$")
axR.legend()

fig.tight_layout()
out = os.path.join(HERE, "figH_spectral.png")
fig.savefig(out); plt.close(fig); print("wrote", out)
