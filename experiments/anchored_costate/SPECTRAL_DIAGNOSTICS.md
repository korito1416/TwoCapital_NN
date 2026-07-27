# Spectral Diagnostics

This note records the first frequency-domain diagnostic pass.  It compares the
original TF/Torch solver with the anchored/federated experiment checkpoints on a
fixed two-dimensional `(Z, Y)` slice.

The diagnostic script is:

```bash
experiments/anchored_costate/spectral_diagnostics.py
```

It evaluates HJB residuals on a grid and applies an orthonormal 2D DCT-II.  DCT
is used instead of a periodic FFT because the state box has nonperiodic finite
boundaries.

## Slice

Unless otherwise noted:

```text
grid    = 32 x 32
logK    = 5.5
logR    = 3.5
lambda3 = 1/6
Z       in [0.05, 0.95]
Y       in [0.2, 3.8]
```

Two robustness values were checked:

```text
xi = 0.05  -> logxi = -2.9957
xi = 0.10  -> logxi = -2.3026
```

Outputs:

```text
experiments/anchored_costate/results/spectral_diagnostics_lowxi
experiments/anchored_costate/results/spectral_diagnostics_xi0p1
experiments/anchored_costate/results/spectral_diagnostics_lowxi_focuspre
```

Each folder contains:

- `spectral_summary.csv`
- `spectral_summary.json`
- residual and DCT coefficient arrays
- `spectral_diagnostics.pdf`

## Main Low-Xi Result

Balanced anchored/federated checkpoint:

```text
experiments/anchored_costate/results/precision_push_batch256/
seed0_20000steps_20260630_210314/fedctrl_long_base
```

At `xi=0.05`:

| model | RMS residual | mean residual | DC energy | centered high-frequency energy | FOC d/g |
|---|---:|---:|---:|---:|---:|
| original root | `6.05e-4` | `-1.28e-4` | `4.47%` | `0.67%` | `3.42e-4 / 3.69e-4` |
| original pre | `2.37e-3` | `2.18e-3` | `85.03%` | `0.73%` | `1.03e-3 / 6.21e-4` |
| new root | `2.40e-4` | `1.08e-4` | `20.22%` | `0.61%` | `4.67e-3 / 1.85e-3` |
| new pre | `1.28e-2` | `-1.28e-2` | `99.86%` | `0.88%` | `2.20e-1 / 2.13e-1` |
| costate root | `1.28e-3` | `7.42e-5` | `0.34%` | `0.07%` | `5.35e-3 / 5.13e-3` |

At `xi=0.10`, the pattern is similar:

| model | RMS residual | mean residual | DC energy | centered high-frequency energy |
|---|---:|---:|---:|---:|
| original root | `5.69e-4` | `-2.63e-5` | `0.21%` | `0.67%` |
| original pre | `1.53e-3` | `6.50e-4` | `18.04%` | `0.42%` |
| new root | `2.48e-4` | `1.44e-4` | `33.54%` | `0.69%` |
| new pre | `1.26e-2` | `-1.26e-2` | `99.84%` | `0.64%` |
| costate root | `1.29e-3` | `1.44e-4` | `1.26%` | `0.07%` |

## Interpretation

The new anchored root is better than the original root on this slice in HJB RMS.
However, the new anchored pre-regime has a large fixed low-`xi` bias on this
slice.  The error is almost entirely the zero-frequency/DC component, not a
high-frequency oscillation.

This matters because random-box validation gave the same checkpoint a strong
overall pre-regime RMS (`~7.85e-4`).  The frequency slice shows that the random
validation is averaging over a localized low-`xi` bias.

The pre-focused checkpoint does not fix this slice:

```text
experiments/anchored_costate/results/precision_push/
seed0_20000steps_20260630_210314/fedctrl_focus_pre
```

At `xi=0.05`, its pre residual is still dominated by a DC mode:

```text
new pre RMS  = 1.60e-2
new pre mean = -1.60e-2
DC energy    = 99.96%
```

So the issue is not ordinary high-frequency roughness; it is a low-frequency
level/Hamiltonian offset at fixed low `xi`.

## Consequences

Fourier/DCT diagnostics support the original identification diagnosis:

- High-frequency residual energy is small across models.
- The hard case is the low-frequency / level-like mode.
- Anchoring fixes cross-regime boundary gaps, but fixed-`xi` pre-regime HJB
  still needs targeted low-frequency control.

Likely next tests:

1. Add fixed low-`xi` validation slices to every architecture run.
2. Oversample low `xi` in training, especially in pre-regimes.
3. Add a DC/low-mode residual penalty computed by map-reduce over `(Z,Y)` slices.
4. Combine costate training with cross-regime anchoring for the pre-regime, not
   only the root regime.
