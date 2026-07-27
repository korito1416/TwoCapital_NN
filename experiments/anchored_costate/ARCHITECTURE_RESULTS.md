# Architecture Sweep Results

Date: 2026-06-30.  All runs use the validated Torch HJB ports and the hard
cross-regime value anchor.  Seed `0`, float32, batch `128`, validation
`2 x 256`, 1000 training steps unless noted.

## Root-Level Identification Benchmarks

Existing root-regime benchmark, 3 seeds, 3000 steps:

| method | HJB RMS | FOC d/g | level spread |
|---|---:|---:|---:|
| baseline direct value | `1.40e-2` | huge / huge | `1.092` |
| no padding | `4.27e-3` | `1.29e-1 / 1.12e-1` | `0.224` |
| structural xi | `3.39e-3` | `2.06e-1 / 1.51e-1` | `0.552` |
| costate / EGM level recovery | `3.00e-3` | `2.06e-2 / 9.53e-3` | `0.0077` |
| recentered value net | `~2.1e-3` | comparable | `0.0051` |

Conclusion: costate/EGM is correct in spirit, but the simpler recentered value
net pins the same additive constant more cleanly.  The two-regime experiments
therefore use recentering as the base layer.

## Two-Regime Architecture Sweep

Regimes: `PostDamagePostTech -> PreDamagePostTech`.

| variant | root HJB | pre HJB | boundary gap | root FOC d/g | pre FOC d/g | read |
|---|---:|---:|---:|---:|---:|---|
| anchor_clean | `2.03e-3` | `6.13e-3` | `1.0e-8` | `3.41e-2 / 1.11e-2` | `4.55e-2 / 5.46e-2` | baseline anchored architecture |
| deep_6x32 | `1.60e-3` | `7.15e-3` | `1.0e-8` | `1.58e-2 / 1.20e-2` | `4.59e-2 / 5.13e-2` | more depth helps root, not pre |
| frozen_middle | `2.26e-3` | `8.41e-3` | `1.0e-8` | `6.70e-2 / 4.25e-2` | `5.48e-2 / 5.67e-2` | frozen benchmark middle layers hurt |
| frozen_middle_control | `1.79e-3` | `6.62e-3` | `1.0e-8` | `4.23e-2 / 3.36e-2` | `5.35e-2 / 5.41e-2` | control steps do not rescue frozen middle |
| control_strong | `1.58e-3` | `4.36e-3` | `1.0e-8` | `8.53e-3 / 5.57e-3` | `4.42e-2 / 5.10e-2` | better root/control, pre still limited |
| federated_grad | `2.32e-3` | `1.10e-3` | `1.0e-8` | `1.52e-2 / 2.06e-2` | `9.36e-3 / 1.02e-2` | best pure cross-regime communication |
| federated_control | `1.81e-3` | `1.06e-3` | `1.0e-8` | `1.18e-2 / 1.65e-2` | `3.09e-3 / 4.15e-3` | best overall |

Additional notes:

- `wide_4x64` was attempted in the all-sweep job, but the 1000-step CPU run was
  cancelled after the wide subprocess dominated runtime.  Given that 6-layer
  depth did not improve pre-regime accuracy and federated coupling did, wide is
  not the current priority.
- A 5000-step conservative baseline without federated gradients reached root HJB
  `6.27e-4`, pre HJB `4.03e-3`, and boundary gap `1e-8`.  Federated coupling
  reaches pre HJB `~1.1e-3` already at 1000 steps.

## Recommended Architecture

Use this as the next production candidate:

1. Recentered value network in every regime:
   `v = phi(x) - phi(anchor(x)) + v_anchor(x)`.
2. Hard cross-regime value anchor at jump boundaries.
3. Federated gradient coupling:
   upstream pre-regime PDE and jump target gradients are allowed to flow into
   the downstream/post-regime value network.
4. Extra control improvement on the pre-regime controls:
   `control_steps=3`, `pre_control_weight=3`.
5. Do not freeze benchmark middle layers as a default.  If transfer learning is
   desired, distill from benchmark outputs or derivatives first; direct frozen
   hidden-layer transplantation was worse in this test.

The best current design is `federated_control`.
