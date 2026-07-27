# Precision Push Log

Goal: push the anchored two-regime sandbox toward `1e-5` HJB RMS without
changing the economics.  A run only counts if boundary gap stays pinned, FOC
does not deteriorate, and validation is evaluated on a fixed seed.

Validation protocol for the overnight jobs:

- regimes: `PostDamagePostTech -> PreDamagePostTech`
- validation: fixed `valid_seed=12345`
- validation batch: mostly `2 x 1024`; batch-256 run uses `2 x 2048`
- boundary gap target: `~1e-8`
- primary metrics: root HJB RMS, pre HJB RMS, root/pre FOC d/g

## Overnight Jobs

| job | variant | steps | batch | seed | hypothesis |
|---:|---|---:|---:|---:|---|
| `51334369` | `fedctrl_long_base` | 20000 | 128 | 0 | baseline long convergence of current winner |
| `51334368` | `fedctrl_long_base` | 20000 | 128 | 1 | robustness / seed sensitivity |
| `51334367` | `fedctrl_low_lr` | 20000 | 128 | 0 | lower LR polish can reduce residual floor |
| `51334370` | `fedctrl_preweight3` | 20000 | 128 | 0 | emphasize pre-regime HJB residual |
| `51334376` | `fedctrl_control5` | 20000 | 128 | 0 | stronger control improvement lowers FOC and residual |
| `51334378` | `fedctrl_focus_pre` | 20000 | 128 | 0 | strongly focus value objective on pre regime |
| `51334375` | `fedctrl_foc_value` | 10000 | 128 | 0 | include FOC terms in value step for consistency |
| `51334377` | `fedctrl_long_base` | 20000 | 256 | 0 | larger batch lowers stochastic residual floor |

Output roots:

- `experiments/anchored_costate/results/precision_push`
- `experiments/anchored_costate/results/precision_push_batch256`

Collect results after jobs finish:

```bash
python experiments/anchored_costate/collect_precision.py
```

This writes:
`experiments/anchored_costate/results/precision_push_summary.csv`.

## Current Best Before Overnight

`federated_control`, 1000 steps:

- root HJB: `1.81e-3`
- pre HJB: `1.06e-3`
- boundary gap: `1e-8`
- pre FOC d/g: `3.09e-3 / 4.15e-3`

Conservative 5000-step baseline without federated gradients:

- root HJB: `6.27e-4`
- pre HJB: `4.03e-3`
- boundary gap: `1e-8`

## Morning Results: 2026-07-01

All eight overnight jobs completed successfully.

| variant | steps | root HJB | pre HJB | gap | pre FOC d/g | read |
|---|---:|---:|---:|---:|---:|---|
| `fedctrl_focus_pre` | 20000 | `3.53e-3` | **`1.21e-4`** | `1e-8` | `8.17e-4 / 5.78e-4` | best pre-regime residual, sacrifices root |
| `fedctrl_preweight3` | 20000 | `1.34e-3` | `3.72e-4` | `1e-8` | `8.03e-4 / 5.86e-4` | best compromise if pre is prioritized |
| `fedctrl_long_base` batch256 | 20000 | **`5.50e-4`** | `7.85e-4` | `1e-8` | `1.11e-3 / 6.93e-4` | best balanced/root-stable run |
| `fedctrl_control5` | 20000 | `5.90e-4` | `8.00e-4` | `1e-8` | `1.36e-3 / 1.05e-3` | extra control steps not a clear win |
| `fedctrl_long_base` seed0 | 20000 | `5.63e-4` | `8.23e-4` | `1e-8` | `1.60e-3 / 6.95e-4` | reproducible baseline |
| `fedctrl_long_base` seed1 | 20000 | `5.82e-4` | `8.73e-4` | `1e-8` | `6.87e-4 / 5.03e-4` | confirms seed stability |
| `fedctrl_foc_value` | 10000 | `4.41e-3` | `2.65e-3` | `1e-8` | `1.78e-4 / 1.40e-4` | makes FOC excellent but hurts HJB |
| `fedctrl_low_lr` | 20000 | `8.57e-3` | `8.44e-3` | `1e-8` | `1.18e-1 / 1.02e-1` | failed; too conservative / bad basin |

Best observed checkpoint:

- `fedctrl_focus_pre`, step `19000`: pre HJB `1.199e-4`, root HJB `3.530e-3`,
  pre FOC `8.18e-4 / 6.13e-4`.

Interpretation:

- The overnight push did not reach `1e-5`.
- It did lower the targeted pre-regime residual from the prior `~1.06e-3` to
  `~1.2e-4`.
- The current trade-off is clear: heavy pre weighting can push pre HJB down, but
  the root equation is no longer solved tightly enough.
- The most academically defensible balanced result is batch256
  `fedctrl_long_base`: root `5.50e-4`, pre `7.85e-4`, stable across seeds.

## Academic Guardrails

- Do not alter `models_torch/*` economics.
- Do not compare against training loss only; use fixed validation.
- Treat a lower HJB as invalid if FOC or feasibility deteriorates materially.
- Keep each line in a separate Slurm job/output directory.
- Prefer stable, reproducible convergence over a single lucky low residual.
