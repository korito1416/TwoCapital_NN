# Anchored Costate / Joint-Regime Experiments

This sandbox tests the proposed fix for weakly identified value levels in the
Torch HJB ports.  Production `models/`, `models_dgm/`, and `models_torch/` are
not modified.

## Core Idea

The HJB pins the value level only through `-δV`, with `δ=0.01`, while jump terms
use level differences inside `exp(-(V_post - V_pre)/ξ)`.  Small level drift can
therefore become a large distortion error when `ξ` is small.

The experiment removes the nearly free additive constant by using

```text
v(x) = phi(x) - phi(x_anchor; pseudo(x)) + v_anchor.
```

For the two-regime case, `PreDamagePostTech` also anchors its value at the damage
jump boundary to the λ3-average of `PostDamagePostTech`.

## Files

- `anchored_nets.py`: shared normalized MLPs, recentered value net, losses, and
  damage-boundary gap diagnostics.
- `train_root_recentered.py`: single-regime terminal test.
- `train_two_regime_joint.py`: joint `PostDamagePostTech -> PreDamagePostTech`
  test with cross-regime communication.
- `costate_robust_root.py`: costate-first terminal-regime prototype with
  `xi`-conditioned Brownian `RobustMinimizerNet`.
- `train_two_regime_learned_jump.py`: two-regime prototype with learned
  `xi`-conditioned damage-jump `log_g` minimizer.
- `pretrain_jump_minimizer.py`: simple benchmark distillation for a
  `JumpRobustNet`, used to test transfer learning before any production work.
- `run_anchored.sbatch`: RCC/Slurm entry point.
- `run_costate_robust_root.sbatch`: RCC entry point for Brownian minimizer
  closed/residual/direct comparisons.
- `run_two_regime_learned_jump.sbatch`: RCC entry point for learned jump
  minimizer comparisons.
- `collect_robust_minimizer_results.py`: scans Slurm logs and writes a summary
  CSV of `RESULT_JSON` rows.
- `train_two_regime_lowmode.py`: warm-started two-regime trainer with
  map-reduced low-frequency/DCT slice losses for the low-`xi` pre-regime
  residual offset.
- `run_lowmode_map_gpu.sbatch`: GPU Slurm array that maps several low-mode
  weighting designs from the same warm-start checkpoint.
- `run_lowmode_reduce.sbatch`: dependent reduce job that summarizes completed
  map runs and runs spectral diagnostics on the best checkpoint.
- `collect_lowmode_results.py`: reducer for `history.jsonl` files from the
  low-mode map array.
- `train_two_regime_analytic_costate.py`: warm-started two-regime trainer that
  uses FOC-implied analytical costate targets
  `q_d=MU/phi_d'(i_d)`, `q_g=MU/phi_g'(i_g)` to correct `V_logK` and `V_Z`.
- `run_analytic_costate_gpu.sbatch`: GPU Slurm array for `q`, `p`, and
  combined analytical-costate projections.
- `run_analytic_costate_reduce.sbatch`: dependent reducer plus spectral
  diagnostics for the analytical-costate runs.
- `collect_analytic_costate_results.py`: reducer that ranks by HJB, FOC,
  low-mode residual, and slice costate errors.
- `ARCHITECTURE_RESULTS.md`: completed architecture sweep and recommendation.
- `PRECISION_PUSH.md`: overnight precision-push runs and guardrails.
- `COSTATE_ROBUST_REDESIGN.md`: costate-first + learned robustness architecture
  note.
- `SPECTRAL_DIAGNOSTICS.md`: DCT/frequency-domain residual diagnostics comparing
  original and anchored/costate experiment solvers.

## Local Smoke Runs

```bash
module unload python
module load python/anaconda-2021.05
cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal

python -u experiments/anchored_costate/train_root_recentered.py \
  --steps 20 --log_every 10 --valid_batch_size 64 --valid_batches 1 \
  --export experiments/anchored_costate/results/root_smoke

python -u experiments/anchored_costate/train_two_regime_joint.py \
  --steps 20 --log_every 10 --valid_batch_size 64 --valid_batches 1 \
  --export experiments/anchored_costate/results/two_smoke
```

## RCC Runs

Project convention: substantive training and sweeps should run on RCC/Midway
via Slurm.  Local runs are only for syntax checks and tiny smoke tests.

```bash
sbatch --export=ALL,MODE=root,STEPS=5000,SEED=0 \
  experiments/anchored_costate/run_anchored.sbatch

sbatch --export=ALL,MODE=two,STEPS=10000,SEED=0 \
  experiments/anchored_costate/run_anchored.sbatch
```

Useful switches:

- `DTYPE=float64` for double precision checks.
- `ROOT_CONTROL_INIT=clean` to remove TF-surrogate controls.
- `COUPLE_ANCHOR_GRAD=1` to let the pre-regime anchor send gradients into the
  root value net.
- `COUPLE_NEIGHBOR_GRAD=1` to let pre-regime jump-target terms send gradients
  into the root value net.
- `EXTRA_ARGS="--foc_in_value"` to include FOCs in the value step.

Default mode is conservative: root controls start from the validated surrogate,
the pre regime reads the root as a detached teacher, and the hard boundary anchor
handles level communication.

## Low-Mode Map-Reduce Run

The current failure in the hard low-`xi` slice is a low-frequency/DC HJB offset,
not high-frequency noise.  The low-mode trainer starts from
`precision_push_batch256/.../fedctrl_long_base`, maps several slice-loss
weighting designs across a GPU Slurm array, and reduces by a combined score over
slice RMS, slice mean/DC, pre/root HJB, and FOC RMS.

Submitted run:

```text
map array job: 51347645
reduce job:    51347646
output root:   experiments/anchored_costate/results/lowmode_map_20260701_122130
```

Partial result from completed task 0 (`dc1_rms025_foc025`, 3000 steps):

| metric | warm start | task 0 final |
|---|---:|---:|
| root HJB RMS | `5.32e-4` | `4.88e-4` |
| pre HJB RMS | `1.30e-2` | `4.01e-3` |
| low-`xi` slice RMS | `1.27e-2` | `1.78e-3` |
| low-`xi` slice mean/DC | `-1.27e-2` | `1.08e-4` |
| low-`xi` slice FOC d/g | `2.21e-1 / 2.12e-1` | `2.26e-2 / 2.53e-2` |

The 32x32 spectral diagnostic confirms that the improvement is not just a
training-grid artifact:

| model/slice | RMS | mean | DC energy | centered high freq |
|---|---:|---:|---:|---:|
| previous anchored pre, low `xi` | `1.28e-2` | `-1.28e-2` | `99.86%` | `0.88%` |
| low-mode task 0 pre, low `xi` | `1.69e-3` | `1.05e-4` | `0.39%` | `0.75%` |
| low-mode task 0 pre, `xi=0.1` | `1.64e-3` | `3.24e-4` | `3.90%` | `0.75%` |

A follow-up refinement is queued separately:

```text
refine job:    51348740
reduce job:    51348741
output root:   experiments/anchored_costate/results/lowmode_refine_20260701_124412
warm start:    lowmode_map_20260701_122130/dc1_rms025_foc025_seed0_task0
design:        24x24 slice grid, stronger RMS/low-DCT/FOC weights, lower LR
```

## Analytical Costate Correction

The next experiment uses the FOCs as analytical teachers for marginal values:

```text
q_d = V_logK - Z V_Z       = MU / phi_d'(i_d)
q_g = V_logK + (1-Z) V_Z   = MU / phi_g'(i_g)
V_Z = q_g - q_d
V_logK = (1-Z) q_d + Z q_g
```

Implementation details:

- Controls are detached in the value step, following the envelope logic.
- The target loss only needs first state derivatives of `V`, so it is much
  cheaper than differentiating the full HJB residual.
- Fixed `(Z,Y)` slice grids are cached once; HJB residual chunks are
  map-reduced, while slice costate errors are evaluated vectorized on the whole
  grid.

Submitted run:

```text
map array job: 51348945
reduce job:    51348946
output root:   experiments/anchored_costate/results/analytic_costate_20260701_130003
warm start:    lowmode_map_20260701_122130/dc1_rms025_foc025_seed0_task0
variants:      q_mild, both_mild, both_strong, p_focus
```

When the dependency finishes, inspect:

```text
experiments/anchored_costate/results/lowmode_map_20260701_122130/lowmode_summary.csv
experiments/anchored_costate/results/lowmode_map_20260701_122130/best_path.txt
experiments/anchored_costate/results/lowmode_map_20260701_122130/spectral_best_lowxi/spectral_summary.csv
experiments/anchored_costate/results/lowmode_map_20260701_122130/spectral_best_xi0p1/spectral_summary.csv
```

## Xi-Conditioned Robust Minimizer Prototype

The learned minimizer is implemented only in this sandbox.  The production model
files are not modified.

Root Brownian distortion test, 2000 steps, seed 11:

| minimizer | HJB RMS | FOC d/g | inner gap | minimizer error |
|---|---:|---:|---:|---:|
| closed form | `2.606e-3` | `6.89e-3 / 4.03e-3` | `~0` | `~0` |
| residual net | `2.602e-3` | `7.42e-3 / 4.23e-3` | `3.75e-9` | `1.28e-5` |
| direct net | `2.592e-3` | `7.30e-3 / 4.14e-3` | `3.11e-5` | `4.12e-3` |

Interpretation: a learned minimizer can be used, but the correct architecture is
closed-form residual parameterization, not pure direct adversarial learning.  The
direct net can give a comparable outer HJB while still failing the inner
robust-control objective.

Two-regime learned damage-jump minimizer, 1500 steps, seed 41:

| minimizer | pre HJB RMS | root HJB RMS | jump gap | `log_g` error |
|---|---:|---:|---:|---:|
| closed form | `4.501e-3` | `1.167e-3` | `~0` | `~0` |
| residual net | `4.472e-3` | `1.189e-3` | `~1e-8` | `1.35e-5` |
| direct net | `4.446e-3` | `1.189e-3` | `3.52e-5` | `6.83e-3` |

The direct jump net starts with pre HJB around `2.22e+1` because the exponential
jump distortion is not anchored near
`log_g_closed = -(V_post - V_pre)/xi`.  It eventually lowers the outer HJB, but
it still fails the inner robust-control check.  The residual jump net matches
the closed-form minimizer while keeping the outer HJB competitive.

## Benchmark Pretraining / Transfer Smoke Test

A small direct `JumpRobustNet` was pretrained on the simple closed-form benchmark

```text
(DeltaV, xi) -> log_g_closed = -DeltaV / xi.
```

Small local distillation run:

| step | `log_g` RMSE | max error |
|---:|---:|---:|
| 0 | `1.61e+0` | `8.19e+0` |
| 600 | `1.16e-1` | `1.88e+0` |

Transfer result:

| initialization | step-0 pre HJB | step-0 jump gap | comment |
|---|---:|---:|---|
| random direct | `2.22e+1` | `1.03e+1` | unstable/explosive |
| benchmark-pretrained direct | `2.77e-2` | `7.68e-4` | explosion removed |

This confirms the transfer-learning idea qualitatively: even a rough benchmark
minimizer teaches the direct net the right `xi`/`DeltaV` scale and prevents the
initial jump explosion.  But it is not accurate enough yet to replace the
residual architecture; frozen transfer with this small net leaves pre HJB around
`4.7e-2`, while closed/residual robust minimizers are around `4.5e-3`.

## First RCC Baseline

Job `51332823`, `MODE=two`, `STEPS=5000`, `BATCH_SIZE=128`,
`VALID_BATCH_SIZE=512`, `VALID_BATCHES=2`, seed `0`, float32:

| metric | step 0 | step 5000 |
|---|---:|---:|
| root HJB RMS | `1.70e-2` | `6.27e-4` |
| pre HJB RMS | `4.71e-2` | `4.03e-3` |
| boundary gap | `1.0e-8` | `1.0e-8` |
| root FOC d/g | `1.95e-1 / 2.33e-1` | `8.12e-3 / 6.20e-3` |
| pre FOC d/g | `8.48e-2 / 7.75e-2` | `3.35e-2 / 4.58e-2` |

Output:
`experiments/anchored_costate/results/rcc/two_seed0_20260630_194540_51332823`.

Interpretation: the value-level / cross-regime gap problem is fixed in this
two-regime sandbox; the next bottleneck is pre-regime control accuracy.
