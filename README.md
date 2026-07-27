# Climate Change, Innovation, and Uncertainty: Neural HJB Solver

This repository solves and analyzes a robust continuous-time climate economy
with dirty capital, green capital, R&D, damage jumps, and one- or two-stage
technology jumps. The numerical method is a neural-network Deep Galerkin /
policy-improvement algorithm applied to a coupled system of HJB equations.

The project has two main parts:

1. **Model solving:** train value and policy networks backward through the jump
   tree.
2. **Model analysis:** simulate deterministic and stochastic paths, plot
   controls and values around jumps, compute first-jump densities, audit HJB
   residuals, and decompose the social value of R&D (SVRD).

The full mathematical specification is in [AGENTS.md](AGENTS.md). The exact
competing-risk first-jump formulas are in
[FIRST_JUMP_DENSITY.md](FIRST_JUMP_DENSITY.md).

## Contents

- [Part I: Solving the Model](#part-i-solving-the-model)
- [Part II: Analyzing Trained Models](#part-ii-analyzing-trained-models)
- [Output Folder Anatomy](#output-folder-anatomy)
- [Python File Reference](#python-file-reference)
- [Slurm Stage File Reference](#slurm-stage-file-reference)
- [Bash Submission File Reference](#bash-submission-file-reference)

## Repository Layout

| Path | Purpose |
|---|---|
| `models/` | Supported feedforward value/policy networks, six HJB regimes, and simulation code. |
| `models_dgm/` | Experimental gated Deep Galerkin architecture with the same HJB equations. |
| `pretrained/` | Repo-local TensorFlow checkpoints used for warm starts. |
| `output_001/` | Main trained-model folders and their analyses. Not source controlled. |
| `output_largebatch_001/` | Large-batch continuation experiments. |
| `output_dgm_001/` | DGM architecture experiments. |
| `output_sensitivity_001/` | Parameter-sensitivity runs. |
| `job-outs/` | Slurm stdout and stderr logs. |
| `reports/` | Standalone report material. |

Paths in this project contain commas, `#`, and sometimes Greek characters.
Always quote a model path in shell commands.

## Model Summary

The dynamic states are

- `logK = log(Kd + Kg)`: log aggregate productive capital;
- `Z = Kg / (Kd + Kg)`: green capital share;
- `Y`: temperature anomaly;
- `logR`: log knowledge/R&D capital in regimes where research remains active.

The networks also receive pseudo-state inputs for robustness `logxi` and, in
post-damage regimes, realized damage curvature `lambda3`. The implementation
repeats `logxi` in several network columns to preserve the original
channel-specific input layout while imposing a common robustness parameter.

The policy rates and levels are related by

```text
Id = id * (1 - Z) * K
Ig = ig * Z * K
Ir = ir * K
```

Rates and levels are not interchangeable. A post-jump density can move in
different directions for `id` and `Id` because the capital stock and green
share also change.

## Part I: Solving the Model

### Six HJB Regimes

The two-stage model has six regimes. The input width is the serialized neural
network width, including fixed and repeated pseudo-state columns.

| Regime folder / Python module | Logical states | Input width | R&D | Continuation values used |
|---|---|---:|---:|---|
| `PostDamagePostTech` | `logK,Z,Y,lambda3,logxi` | 7 | No | None; terminal regime. |
| `PostDamageIntermTech` | `logK,Z,Y,logR,lambda3,logxi` | 8 | Yes | `PostDamagePostTech`. |
| `PostDamagePreTech` | `logK,Z,Y,logR,lambda3,logxi` | 8 | Yes | `PostDamageIntermTech`, `PostDamagePostTech`. |
| `PreDamagePostTech` | `logK,Z,Y,logxi` | 6 | No | Damage realizations in `PostDamagePostTech`. |
| `PreDamageIntermTech` | `logK,Z,Y,logR,logxi` | 7 | Yes | `PostDamageIntermTech`, `PreDamagePostTech`. |
| `PreDamagePreTech` | `logK,Z,Y,logR,logxi` | 7 | Yes | `PostDamagePreTech`, `PreDamageIntermTech`, `PreDamagePostTech`. |

### Two-Stage Technology Model

Solve backward through the jump tree. The dependency graph used by the modern
submission scripts is:

```text
PostDamagePostTech
  |-- PostDamageIntermTech
  |     |-- PostDamagePreTech
  |     `-- PreDamageIntermTech --.
  `-- PreDamagePostTech ---------+--> PreDamagePreTech
```

More explicitly:

1. Train `PostDamagePostTech`.
2. Train `PostDamageIntermTech` after step 1.
3. Train `PreDamagePostTech` after step 1; it can run in parallel with step 2.
4. Train `PostDamagePreTech` after step 2.
5. Train `PreDamageIntermTech` after steps 2 and 3.
6. Train `PreDamagePreTech` after steps 4 and 5.

Use the dependency pattern in
`submit/submit_two_stage_parameter_sensitivity_sweep.sh` or
`submit/submit_dgm_two_stage_sweep.sh`: each HJB is a separate Slurm job and downstream
jobs use `afterok` dependencies.

### One Technology Jump Models

There are two different one-jump reductions. They should not be mixed.

#### Direct jump to final technology (`pi = 1`)

This is the current recommended one-jump comparison. The intermediate branch
has zero intensity. The model has four regimes:

```text
PostDamagePostTech   PreDamagePostTech
PostDamagePreTech    PreDamagePreTech
```

The two post-tech HJBs are mathematically identical to the matching post-tech
HJBs in the two-stage model at the same technology intensity and calibration.
The fast, internally consistent workflow therefore:

1. Copies `PostDamagePostTech` and `PreDamagePostTech` from the matching
   two-stage model.
2. Trains `PostDamagePreTech` with a direct continuation to final technology.
3. Trains `PreDamagePreTech` after the post-damage/pre-tech model.
4. Runs deterministic simulations and plots.

Run it with:

```bash
bash submit/submit_pi1_inherit_posttech_fast.sh
```

The default script submits intensity scales 1 and 2. Its output folder names
begin with `OneTechJump_Pi_1p0_InheritedPostTech_`. No intermediate-stage
folder should exist in those results.

`submit/submit_pi1_direct_one_jump_and_y12_diagnostics.sh` is the older full-retraining
alternative. It trains the post-tech networks again, so numerical differences
from the two-stage post-tech solution can remain even though the HJB is the
same.

#### One jump to intermediate technology (`pi = 0`)

This is a historical exercise in which the direct final-tech branch and the
second technology jump are muted. It is launched by
`submit/submit_one_tech_jump_pi0_lr_sweep.sh`. It is not the direct-to-final one-jump
model and should not be used for that comparison.

### Neural Network Structure

Every regime has a separate value network `v_nn` and separate policy networks
`i_d_nn`, `i_g_nn`, and, when R&D is active, `i_r_nn`.

The supported `models/feedforward_subnet.py` architecture is:

1. Batch-normalize the input.
2. Apply each dense hidden layer and batch-normalize its output.
3. Sum all hidden-layer outputs before the final dense layer.
4. Apply a network-specific output transformation.

All hidden layers must have the same width because their outputs are summed.
The baseline uses Glorot initialization and the following configuration:

| Network | Hidden activation | Output transformation | Interpretation |
|---|---|---|---|
| Value `v_nn` | `swish` | `softplus` | Positive transformed value approximation. |
| Green rate `i_g_nn` | `tanh` | Custom bounded activation | Range `(-1/theta_g, 1)` keeps `log(1+theta_g*i_g)` valid. |
| Dirty rate `i_d_nn` | `tanh` | Custom bounded activation | Range `(-1/theta_d, 1)`; negative investment is allowed but bounded. |
| R&D rate `i_r_nn` | `softplus` | `softplus`, then `exp(-output)` | Positive R&D rate, at most one under this parameterization. |

The custom dirty/green activation is

```text
i(x; theta) = 1 - (1 + 1/theta) / (exp(2x) + 1).
```

It must use the run-specific `theta_d` and `theta_g`. Sensitivity jobs pass
those values through `models/params.py`, and deterministic simulation reloads
them from the trained stage's `params.txt`.

### Training Algorithm

Each training iteration draws a fresh stratified sample over the state and
pseudo-state ranges and performs two updates:

1. **Value update:** hold policy-network parameters fixed and update `v_nn`
   using the HJB residual, investment FOC residuals, feasibility penalties,
   and derivative-sign penalties.
2. **Policy update:** hold value-network parameters fixed and update all active
   policy networks jointly using the Hamiltonian/control objective and FOC
   residuals.

The implementation uses TensorFlow automatic differentiation for first,
second, and cross derivatives. Closed-form robust drift and jump distortions
are substituted into each HJB.

At each logging interval, the code evaluates independent large-sample
validation batches. It retains the checkpoint with the best finite validation
score and stops if the validation residual becomes nonfinite. Current stage
wrappers use Adam with global gradient clipping.

Important learning-rate detail: current model classes use optimizer 0 for the
value network and optimizer 1 for the combined controls. If four rates are
provided, optimizers 2 and 3 are constructed but are not used by the current
`train_step`; the first two rates are the effective rates.

### Baseline Hyperparameters

The main two-stage benchmark is:

```text
output_001/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000
```

| Setting | Post-tech regimes | R&D-active regimes |
|---|---:|---:|
| Hidden layers | 4 | 4 |
| Width | 32 | 32 |
| Batch size | 128 | 128 |
| Iterations | 1,000,000 | 1,000,000 |
| Logging interval | 1,000 | 1,000 |
| Value learning rate | `1e-5` | `1e-5` |
| Combined-control learning rate | `4e-3` | `1e-3` |
| Schedule | Warmup cosine | Warmup cosine |
| Warmup | First 1% of iterations | First 1% of iterations |

Training ranges are `logK in [4,7]`, `Z in [0.01,0.99]`, `Y in [0,4]`,
`logR in [1,6]`, `lambda3 in [0,1/3]`, and `logxi in [-3,5]` for the baseline.

Key baseline economic and jump parameters are:

| Parameter | Value |
|---|---:|
| `A_d`, `A_g`, `A_g_prime`, `A_g_prime_prime` | `0.1303`, `0.1085`, `0.1303`, `0.1567` |
| `alpha_d = alpha_g` | `-0.035` |
| `Gamma_d = Gamma_g` | `0.060` |
| `theta_d = theta_g` | `16.7` |
| `sigma_d = sigma_g` | `0.01` |
| `psi0`, `psi1`, `sigma_kappa` | `0.10583`, `0.5`, `0.0078` |
| Technology branch probability `pi` | `0.04` in the two-stage baseline |
| Technology intensity | `tech_jump_intensity_scale * R / varrho`, with `varrho=746.67` |
| Damage realizations | `lambda3 = {0,1/12,1/6,1/4,1/3}` |
| Damage intensity parameters | `r1=1.5`, `r2=0.36`, active above `Y=1.5` |

The complete calibration is in `models/params.py` and `AGENTS.md`. Current
analysis uses `Y0=1.2`. Some historical stage `params.txt` files record
`Y0=1.1`; an explicit `SIMULATION_Y0`/`--y0` setting is authoritative for a
new analysis run.

### Pretrained Checkpoints

Two checkpoint bundles are included:

- `pretrained/nber_legacy/`: the four-regime legacy initialization used when a
  model is trained with `pretrained_path=None`;
- `pretrained/two_stage_tech_base/`: a complete six-regime two-stage solution
  for downstream continuation and sensitivity runs.

Verify them with:

```bash
cd pretrained
sha256sum -c MANIFEST.sha256
cd ..
```

To override the legacy source:

```bash
export NBER_PRETRAINED_FOLDER=/path/to/nber_legacy
```

To continue from the bundled six-regime model:

```bash
export PRETRAINED_FOLDER="$PWD/pretrained/two_stage_tech_base"
```

When `PRETRAINED_FOLDER` is set, each solver loads the same regime's value and
policy weights from that folder. Continuation values for jump destinations are
loaded from the new target folder, which is why the stage dependencies must be
respected.

## Part II: Analyzing Trained Models

Set a model folder once for the examples below:

```bash
MODEL="$PWD/output_001/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
```

Before analysis, verify that every required regime contains `checkpoint`,
`*.index`, and `*.data-*` files.

### Deterministic Simulation

The deterministic simulator evaluates optimal controls on a reference-measure
state path without Brownian draws or realized jumps. It still evaluates robust
distortions, state-dependent jump intensities, and conditional first-jump
densities along that path.

Submit simulation and plotting jobs with an explicit dependency:

```bash
sim_jid=$(EXPORT_FOLDER="$MODEL" \
  SIMULATION_XIS="0.05,0.1,0.3,148.6" \
  SIMULATION_Y0="1.2" \
  SIMULATION_T="60" \
  SIMULATION_DT="0.08333333333333333" \
  sbatch --parsable --export=ALL,MODE=simulate sbatch/deterministic_stage.sbatch)

EXPORT_FOLDER="$MODEL" \
SIMULATION_XIS="0.05,0.1,0.3,148.6" \
sbatch --dependency="afterok:${sim_jid}" \
  --export=ALL,MODE=plot sbatch/deterministic_stage.sbatch
```

`148.6` is used as the numerical uncertainty-neutral/infinite-`xi` case.

Results are written to:

```text
<MODEL>/SimulationDeterministic/SimulationOutputs_ξ_<xi>/
<MODEL>/SimulationDeterministicPlot/
```

Per-`xi` output includes state paths, rates `i_d/i_g/i_r`, levels
`I_d/I_g/I_r`, consumption, output, the consumption/output ratio, distortions, jump intensities,
first-jump subdensities, conditional densities, and cumulative jump
probabilities. The second folder contains cross-`xi` figures.

Read the first-jump outputs carefully:

- `dmg_jump_subdensity` and `tech_jump_subdensity` are unnormalized competing-
  risk subdensities. Their joint area is the probability of a first jump by
  the finite simulation horizon.
- `conditional_dmg_jump_density` and `conditional_tech_jump_density` condition
  on a first jump occurring by that horizon. Their joint area should be one.
- `first_jump_type_*_prob` conditions on the exact first-jump time and reports
  which jump type occurs.

See `FIRST_JUMP_DENSITY.md` for the equations and accounting identities.

### Stochastic Simulation with Jumps

Use `models/SimulationStochasticJumps.py`, not the legacy
`models/SimulationRandom*.py`, for new work. It simulates diffusion paths,
damage events, technology transitions, controls, values, and event windows.

Direct Python example:

```bash
python models/SimulationStochasticJumps.py \
  --export-folder "$MODEL" \
  --xi 0.1 \
  --seed 1 \
  --n-paths 100 \
  --years 60 \
  --dt 0.08333333333333333 \
  --y0 1.2
```

Direct Slurm-array example:

```bash
EXPORT_FOLDER="$MODEL" XI=0.1 N_PATHS=100 YEARS=60 \
DT=0.08333333333333333 Y0=1.2 \
sbatch --array=1-100 --export=ALL sbatch/stochastic_jump_stage.sbatch
```

The standard three-model submission is:

```bash
bash submit/submit_stochastic_jump_simulations.sh
```

With the defaults, each `xi` uses 100 array tasks and each task simulates 100
paths, for 10,000 paths per model/`xi` combination.

Results are saved under:

```text
<MODEL>/SimulationResults/paths_ξ_<xi>/
```

Important files include:

- `paths_controls_values_<seed>.npz`: compact state, policy, value, and event
  arrays;
- individual `*_array_<seed>.npy` files for compatibility;
- `event_windows_<seed>.csv`: observations immediately around each jump;
- `event_summary_<seed>.csv`: event counts and summaries;
- `metadata_<seed>.txt`: model, seed, `xi`, `Y0`, horizon, and jump settings.

### Stochastic Control, Value, and Marginal-Value Densities

After stochastic simulations finish, run:

```bash
EXPORT_FOLDER="$MODEL" Y0=1.2 SEED_MIN=1 SEED_MAX=100 \
COMPARISON=pre-post sbatch --export=ALL sbatch/stochastic_density_plot_stage.sbatch
```

The stage runs both plotting programs:

```bash
python analysis/plot_stochastic_control_densities.py \
  --export-folder "$MODEL" --control-group all \
  --comparison pre-post --y0 1.2 --seed-min 1 --seed-max 100

python analysis/plot_stochastic_marginal_value_densities.py \
  --export-folder "$MODEL" --comparison pre-post \
  --y0 1.2 --seed-min 1 --seed-max 100
```

`pre-post` compares the time index immediately before a jump with the first
index after it. `post-next` compares the first and second post-jump indices.
The blue solid curve is the left comparison distribution; the red dashed curve
is the right distribution.

The plotter recognizes four event types:

- `tech_0_to_1`: pre-tech to intermediate tech;
- `tech_1_to_2`: intermediate to final tech;
- `tech_0_to_2`: direct pre-tech to final tech;
- `damage_jump`: pre-damage to realized-damage regime.

Figures are written inside each `paths_ξ_*` folder and copied into the compact
report tree:

```text
<MODEL>/SimulationResultsPlot/paths_ξ_<xi>/ControlDensities/
```

The groups are:

- `rate_*`: `i_d`, `i_g`, and active `i_r`;
- `level_*`: `I_d`, `I_g`, and active `I_r`;
- `value_*`: value function before and after the event;
- `marginal_*`: `V_Kd`, `V_Kg`, and positive climate marginal cost `-V_Y`.

If an event has very few observations, interpret its KDE cautiously. The
plotter reports seeds and event counts in its summary CSV.

### SVRD Decomposition

`analysis/compute_svrd_decomposition.py` computes the social value of R&D before the
first jump by robust no-jump diffusion Monte Carlo. It decomposes the result:

1. by potential first-jump contribution;
2. by state-variable/cash-flow channel.

Run one model:

```bash
MODEL_FOLDER="$(basename "$MODEL")" \
SVRD_XIS="0.05 0.1 148.6" Y0=1.2 \
bash submit/submit_svrd_decomposition.sh
```

Run the standard one-jump and two two-stage models:

```bash
bash submit/submit_svrd_control_density_models.sh
```

Direct Python example:

```bash
python analysis/compute_svrd_decomposition.py \
  --export-folder "$MODEL" --xi 0.1 --n-paths 512 \
  --years 80 --dt 0.08333333333333333 --y0 1.2
```

Results are saved under `<MODEL>/SVRDDecomposition/xi_<xi>/`:

- `svrd_decomposition.csv`: all components;
- `svrd_by_potential_jump.csv` and `.png`;
- `svrd_by_state_channel.csv` and `.png`;
- `svrd_path_totals.npz`;
- `metadata.txt` with the direct derivative and Monte Carlo identity checks.

The jump decomposition total and state-channel total should agree with each
other and with the direct scaled SVRD within Monte Carlo error. A large gap is
a diagnostic failure, not an economic result.

### Residual and Network Audits

Summarize logged training histories:

```bash
python analysis/summarize_training_histories.py \
  --folder "$MODEL" --output "$MODEL/training_history_summary.csv"
```

Evaluate all trained networks on independent samples:

```bash
EXPORT_FOLDER="$MODEL" SAMPLE_SIZE=16384 CHUNK_SIZE=1024 \
sbatch --export=ALL sbatch/network_audit_stage.sbatch
```

The audit reports PDE/FOC residual distributions, control distributions,
finite-value fractions, and optional checkpoint distance from a reference
folder. Training loss alone is not sufficient evidence that two models agree.

For the baseline two-stage model, final logged PDE RMSEs are approximately
`1.2e-3` to `2.2e-3`, while FOC RMSEs are generally `1e-5` to `2e-4`. Treat
these as scale references, not universal acceptance thresholds.

### Comparing Models Correctly

Use the following comparison rules:

1. Compare the same regime, state vector, `xi`, damage realization, and green
   productivity.
2. Use the same `Y0`, horizon, time step, and random seeds for path comparisons.
3. Compare investment rates and levels separately.
4. For a direct-final one-jump model, post-tech/post-damage and
   post-tech/pre-damage networks should be identical to the inherited
   two-stage checkpoints. Checkpoint hashes or pointwise network evaluations
   are stronger evidence than similar training losses.
5. Compare stochastic density shapes only after checking event counts and the
   metadata `Y0` filter.
6. Compare conditional first-jump densities with conditional densities, not
   with raw subdensities.
7. Require SVRD jump totals and state-channel totals to satisfy their reported
   identity check within Monte Carlo uncertainty.

## Output Folder Anatomy

A complete trained model usually contains:

```text
<MODEL>/
  run_manifest.txt
  PostDamagePostTech/
  PostDamageIntermTech/       # absent in direct-final one-jump models
  PostDamagePreTech/
  PreDamagePostTech/
  PreDamageIntermTech/        # absent in direct-final one-jump models
  PreDamagePreTech/
  SimulationDeterministic/
  SimulationDeterministicPlot/
  SimulationResults/
  SimulationResultsPlot/
  SVRDDecomposition/
```

Each trained regime contains:

- `params.txt`: economic, sampling, optimizer, and path metadata;
- `params_*_nn_config.txt`: architecture metadata;
- `training_history.csv`: validation residual history;
- `v_nn_checkpoint_*`: value checkpoint;
- `i_d_nn_checkpoint_*`, `i_g_nn_checkpoint_*`, and optionally
  `i_r_nn_checkpoint_*`: policy checkpoints;
- `loss_*.png`: loss-history plots.

Read `run_manifest.txt` first. Folder names are useful labels, but the manifest
and stage `params.txt` are the authoritative record of what ran.

## Python File Reference

Run CLI programs from the repository root. The six regime files are normally
invoked through Slurm stage wrappers rather than by hand.

| File | Role |
|---|---|
| `models/PostDamagePostTech.py` | Terminal post-damage/final-tech HJB; trains value, dirty rate, and green rate networks. |
| `models/PostDamageIntermTech.py` | Post-damage/intermediate-tech HJB with R&D and a continuation to final technology. |
| `models/PostDamagePreTech.py` | Post-damage/pre-tech HJB with intermediate and direct-final technology branches. For `pi=1`, only the direct-final branch is active. |
| `models/PreDamagePostTech.py` | Pre-damage/final-tech HJB; integrates over damage realizations and has no R&D. |
| `models/PreDamageIntermTech.py` | Pre-damage/intermediate-tech HJB with technology and damage continuations. |
| `models/PreDamagePreTech.py` | Initial HJB with all active diffusion and jump channels. |
| `models/feedforward_subnet.py` | Supported feedforward subnet, stratified sampling, LR schedules, optimizers, validation scoring, and gradient clipping. |
| `models/params.py` | Baseline economic/state calibration plus environment-based sensitivity overrides and custom investment activation. |
| `models/pretrained_paths.py` | Resolves repo-local or environment-specified legacy warm-start checkpoints. |
| `models/SimulationDeterministic.py` | Loads a trained one- or two-stage model, generates deterministic paths, distortions, and first-jump accounting. Run as `python models/SimulationDeterministic.py <MODEL>`. |
| `models/SimulationDeterministicPlot.py` | Aggregates deterministic text outputs into cross-`xi` figures. Run as `python models/SimulationDeterministicPlot.py <MODEL>`. |
| `models/SimulationStochasticJumps.py` | Supported stochastic diffusion/jump simulator with controls, values, event windows, and metadata. |
| `models/SimulationRandom.py` | Legacy hardcoded six-regime random simulator. Retained for provenance; use `SimulationStochasticJumps.py` instead. |
| `models/SimulationRandom_1.py` | Older simulator tied to another project path and obsolete imports. Not supported in this checkout. |
| `analysis/audit_trained_networks.py` | Large independent-sample network/residual audit, optionally relative to a reference folder. |
| `analysis/summarize_training_histories.py` | Writes initial, best, and final loss values from every stage's `training_history.csv`. |
| `analysis/plot_stochastic_control_densities.py` | KDE plots for control rates, levels, and value around stochastic jump events. |
| `analysis/plot_stochastic_marginal_value_densities.py` | Evaluates network derivatives on stochastic event states and plots `V_Kd`, `V_Kg`, and `-V_Y`. |
| `analysis/plot_brown_capital_marginal_tech_jump.py` | Temperature profiles and FOC gaps for dirty-capital marginal values in selected regimes. |
| `analysis/compute_svrd_decomposition.py` | Monte Carlo SVRD decomposition by potential jump and state channel. |
| `analysis/generate_control_density_tex.py` | Generates the three hardcoded LaTeX density reports used by the current paper workflow. Edit `MODEL_SPECS` for other folders. |
| `models_dgm/PostDamagePostTech.py` | DGM-architecture version of the terminal post-damage/final-tech HJB. |
| `models_dgm/PostDamageIntermTech.py` | DGM version of the post-damage/intermediate-tech HJB. |
| `models_dgm/PostDamagePreTech.py` | DGM version of the post-damage/pre-tech HJB. |
| `models_dgm/PreDamagePostTech.py` | DGM version of the pre-damage/final-tech HJB. |
| `models_dgm/PreDamageIntermTech.py` | DGM version of the pre-damage/intermediate-tech HJB. |
| `models_dgm/PreDamagePreTech.py` | DGM version of the initial pre-damage/pre-tech HJB. |
| `models_dgm/feedforward_subnet.py` | Gated DGM layers, teacher distillation, sampling, optimizers, and validation helpers. |
| `models_dgm/params.py` | DGM-side copy of the baseline calibration. Keep it synchronized when changing model parameters. |
| `models_dgm/SimulationDeterministic.py` | Runs the shared deterministic workflow with DGM model classes first on `sys.path`. |

## Slurm Stage File Reference

Stage files execute one unit of work. Most require environment variables and
are intended to be called by a `submit_*.sh` launcher.

| File | Invocation and result |
|---|---|
| `sbatch/one_tech_jump_stage.sbatch` | Submit with `STAGE`, `FOLDERNAME`, `PRETRAINED_FOLDER`, `TECH_JUMP_PROBABILITY`, and LR variables; trains one selected MLP regime and rejects intermediate stages when `pi=1`. |
| `sbatch/half_rd_stage.sbatch` | Trains one R&D-active regime for an intensity experiment; used when post-tech regimes are copied unchanged. |
| `sbatch/half_rd_full_stage.sbatch` | Trains any of the six MLP regimes with separate post-tech/active LR settings and selectable `MODEL_DIR`. |
| `sbatch/sensitivity_stage.sbatch` | Trains one regime with run-specific `sigma`, `Gamma`, `theta`, or `psi0` environment overrides. |
| `sbatch/dgm_stage.sbatch` | Trains one gated-DGM regime, including optional MLP-teacher output distillation. |
| `sbatch/deterministic_stage.sbatch` | `MODE=simulate` runs deterministic paths; `MODE=plot` creates aggregate figures. Requires `EXPORT_FOLDER`. |
| `sbatch/dgm_deterministic_stage.sbatch` | Same deterministic interface for DGM checkpoints. |
| `sbatch/stochastic_jump_stage.sbatch` | One stochastic array task. Requires `EXPORT_FOLDER` and `XI`; writes one seed's path/event files. |
| `sbatch/stochastic_density_plot_stage.sbatch` | Plots rates, levels, value, and marginal-value densities and creates the compact `SimulationResultsPlot` tree. |
| `sbatch/svrd_decomposition_stage.sbatch` | Runs one model/`xi` SVRD decomposition. |
| `sbatch/network_audit_stage.sbatch` | Runs residual/network audit and training-history summary for one folder. |
| `sbatch/brown_capital_marginal_stage.sbatch` | Runs dirty-capital marginal-value/FOC plots for all six regimes in `EXPORT_FOLDER`. |

## Bash Submission File Reference

Unless marked otherwise, run a launcher from the repository root with
`bash <file>`. Review its model-folder arrays and LR grids before submission;
many launchers intentionally encode a specific completed experiment.

| File | What it submits / produces | Status |
|---|---|---|
| `submit/submit_pi1_inherit_posttech_fast.sh` | Recommended direct-final one-jump training, inherited post-tech checkpoints, deterministic paths, and plots for intensity 1 and 2. | Current |
| `submit/submit_stochastic_jump_simulations.sh` | Stochastic arrays for the standard one-jump, baseline two-stage, and doubled-intensity two-stage models. | Current |
| `submit/submit_inherited_pi1_stochastic_density.sh` | Stochastic arrays plus dependent density plots for the inherited direct-final one-jump model. | Current |
| `submit/submit_svrd_decomposition.sh` | SVRD jobs for one configurable model across `SVRD_XIS`. | Current |
| `submit/submit_svrd_control_density_models.sh` | SVRD jobs for the standard three comparison models. | Current |
| `submit/submit_two_stage_parameter_sensitivity_sweep.sh` | Four-LR sweeps for half capital volatility, half adjustment cost (`Gamma*2`, `theta/2`), and `psi0=0.05`, followed by deterministic analysis. | Current |
| `submit/submit_largebatch_retrain_stochastic_models.sh` | Large-batch continuation of the standard three stochastic models plus independent audits. | Current experiment |
| `submit/submit_largebatch_onejump_lr_sweep.sh` | Large-batch LR sweep for the direct-final one-jump model plus audits. | Current experiment |
| `submit/submit_dgm_two_stage_sweep.sh` | DGM width/layer/batch sweep over the six-regime dependency graph. | Experimental |
| `submit/submit_dgm_lr_sweep.sh` | DGM value/control LR sweep with teacher distillation settings. | Experimental |
| `submit/submit_deterministic_largebatch_dgm.sh` | Deterministic simulation/plot jobs for completed large-batch and DGM folders. | Experimental analysis |
| `submit/submit_additional_one_two_jump_lr_sweep.sh` | Historical broad LR and intensity sweep for one- and two-jump models, with deterministic first-jump plots. | Historical |
| `submit/submit_double_tech_intensity_lr_sweep.sh` | Doubled technology-intensity two-stage LR sweep; copies unchanged post-tech regimes. | Historical experiment |
| `submit/submit_half_rd_intensity.sh` | Initial half-technology-intensity training chain. | Historical experiment |
| `submit/submit_half_rd_lr_sweep.sh` | Full half-intensity LR sweep over all six regimes. | Historical experiment |
| `submit/submit_half_rd_lr_variants.sh` | Additional/continuation LR variants for half-intensity models. | Historical experiment |
| `submit/submit_deterministic_paths.sh` | Deterministic paths and plots for three hardcoded half-intensity folders. | Hardcoded analysis |
| `submit/submit_first_jump_density_diagnostics.sh` | First-jump deterministic density diagnostics for hardcoded trained folders. | Hardcoded analysis |
| `submit/submit_half_rd_deterministic_and_brown.sh` | Deterministic analysis and dirty-capital marginal-value plots for selected half-intensity models. | Hardcoded analysis |
| `submit/submit_one_tech_jump_pi0_lr_sweep.sh` | `pi=0` intermediate-only one-jump LR/intensity sweep. | Historical model |
| `submit/submit_pi1_direct_one_jump_and_y12_diagnostics.sh` | Full-retrained `pi=1` one-jump LR sweep plus deterministic `Y0=1.2` diagnostics. | Superseded by inherited workflow |
| `submit/submit_pi1_direct_deterministic_density.sh` | Deterministic first-jump plots for hardcoded direct-final one-jump folders. | Hardcoded analysis |
| `submit/submit_pi1_scale1_all_deterministic.sh` | Deterministic analysis for all matching `pi=1`, intensity-1 folders. | Hardcoded analysis |
| `submit/submit_one_jump_resimulations.sh` | Re-runs deterministic simulation/plot pairs for listed one-jump folders. | Hardcoded analysis |
| `submit/submit_stochastic_jump_scale2_remaining.sh` | Fills missing stochastic seeds for the doubled-intensity model; default array range is 9-100. | Repair/continuation |
| `submit/submit_logxi001_extension_sweep.sh` | Fine-tunes listed models on `logxi in [-5,5]`, then runs deterministic analysis including `xi=0.01`. | Historical fine-tuning |
| `submit/submit_logxi001_extension_low_lr_sweep.sh` | Two-block low-LR continuation for the `[-5,5]` extension. | Historical fine-tuning |
| `submit/submit_logxi001_extension_tiny_lr_sweep.sh` | Creates a temporary tiny-LR variant of the low-LR extension submitter and runs it. | Historical fine-tuning |
| `submit/submit_logxi_m4_three_model_sweep.sh` | Fine-tunes three selected models on `logxi in [-4,5]`, followed by deterministic paths at `xi>=0.02`. | Historical fine-tuning |
| `submit/submit_logxi_m4_rd05_sweep.sh` | `[-4,5]` continuation specifically for the half-intensity two-stage model. | Historical fine-tuning |
| `SimulationDeterministic.sh` | Generates and submits a deterministic job for one hardcoded baseline folder. | Legacy wrapper; prefer `sbatch/deterministic_stage.sbatch` |
| `SimulationDeterministicPlot.sh` | Generates and submits a plotting job for one hardcoded baseline folder. | Legacy wrapper |
| `SimulationRandom.sh` | Generates 100 hardcoded legacy `SimulationRandom.py` jobs for one overwritten `xi` value. | Legacy; do not use for new runs |
| `submit/submit_half_rd_fourier_resnet_distill.sh` | Fourier-ResNet half-intensity distillation experiment. | Not runnable: `models_fourier_resnet/` is absent |
| `submit/submit_half_rd_piratenet.sh` | PirateNet half-intensity architecture experiment. | Not runnable: `models_piratenet/` is absent |
| `submit/submit_half_rd_piratenet_biglr_warmup.sh` | PirateNet large-LR warmup sweep. | Not runnable: `models_piratenet/` is absent |
| `submit/submit_half_rd_piratenet_scheduler_sweep.sh` | PirateNet scheduler sweep. | Not runnable: `models_piratenet/` is absent |

## DGM Architecture

`models_dgm/` replaces the summed feedforward subnet with an initial state
embedding and repeated gated DGM layers. Its checkpoints are not shape-
compatible with the MLP checkpoints. Initialization therefore uses sampled
teacher-output distillation from an MLP folder before normal HJB/FOC training.

See [models_dgm/README.md](models_dgm/README.md). DGM remains experimental;
compare its residual audit and economically visited-state behavior against the
supported MLP before using its simulations.

## Cluster Environment

The Slurm files assume the Midway environment:

```bash
module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05
```

Runtime Python dependencies include TensorFlow, NumPy, SciPy, and Matplotlib.
Stochastic plotting also sets `PYTHONNOUSERSITE=1` to prevent incompatible
user-site NumPy/TensorFlow packages from shadowing the cluster module.

## Practical Failure Checks

- **`NaN` losses:** inspect the first nonfinite step, lower both effective
  learning rates, confirm consumption and `1+theta*i` remain positive, and
  verify the correct pretrained stage was loaded.
- **`DependencyNeverSatisfied`:** find the earliest failed parent job and read
  its `.err` file; cancelling only the downstream jobs does not repair the
  chain.
- **Missing intermediate folder:** correct for a direct-final `pi=1` one-jump
  model; incorrect for a two-stage model.
- **Wrong initial temperature:** check each stochastic `metadata_<seed>.txt`
  and deterministic job manifest. Current analysis should use `Y0=1.2`.
- **Unexpected one-jump/two-stage post-tech difference:** compare inherited
  checkpoint hashes and evaluate both networks on identical states.
- **Figures disagree with folder label:** read `run_manifest.txt` and stage
  `params.txt`; historical folder names are not always authoritative.
