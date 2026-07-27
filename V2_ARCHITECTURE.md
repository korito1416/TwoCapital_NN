# v2 architecture — `hjx`: anchored regime-graph HJB solver in JAX

**Date:** 2026-07-14. **Status:** design, approved base + grafts from a 3-architect × 3-judge design panel
(lenses: identification-first / extensibility-first / performance-first; judges: PI-science /
maintainability / pragmatics). Identification-first design won 2 of 3 panels and tied the third.
**Evidence base:** `benchmarks/jax_spike/RESULTS.md` (exact TF parity, +60% at batch 128),
`benchmarks/cpu_throughput/RESULTS.md`, the confirmed level-anchor defect, the costate/recentering
level-pin measurements, and the warm-start RCT ("the warm start selects the solution").

## Why a v2 (three confirmed failure classes of the current stack)

1. **Identification, not accuracy.** The DGM residual pins the value *level* only through the δ=0.01
   discount anchor (∂res/∂level = −δ, measured exactly); the four/six regimes train backward as
   independently frozen nets with no level tying; the 1/ξ jump exponential amplifies the loose level
   into the ξ*≈0.025 wall. Fixes (anchored ansatz, joint training, cross-regime matching) do not fit
   the current one-file-per-regime architecture — every experiment so far forked a `models_*` tree.
2. **Extensibility.** The diffusion block is hand-expanded per regime file, duplicated across
   `models/`, `models_dgm/`, `models_precond/`, … Correlated innovation (non-diagonal σσ′: h* stops
   separating by channel, cross partials V_{logK,logR}, V_{Y,logR} enter) would require editing 6+
   files consistently. ŷ-as-state or a third tech stage — the same.
3. **Provenance/ops.** Implicit global seeds (three separate seed incidents), losses that are
   Monte-Carlo averages under undocumented per-run measures ν (the "trainings can't be compared" PI
   report), run identity living in comma/#/ξ directory names, RCT studies costing 40+ Slurm jobs.

## The design in one paragraph

One declarative **ModelSpec** (the whole 6-regime jump tree, the *full* shock-loading matrix σ(x)·Chol(P),
damage family, preferences — all data) feeds one **generic residual builder** (matrix-form robust HJB,
batched-HVP second-order terms). The value function passes through a composable **ansatz layer** whose
default removes the level null-space *by construction* (re-centered net + one shared anchor across
regimes). Training is a **LossProgram** of named, schedulable terms over a **RegimeGraph trainer** with
modes from `legacy_frozen_backward` (exact production reproduction) to `joint_anchored` (the
identification mode). All randomness is keyed, every sampling measure is hashed, every checkpoint is
self-describing. Substrate: JAX 0.4.13 already on the cluster, hand-rolled MLP+Adam core (spike-proven
to exact parity), zero new hard dependencies.

## Core abstractions (`hjx/` package, committed on `TwoTechJump`)

| module | contents |
|---|---|
| `spec/` | `climate_bbhhh.py` — the 6 regimes as `Regime` objects (states, pseudo-states, controls+bound activations, drift, **full σ(x) matrix**, flow utility, `Jump` edges with intensity/target/state-map); `CORR_CHOL` is the *single* correlation object. Damage family ships `(logN, dlogN/dY)` together so the h_y slope consistency is enforced at the type level. Typed params, one documented env-override loader, no import-time globals. |
| `nets/` | sum-skip `FeedForwardSubNet` clone (TF-parity verified), explicit init keys, activations incl. `inv_rate`. |
| `ansatz/` | composable wrappers that **stack**: `recentered` (V = N(x) − N(x*) + c*, one `SharedAnchor` across regimes = the FETI coarse space; measured ~24× level pin at zero cost), `structural_xi` (V = V_∞ − W/ξ, ~2×), `structural_logk`, `plain` (parity). Anchor registry stores x* (mechanically certified inside the simulated visited cloud — the Z=0.98 boundary trap) and c* (stable-FD reference cross-checked by Richardson + a simulated-discounted-payoff estimate). |
| `residual/` | generic augmented-HJB residual: h* = −(1/ξ)σ′∇V as a matrix product (the per-channel h_d/h_g/h_r/h_y formulas are never written down — they are the diagonal special case); tr(σ′Hσ) as **d whole-batch HVPs** (jvp-of-grad), never per-sample `jax.hessian` (the spike's large-batch regression, fixed structurally); jump distortions with a **gradient-preserving soft clamp** (`custom_jvp`: primal clamped ±60, tangent live — kills the clamp-zero-grad disease); FOCs autodiff-derived from the Hamiltonian with the closed-form path as a first-class override. |
| `losses/` | named, weighted, schedulable terms: `hjb_residual`, `foc_*`, `dvdY_monotonicity`, `constraint_guards` (original max-clamp semantics), `cross_regime_value_match` (at jump-entry slices, logN-corrected), `anchor_consistency`, `costate/ctrlfit` supervision, ξ-curriculum ramps. Loss-decomposition reports become a free byproduct. |
| `train/` | one graph trainer, modes: `legacy_frozen_backward` / `gauss_seidel_dag` (stability middle ground) / `joint_anchored` (identification mode — the only one where cross-regime matching has a live gradient path) / `howard_pia` (EGM program, optional flag) / `curriculum_xi`; hand-rolled Adam, per-group LR; **ensemble axis = vmap over (seed × scalar-arm)** — an RCT becomes one process per method, min-over-seeds gating built in. `distill` mode = unified warm-start factory (replaces `make_warmstart.py`) with shape-aware target validators encoding the RCT lesson that level-only initialization fails. |
| `sample/` | `SamplerSpec` with a ν-hash on every loss value; the eval harness **refuses (typed error)** loss comparisons across different ν, with reconstruction shims for the canonical historical TF runs; `init_seed` and `sample_seed` are separate config fields; frozen keyed common eval samples are versioned artifacts. |
| `eval/` | extends `benchmarks/solution_comparison` via a `--backend {tf,hjx}` flag (its conventions — Y≥ŷ entry slices, path-visited sampling — are hard-won; do not rebuild); identification probes (∂res/∂level, ε(ξ), ξ* estimator, frac-admissible min-over-seeds); FD gates for 3-state regimes; the **vector distortion field σh′** in outputs so the PI-mandatory distortion histograms survive ρ≠0; the **path simulator is generated from the same spec/σ** (same Cholesky) so trained model and simulation can never disagree on shock structure. |
| `compat/`, `io/` | productionized TF checkpoint extractor (from the spike); **single-file npz checkpoints embedding weights + full resolved config + spec/ν hashes + git sha + RNG state** — a checkpoint alone re-instantiates a run; manifest at launch; lineage registry ("which runs warm-start from X" answered mechanically). |
| `tests/` | frozen-sample TF-parity as a permanent merge gate; **sympy residual oracle** — symbolically expand the augmented robust HJB for random small specs *including dense ρ* and cross-check the generic builder numerically; rule: *no new spec primitive without an oracle test*; determinism test (same key ⇒ bit-identical step). |

## What correlated innovation becomes

`CORR_CHOL = cholesky(P)` in `spec/` — nothing else changes. The matrix-form h*, the robust drag
−(1/2ξ)∇V′σσ′∇V, and all cross second-derivatives fall out of the generic operator automatically.
Verification ladder: (i) `CORR_CHOL=I` reproduces the hand-expanded TF residual to float32 round-off
(regression gate forever); (ii) h* channel-separation test reproduces the four printed per-channel
formulas at diagonal σ; (iii) analytic ∂residual/∂ρ_ij direction check at ρ=0; (iv) FD gate on random
dense loadings. **Extensibility acceptance is falsifiable:** `variants/correlated_rho.py`,
`variants/yhat_state.py`, `variants/three_stage_tech.py` ship as spec-only files, and the gate is
*git diff touches only `spec/`*.

## Migration plan (phases gate on acceptance tests; TF `models/` stays frozen + canonical throughout)

| phase | deliverable | acceptance gate | effort |
|---|---|---|---|
| **P0** scaffold + compat | package skeleton; nets/optim from the spike; TF extractor for all 6 regimes; frozen common samples; **batched-HVP batch-2048 throughput proof** (moved here deliberately: prove ≥ TF's 27 steps/s before building on it) | parity < 1e-6 rel on PostDamagePostTech (spike: 9.1e-8); same-key bit-identical step | ~3 d |
| **P1** spec + generic residual | all 6 regimes as data; matrix operator; soft-clamped jumps; FOCs; sympy oracle | all 6 regimes' residual+FOC parity ≤ 1e-6 vs TF checkpoints; HVP ≡ per-sample Hessian at round-off | 5–6 d |
| **P2** trainer/samplers/manifests | `legacy_frozen_backward` mode reproducing the production 6-term loss + schedules; ν-hash; manifests; npz checkpoints; harness `--backend hjx` | terminal-regime 300k retrain ≥ production accuracy *on the common sample*; ≥250 steps/s @128 on 2 cores | 5–6 d |
| **P3** identification modes (the point) | recentered ansatz + SharedAnchor registry; `joint_anchored`, `gauss_seidel`; cross-regime matching; ξ-curriculum; (seed×arm) ensembles; probes | ε(ξ=0.01) < 0.003; frac-admissible ≥ 0.95 min-over-seeds at ξ=0.025; residual within 20% of baseline; anchor certified in visited cloud. **Pre-registered as a research outcome, not an engineering deliverable — P0–P2 value stands on its own.** | 7–8 d |
| **P4** correlated-innovation pilot | ρ-sweep configs (e.g. ρ_{Wr,Wg} innovation–green comovement); trained pilot + writeup | sweep launches with **zero edits outside `spec/`**; FD < 1e-5 on random Chol; ρ=0 arm ≡ P3 baseline | 3–4 d |
| **P5** certification vs TF canonical | full 6-regime v2 (legacy AND joint modes) vs TF via the extended harness; PI-format report; `howard_pia` behind a flag | legacy-mode v2 within run-to-run noise of TF on every eval metric at ξ≥0.05 → *only then* does canonical flip | 6–8 d |

Total ≈ 30–35 person-days, part-time compatible: every phase leaves a working, tested, useful artifact
(P0 alone productionizes TF-checkpoint loading + fast eval; P2 alone gives comparable losses + real
provenance), so the migration cannot strand a half-finished rewrite.

## Risks (with mitigations)

- **Joint-training instability is real** (Howard late-divergence, S1 bimodal seed failure) →
  `gauss_seidel` middle ground; min-over-seeds gating; P3 pre-registered as research.
- **Anchor correctness ≠ anchor existence** (a wrong c* propagates everywhere) → triple-sourced c*
  (FD + Richardson + simulated payoff), certification test, anchors versioned like checkpoints.
- **Parity erosion under refactors** → frozen-sample parity is a permanent merge gate; sympy oracle
  for anything new.
- **JAX 0.4.13 pinned to the cluster module** → hand-rolled core minimizes API surface; fallback
  `pip --user` newer jax (CPU wheels) is cheap.
- **Deep-ξ float32 overflow is intrinsic** (exponent > 88 ⇔ mismatch > 88ξ) → soft clamp + one-flag
  float64 for claims below ξ≈0.01.
- **EGM/Howard scope creep** → optional P5 flag; the mesh-free inner solve remains a bet to be proven.

## What this is *not*

Not a migration of the production TF solver: PI deliverables (warm-start RCT extensions, reports)
continue on TF untouched until the P5 certification gate. v2 is a parallel package that must *earn*
canonical status by reproducing production within noise, then beating it on the identification probes.
