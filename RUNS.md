# RUNS.md — canonical run registry & warm-start dependency spine

Built 2026-07-06 from the `pretrained_path` graph over every trained run (params.txt). Purpose: make
"which run reproduces which figure" and "which run is a load-bearing base" instantly findable, and stop
accidental deletion of depended-on checkpoints. Companion: memory `haoyang-training-viz-reference`.
Paths contain commas, `#`, Greek `ξ` — always quote.

## 1. Which run produces which deliverable

| Deliverable | Run (under repo root) | Notes |
|---|---|---|
| **Haoyang `OneJump1-1.pdf`** (π=1 OneJump figure) | `output_001/OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000` | float32 (`models/`), LR **40e-4**, logξ∈[−3,5]. Plot LEVELS `I_g/I_d`, ratio `RD`. |
| **Low-ξ worst-case report** (`report_mike_defense`) | `output_lowxi_float64/OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostFull_logximin_m5p30_LR_warmup_cosine_10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations50000` (+ `…AdjustmentCostHalf…`) | float64 low-ξ, 50k. Feeds `benchmarks/report_one_jump_lowxi_stable/`. |
| **Converged retrain** (cross-regime verdict) | `output_warmcontinue_001/OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostFull_logximin_m5p30_LR_warmup_cosine_10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations500000_fromBase1e6` | float64, warm-started from the 40e-5 1e6 base; loss_v ~1.9–3.7e-3. Pending: low-ξ ḡ re-check. |

## 2. Warm-start dependency SPINE — NEVER delete (in-degree = # runs that warm-start from it)

| in-deg | Base run (all in `output_001/`, 1e6 iters) |
|---|---|
| **23** | `TwoStageTech_LR_warmup_cosine_10e-6,10e-4_…` — root base of nearly everything |
| 14 | `TwoStageTech_TechIntensityScale_2p0_LR_warmup_cosine_10e-6,40e-4_…` |
| 11 | `TwoStageTech_LR_warmup_cosine_10e-6,40e-4_…` |
| 11 | `TwoStageTech_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_…` |
| 5 | `OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_…` ← Haoyang figure base |
| 2 | `OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-5_…` ← retrain base |
| 1 | `TwoStageTech_LR_warmup_cosine_40e-5,40e-5,40e-5,40e-5_…` |
| 1 ea | low-ξ chain `output_lowxi_001/…_logximin_{m3p51,m3p91,m4p61}_…50000` (feeds the 50k + report) |

## 3. Keep / archive (not depended-on, but valuable)

- `output/` — 8 × **2M-iter TwoStageTech** runs = paper-grade (main 2-jump model). **Archive, keep.**
- `output_sensitivity_001/` — active adjustment-cost sensitivity study.
- `output_delta025_ab/`, `ab_delta_override/` — δ=0.25 A/B (recent).
- `output_dgm_001/`, `output_v2/`, `output_v3/` — method variants (tied to `models_dgm/`, `models_v2/`, `models_v3/`).
- `output_converge_20260707/…_logximin_m5p30_…num_iterations2000000_fromBase1e6` — **"training 2"** of the
  July-2026 two-trainings comparison sent to the PIs (solution_identification / solution_uniqueness reports).
  Full 4-regime chain, wide-ξ [−5.3,5], LR 40e-5, 2M, float64, warm-started from the Haoyang 1M base.
  **KEEP** — referenced by sent reports; MANIFEST.txt inside. (Registered 2026-07-13.)
- `output_seed_control_20260709/` — 16 single-regime (PreDamagePreTech) arms of the seed/perturbation study:
  `seed1/2` (MODEL_SEED-only pair, NULL result), `lr40em5` (LR-knob arm), `perturbA–F` + `perturb_ctrl`
  (perturbed-warm-start, noise 0.1→heals / 0.3→NaN), `gentle1–4` (noise 0.15 + low-LR re-settle),
  `runbrep_s1/2` (RUNB-replica seed pair with drift logs). Per-arm MANIFEST.txt written 2026-07-13
  (params.txt does NOT record MODEL_SEED — seeds reconstructed from the launch sbatch headers).
  **KEEP** — evidence base for the identification reports.

## 4. Deleted in the 2026-07-06 cleanup (recorded for traceability)

`output_001/*_lowLR_*|*_tinyLR_*|*_sweepLR_*` (dead LR sweeps), `output_largebatch_001/`,
`output_001/` non-depended LR-variant leaves (kept spine + Haoyang-figure + retrain-base),
`output_hardened_scratch/`, `output_precision/`, `output_ablation/`, stale `job-outs/*LargeBatch*`.

## 5. Go-forward convention (see CLAUDE.md "Output organization")

New run → `output_<study>_<YYYYMMDD>/<config-named-run>/`; drop a `MANIFEST.txt` in each run
(config + warm-start base + purpose + date). Before deleting ANY run, check §2 here (in-degree > 0 = base).
