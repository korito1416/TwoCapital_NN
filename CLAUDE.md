# CLAUDE.md — live workbench (TwoStageTechJump_FOCIr_orignal)

Claude Code project instructions. **Migrated from Codex's `AGENTS.md` (2026-06-28).** Codex auto-loaded
`AGENTS.md`; Claude Code auto-loads this file. The full math spec is imported below.

## Model specification (the HJB math) — imported
@AGENTS.md
@PAPER_HJB_REFERENCE.md

## What this repo is
The **live canonical** neural solver for *"A Deep Learning Analysis of Climate Change, Innovation, and
Uncertainty"* (Barnett, Brock, Hansen, Hu, Huang). Deep-Galerkin / policy-improvement NN solver of a
**6-regime** two-capital climate HJB system (PreDamage/PostDamage × Pre/Interm/PostTech). git branch
`TwoTechJump`, repo `korito1416/TwoCapital_NN`.

## Where things are
- `models/` — the 6 HJB regime solvers + `params.py`, `feedforward_subnet.py`, `pretrained_paths.py`.
  Solver variants: `models_precond/` (÷preconditioner), `models_dgm/` (DGM subnet), `models_v2/`
  (schedule_v2 dual-optimizer), `models_v3/` (closed-form/Newton controls). **Current work:**
  `torch_egm/`, `models_torch/` (PyTorch EGM/Howard port) + `benchmarks/two_capital_deterministic/`.
- **Trained weights:** per-regime TF checkpoints at `output*/<run>/<Regime>/{v_nn,i_g_nn,i_d_nn,i_I_nn}_checkpoint_<Regime>.{index,data}`.
  Stable warm-start bases: `pretrained/{two_stage_tech_base (6-regime), nber_legacy (4-regime)}`.
  Warm-start: `--pretrained_path <ckpt>` (CamelCase regimes).
- **Figures:** `Simulation.ipynb` simulates *and* plots the main paths (RD/E/I_d/I_g);
  `SimulationDeterministicPlot.py` makes the distortion PNGs; `Loss_3D_plot_xi*.ipynb` reload
  checkpoints for the appendix surfaces.
- **Organized map of the whole project** (version history, by-feature models, plotting catalog,
  checkpoint index): `/project/lhansen/TwoCapital_Project/` — see its `README.md`, `MODELS/`,
  `plotting/`, `CHECKPOINTS.md`, `LINEAGE.md`, `MODEL_VERSION_TREE.md`.

## Conventions / gotchas
- **Quote every path.** Run/output dirs contain commas, `#`, and Greek (ξ), e.g.
  `"output_001/TwoStageTech_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_...".`
- Source is tiny; `output*/`, `job-outs/`, checkpoints are huge and gitignored — never read/copy wholesale.
- Pseudo-states: `λ3` (damage curvature, L=5) and `logξ` — **single ξ** (it's duplicated 2–3× only to
  match checkpoint input width 6/7/8; it is NOT dual-ξ).
- Two-stage jump: `A_g → A_g'(catch-up=A_d) → A_g''(breakthrough)`, `π=0.04`. **OneJump** = runtime
  `π=1`; **TwoJump2** = `tech_jump_intensity_scale=2.0`. These are runtime knobs, not code forks.
- Runtime overrides via env (`MODEL_*`, `π`, `tech_jump_intensity_scale`) per `params.py`
  `_ENVIRONMENT_OVERRIDES`; reproducibility via `MODEL_SEED`.
- Run DAG: `submit/` / `parallel_handle.sbatch` (afterok-chained, post→pre regimes).
- Training loss: `<regime>/training_history.csv` col `loss_v` (~10⁻³ = paper-grade).
- **Warning:** many low-loss runs had their weights deleted — a low `loss_v` does NOT mean checkpoints
  still exist; verify before warm-starting (`MODELS/<v>/checkpoint` points to runs that retain weights).

## Output organization (registry-backed, 2026-07-06)
- **`RUNS.md`** (repo root) = canonical registry: which run makes which figure, the warm-start
  **dependency spine** (bases with in-degree — NEVER delete), keep/archive/deleted status. Consult it
  before deleting ANY `output*/` run (in-degree > 0 ⇒ a base others warm-start from).
- **New runs:** `output_<study>_<YYYYMMDD>/`; drop a `MANIFEST.txt` (config + warm-start base + purpose +
  date) in each run so provenance isn't only in the (comma/#/ξ-laden) dir name.
- **Plotting investment vs a Haoyang/paper reference:** consult memory `haoyang-training-viz-reference`
  FIRST — green/dirty = LEVELS `I_g.txt`/`I_d.txt` (NOT `GreenInvestment`/`DirtyInvestment` = I/Y ratios);
  R&D = `RD.txt` (=I_r/Output, NOT `i_r.txt`=I_r/K). The level≈ratio-at-t=0 trap caused 3 false alarms.

## Current work (UNCOMMITTED — commit it)
The PyTorch EGM/Howard alternative solver (`torch_egm/`, `models_torch/`) and the numerical-methods A/B
benchmark suite (`benchmarks/two_capital_deterministic/`) exist only in the working tree (git branch
`TwoTechJump`). Commit them so they have a real home + history.
