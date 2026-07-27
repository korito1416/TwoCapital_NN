# solution_comparison — harness for comparing trained solutions across runs

Consolidated 2026-07-13 from the evaluation tooling built for the July-2026 two-trainings
reports (`benchmarks/report_lowxi_structural_defect/`, `benchmarks/report_uniqueness/`).
Designed to compare ANY set of runs (next use: the warm-start RCT), not just two.

## Tools

| script | what it does |
|---|---|
| `solution_loader.py` | shared loader: per-jump-state model reconstruction from checkpoints (float32/float64 auto-cast), value-net-only loader, path/LHS state samplers, per-term RMS evaluation |
| `eval_path.py` | every objective term, per jump state × ξ, **averaged (RMS) along the simulated 60-year path** → .npy |
| `eval_lhs.py` | same on one **shared Latin-hypercube evaluation sample** (post-damage states on Y∈[2.5,4]) |
| `plot_cross_state.py` | CS1 (HJB error by jump state, 2×2) + CS2 (FOC errors, 3×4) from an eval .npy |
| `plot_four_states.py` | V and marginal values (V_logK, V_Z, V_logR) by jump state along the path, row-shared scales |
| `plot_value_slices.py` | V one state variable at a time, others at the initial state |
| `plot_loss_composition.py` | training-loss composition (percent + levels) per jump state from training_history.csv |

All scripts take repeated `--run "label=/abs/run/root"`; run roots must contain
`<Regime>/{v,i_g,i_d[,i_r]}_nn_checkpoint_<Regime>` for the four jump states.

## Conventions (hard-learned — do not silently change)

1. **Post-damage jump states live at Y ≥ ŷ = 2.5 only.** They are entered at the damage
   threshold (`models/PreDamagePreTech.py` L282 evaluates the continuation at `y_upper`),
   and temperature never falls. Evaluate them at the entry slice Y=2.5 (path scripts) or on
   Y∈[2.5,4] (LHS script). Evaluating them at pre-damage path temperatures (Y≈1.2) produces
   economically meaningless disagreement — "inside the training box ≠ economically visited".
2. **λ3 pseudo-state fixed at 1/6** (grid midpoint) unless a study varies it.
3. **Checkpoints may be float64** (wide-ξ runs); the loader casts to float32. Consequence:
   at very deep ξ (≤0.005) the jump term can overflow in re-evaluation even when the run's
   own training (float64) was fine — report such NaNs as an evaluation limitation.
4. **Training losses are comparable across runs ONLY if the sampled ranges (measure ν)
   are identical.** Each training loss is a Monte-Carlo average under its own sampling
   distribution; `eval_path.py` / `eval_lhs.py` exist precisely to put all runs on one
   common evaluation sample.
5. Figure grammar: no text on figures (axis labels + legend only), red/blue first two
   run colors, large fonts; captions carry the message.

## Reproducing the July-2026 report figures

```
RUNA="<repo>/output_001/OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
RUNB="<repo>/output_converge_20260707/OneTechJump_..._logximin_m5p30_..._num_iterations2000000_fromBase1e6"
python eval_path.py --run "training 1=$RUNA" --run "training 2=$RUNB" \
       --paths-from "$RUNB" --path-xis 0.050,148.600 --out cs_path.npy
python plot_cross_state.py --data cs_path.npy --xmin 0.05
python plot_four_states.py --run "training 1=$RUNA" --run "training 2=$RUNB" \
       --paths-from "$RUNB" --xi 148.6 --path-xi-dir 148.600 --out four_states_xi148.png
```

Environment: `module load python/anaconda-2021.05` (prod TF).
