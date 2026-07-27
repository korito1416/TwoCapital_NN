# Economy Zoo — distinct solvable economies as NN warm-start maps

Goal (2026-07-18 directive): design MULTIPLE economies, each with the full (i_d, i_g, i_r)
decision complexity but DIFFERENT settings/interactions/couplings; solve each to a verified
fixed-form solution; PLOT the solutions; lift each into a warm-start MAP for the production
NN; train identically from every map; present that the delivered answers differ. Truth-seeking:
every economy fully derivable (Process → solution → map → chain), no inherited checkpoints.

## Pipeline (shared engineering, built 2026-07-18)

1. **Design** — multi-lens design panel (workflow `economy-zoo-design`): 8 candidate economies
   from 4 lenses (closed-form macro, homogeneity reduction, climate coupling, innovation theory),
   adversarially critiqued, synthesized into a 4–6 member portfolio. Output = implementation-ready
   Process + HJB + solution formulas + lift map per economy.
2. **Solve** (`solvers/<econ>.py`) — closed-form members: direct evaluation scripts with algebraic
   verification; FD members: ≤3-D reductions on the verified PIBYS pattern
   (`benchmarks/post_damage_post_tech/fd_pdpt_v5_stable.py`), Richardson + limit checks.
   Every solve writes an npz + PROVENANCE json.
3. **Plot** (`figures/`) — per economy: value + i_d/i_g/i_r policy fields/paths.
4. **Map** (`maps/<econ>.py`) — module with `MAP_NAME`, `PROVENANCE`, `fields(reg, lk,Z,Y,lr,l3,lx)`
   returning v/i_d/i_g/i_r targets at production states. Built into 4-regime checkpoints by
   `models_terminal_anchor/make_map_anchor.py` (ACTIVE i_r: net fit to −log(i_r); convention
   verified against models_warmstart training code).
5. **Train** — identical gentle chains (LR 10e-6,40e-6 warmup-cosine, 300k, MODEL_SEED=1,
   backward 4-regime DAG + sim) from every map: `submit/submit_zoo_chains.sh`
   (`ZOO_ARMS="..." bash submit/submit_zoo_chains.sh`), arms in `output_zoo_20260718/`.
6. **Readout** — Mike-format grid, one ROW per economy map: delivered 60-year paths
   (I_g/I_d/E/C-Y at ξ=0.05) + V(x₀) per regime + basin classification
   (decarbonizing / over-accumulation / other), appended below the existing figures.

## Known context the zoo builds on

- The bounded-box residual admits a multi-D solution family; training has an attractor;
  the INIT MAP selects the delivered basin (FD map → decarbonizing; refit-NN map → dirty).
- Naked maps cannot hold the VALUE LEVEL (collapses to ~4.8 attractor); the zoo measures
  the POLICY-BASIN selection power of economically-distinct maps.
- i_r net convention: output = −log(i_r). (A latent bug in make_warmstart analytic mode —
  it fit rates directly, making effective i_r≈0.99 — explains the RCT analytic arm failure.)
