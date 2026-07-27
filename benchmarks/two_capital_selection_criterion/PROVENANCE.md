# Full-chain provenance — the FD-anchored reproducible initialization program

Goal (2026-07-18): derive economically-sensible warm starts from *verifiable equations* —
never from an inherited checkpoint — and show that launching the same training from a
different, economically-coherent starting point delivers a different solution. Every link
below is on disk and re-derivable.

## Chain 0 — the verified FD ground truth (terminal regime, logK×Z×Y)

- Solver: PIBYS (policy iteration by simulation, exact-characteristic policy evaluation),
  `benchmarks/post_damage_post_tech/fd_pdpt_v5_stable.py`. Level uniqueness = δ-contraction
  of the forward discounted-flow integral (no interior boundary condition).
- Verification (all PASS, logs in scratchpad/fdlogs/):
  - Richardson grid convergence: v_logK 0.9505→0.9510, 2nd- & 4th-order sequences agree;
    nY direction flat (fully converged); vZ, i_d stable (`fd_v5_richardson.py`).
  - Analytic corner: Z→1 BGP v_logK=1.025 (theory 1.0), i_g=0.1247 (sympy 0.1258).
  - Mutual-consistency gate ~1e-8; interior residual RMS 1.6e-4.
  - Negative control: upwind ADI/implicit Howard smear the tiny de-invest costate
    (di_err_vZ≈122) — evidence FOR exact characteristics, documented not hidden.
- Five damage-curvature solutions (frozen V^ℓ):
  `outputs/fd_pdpt_v5_stable_lam3_{0000,0083,0167,0250,0333}_xi148.npz` (maxR ≈ 7e-4).

## Chain 1 — the calibration family (economically-coherent level movers)

`outputs_variants/fd_<tag>.npz` + `fd_<tag>_PROVENANCE.json` (method, grid, iters,
residual, overrides, full calibration). Readout at ref=(log 880, 0.7, 3.0), baseline
v=6.417, i_d=+0.0105, i_g=0.1636 — all variants remain decarbonizing (i_g ≫ i_d):

| knob            | Δv (level) | Δi_g (policy) | verdict |
|-----------------|-----------|----------------|---------|
| δ=0.008         | **+0.823**| +5.2%          | LEVEL knob |
| δ=0.009         | +0.360    | +2.5%          | level knob |
| δ=0.011         | −0.286    | −2.4%          | level knob |
| δ=0.0125        | **−0.615**| −6.0%          | LEVEL knob |
| A_g''=0.165/0.150 | ±0.25   | ±5% (entangled)| mixed |
| σ ×2 / ÷2       | ≤0.014    | ~0             | risk-only |
| η ±13%          | ≤0.009    | i_d ±50%       | policy-only |

⇒ the **discount-rate family** is the economically-honest source of "same decarbonizing
paths, different welfare level": a δ=0.008 planner values the same transition +0.82 higher.

## Chain 2 — FD → NN anchors (supervised fits)

`models_terminal_anchor/make_fd_anchor.py` → `output_fdanchor_20260718/<arm>/` with
`MANIFEST_FD_ANCHOR.json` (FD sources, variant, seed). Terminal nets fit the five-λ3 FD
family (λ3-interpolated, flat in logξ); non-terminal v broadcast the same value (one
consistent level); i_g/i_d fit the FD controls; i_r fresh (Stage 2).
Arms: `fdanchor_base` (δ=0.01), `fdanchor_dlt0080` (+0.82), `fdanchor_dlt0125` (−0.62)
— variant targets = variant field + (base^{λ3} − base^{1/6}) λ3-structure correction.

## Chain 3 — training chains from the derived warm starts

`submit/submit_fdanchor_chains.sh` → 300k screening, GENTLE control-LR peak 40e-6
(the reheat lesson: aggressive 40e-4 destroys even a correct-policy init — level-shift
probe wave-1, `output_levelshift_20260717/`), backward 4-regime DAG, per-arm MANIFEST.
KEY READOUT: does each chain (a) stay decarbonizing, (b) hold its OWN anchor level
(base vs ±δ-variants separating), (c) satisfy realized-welfare self-consistency?
(a)+(b)+(c) = "new economically-derived warm start ⇒ different delivered solution",
with every step above re-derivable from equations + configs.

## Companion probes

- Arbitrary level-shift probe (`output_levelshift_20260717/`, aggressive LR): level
  wandered down AND economics broke (I_d60 27–37, E60 17–20) — reheat preserves nothing.
- Gentle level-shift (`output_levelshift_gentle_20260718/`, 40e-6): the clean test of a
  pure welfare-level degree of freedom at fixed economics.
