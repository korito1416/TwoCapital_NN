# Ladder launch decision (2026-07-19)

Coarse-validation status at launch: V1 PASS both economies (7.7e-7 / 2.2e-7, gate 1e-4).
JONES V2 substantive PASS (s-top exact-step gap 0.0032 vs left-rect 34.5; monotonicity
violations 0.013% at 4e-6 magnitude — boolean gate over-strict, overridden WITH numbers).
PHYSRISK V2 amber: high-lambda3 s-top boundary layer at the COARSE grid (residual/rate
up to 0.17 on the top slice only) — expected to shrink at the design grid; FINAL acceptance
moves to the real ladder cells' own gates (finiteness, gate_int, residual, s-top gap, all
written to per-cell provenance). V3 both economies = pure Howard-budget shortfall
(h_clip_bind_frac = 0.0 → mechanism healthy); ladders launched at HOWARD=150.
Maps are gated on the LADDER provenance, not the coarse probes. JONES grid raised to the
design (61,41,61) per audit comparability flag; walltimes raised accordingly (12/40/60h).
