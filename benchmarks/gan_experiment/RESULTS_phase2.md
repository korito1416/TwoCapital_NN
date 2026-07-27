# Phase 2 Results — adversarial FOC-consistency costate-attack vs uniform baseline

Testbed: 3-D post-damage-post-tech HJB (logK,Z,Y), xi=148.4, lam3=1/6.
Solver: self-contained torch value net (swish 4x32) + SEPARATE control nets i_d,i_g (tanh-bounded,
de-invest representable down to -1/theta). Physics byte-faithful to fd_pdpt_v5 (P, _drift, FOC controls,
damage horizon Y_CAP, closed-form robustness drag O(1e-6)). Trained on strong HJB residual + FOC residual
+ FOC-consistency D^2. Script: `gan_costate_attack_3d.py`. Raw: `gan_results_full.npz`, log `run_full.log`.

Two matched-compute arms (3 seeds, 8000 steps, bs 2048; both converge to HJB-RMS ~4-6e-5, FOC ~3e-8):
- UNIFORM : uniform collocation only.
- ADV     : 50% of each batch adversarially hard-mined on D^2 over a de-invest-corner pool, with the
            FOC-consistency term up-weighted 3x on the mined points (the costate-attack).

GRADED on TRUE FD costate via stable_fd_eval.grade (NOT on residual/loss). Median [min,max] over 3 seeds:

| metric (de-invest corner Z>=0.9,Y>=3.5) | UNIFORM | ADV | INCUMBENT ctrlfit | FD truth |
|---|---|---|---|---|
| di_err_vZ (DECISIVE costate error) | 0.881 [0.866,0.885] | 0.868 [0.860,0.884] | **0.114** | 0 |
| di_min_i_d (de-invest depth) | +0.0139 | +0.0223 | -0.0206 | **-0.0248** |
| di_frac_neg (does it de-invest?) | 0.000 | 0.000 | 0.183 | **0.167** |
| di_match_depth | 0.039 | 0.047 | 0.0042 | 0 |
| box_err_i_d | 0.144 | 0.148 | 0.038 | 0 |
| box_err_i_g | 0.295 | 0.275 | — | 0 |

## Verdict: REDUNDANT. Adversarial does NOT help the costate.

1. di_err_vZ: ADV improves it by 1.5% (0.881 -> 0.868), within seed spread ([0.860,0.884] overlaps
   [0.866,0.885]) -> NOISE, not signal. Both arms sit at ~0.87, i.e. the UNDER-IDENTIFIED floor; neither
   approaches the incumbent's 0.114.
2. De-investment is NOT captured by EITHER arm: di_frac_neg = 0 for all 6 runs (FD 0.167), and di_min_i_d
   is POSITIVE (+0.014 / +0.022) where the FD goes to -0.0248. The adversary even pushes i_d MORE positive
   (di_min +0.014 -> +0.022, di_match_depth 0.039 -> 0.047), i.e. AWAY from the FD de-invest sign.
3. Both pure-collocation arms are ~8x worse on di_err_vZ and ~3-4x worse on box_err_i_d than the ctrlfit
   incumbent (which injects the costate by FD-anchored Howard control-fitting + grids).

## Why (tie-back to the structural facts)
- The robust inner min (h,g) is CLOSED FORM, so there is no worst-case for an adversary to learn that the
  exact tilt does not already give. The drag is O(1e-6) here anyway.
- The intended non-residual channel — the FOC-consistency discrepancy D = phi_d'(i_d)qd - phi_g'(i_g)qg —
  is driven to ~1e-8 by the SHARED residual+FOC training in BOTH arms (CONS ~3e-9 in the log). So D
  carries NO information beyond the residual once the control nets are FOC-optimal: the two FOCs are
  satisfied by co-adapting (i_d,i_g) to whatever vZ the residual leaves, rather than pinning vZ. The
  adversary hard-mines a discrepancy that is already ~0 -> nothing to inject. The over-determination that
  would pin vZ only bites if the controls are an INDEPENDENT, ALREADY-CORRECT reference (i.e. supervision /
  a converged Howard control field) — which is exactly the FD-anchored information the incumbent uses and
  the label-free adversary lacks.
- The strong residual remains FLAT in vZ where a_Z->0 (the corner), so residual + a self-consistent FOC
  term cannot see vZ. Confirmed empirically: HJB-RMS reaches 4e-5 while di_err_vZ stays 0.87.

CONCLUSION: On this benchmark adversarial/GAN methods are REDUNDANT for the v_Z under-identification.
The worst case is closed-form (a learned adversary re-approximates an exact tilt), and a costate-attack on
a self-consistent FOC discrepancy collapses to the residual it is paired with — it cannot break the costate
floor. Only a NON-self-referential costate channel (FD supervision, or a converged Howard control field
used as a fixed target) breaks it, as the ctrlfit incumbent does (di_err_vZ 0.114, de-invest captured).
