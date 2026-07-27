# Phase 1 Design — GAN/adversarial methods on the robust climate-HJB solver

Testbed: 3-D post-damage-post-tech benchmark (logK, Z, Y), xi=148.4, lam3=0.167.
Ground truth: `outputs/fd_pdpt_v5_stable_lam3_0167_xi148.npz` (stable FD; true costate vZ).
Grader: `stable_fd_eval.grade` -> `box_err_i_d/i_g`, and in the de-invest corner (Z>=0.90, Y>=3.5):
`di_err_vZ` (TRUE costate error), `di_min_i_d`, `di_match_depth`, `di_err_i_d`.

## The structural fact that decides everything
1. The robust HJB inner min over (h, g) is CLOSED-FORM (Hansen-Sargent relative-entropy):
   h* = -(1/xi) sigma' V_x ;  g*^l = exp(-(1/xi)(V^l - V)).  A learned adversary can at best
   re-approximate an EXACT analytic tilt -> redundant for the worst case itself.
2. The PROVEN failure mode is the vZ UNDER-IDENTIFICATION: the strong HJB residual is FLAT in vZ
   wherever the Z-transport drift a_Z -> 0 (de-invest / high-Z corner). ENGD drove residual to 1e-11
   yet true costate error floored at ~0.1. So NO pure-residual method breaks the floor; only INJECTING
   costate information does (scratch qd-direct/ctrlfit reached di~1.6e-3 by injecting the costate).

## Litmus test for each formulation
Does the adversary inject information that is NOT a function of the strong residual (which
under-identifies vZ in the corner)? If yes -> can break the floor. If it only re-weights/re-learns a
residual- or closed-form-derived quantity -> redundant.

---

## Formulation (a): Saddle-point robust solver (value net vs adversary nets for h, g)
min-max:  min_{V-net}  max_{h-net, g-net}  (1/M) sum [ -delta V + L^{a} V + f
              + V_x.(sigma h) + (xi/2)|h|^2 + sum_l J^l ( g^l (V^l-V) + xi(1-g^l+g^l log g^l) ) ]^2 ... (planner min residual)
with the adversary maximizing the bracketed penalized Hamiltonian.
ASSESSMENT: REDUNDANT vs closed form. The inner argmax IS h*=-(1/xi)sigma'V_x, g*=exp(...). The learned
adversary converges (slower, noisier) to the analytic tilt. Critically, h* itself is a FUNCTION of the
under-identified vZ (h_g, h_d carry V_Z), so the learned adversary INHERITS the same flatness -- it
cannot supply information the residual lacks. Does NOT target vZ under-identification. Likely adds
optimization noise. -> redundant, do not prototype.

## Formulation (b1): Adversarial collocation / RAR on the RESIDUAL
min-max:  min_{V,ctrl} max_{sampler G}  E_{x~G} [ strong-HJB-residual(x) ]^2,  G concentrates mass where
residual is largest.
ASSESSMENT: REDUNDANT for the floor. Residual is FLAT in vZ in the corner; high-residual points are NOT
where vZ is wrong. This re-allocates samples by the very signal proven insufficient (ENGD already nuked
residual everywhere AND vZ stayed wrong). Useful only for generic residual hot-spots, not the costate
floor. Does NOT target vZ under-identification. -> redundant for THIS defect.

## Formulation (c): Adversarial COSTATE-ATTACK (FOC-consistency adversary)  ***PICK***
Use a discrepancy that pins vZ WITHOUT dividing by a_Z, i.e. NOT the strong residual. The planner's own
FOCs (no FD labels needed) couple vZ to the controls:
   delta/(C/K) = phi_d'(i_d) [ v_logK - Z v_Z ]          (FOC-d)
   delta/(C/K) = phi_g'(i_g) [ v_logK + (1-Z) v_Z ]      (FOC-g)
Eliminating delta/(C/K) gives a relation that determines v_Z from (i_d, i_g, v_logK) that is
INDEPENDENT of the Z-drift a_Z -> exactly the channel the strong residual lacks in the corner. Define the
per-point FOC-coupling discrepancy
   D(x) = phi_d'(i_d)[v_logK - Z v_Z] - phi_g'(i_g)[v_logK + (1-Z) v_Z]
(=0 when controls and costate are mutually consistent). The min-max:
   min_{V-net, ctrl-net}  max_{adversary A=(sampler G, weighting)}  E_{x~G}[ w(x) D(x)^2 ]
                          + lambda * E_uniform[ strong-HJB-residual^2 ]   (anchor term)
The adversary HUNTS the (region, direction) where the network's vZ most violates FOC-coupling and forces
the solver to fix it there. This INJECTS costate information self-consistently (from the planner's own
FOC structure), which is the supervision-equivalent the scratch winners used to break the floor -- but
WITHOUT FD labels. This is the ONLY formulation that structurally targets the vZ under-identification.

Architecture (prototype):
  - value net V(logK,Z,Y) swish, 4x32; controls i_d,i_g via softplus-bounded heads (reuse incumbent).
  - adversary = lightweight sampler over the box that up-weights points by current D(x)^2 (a learned
    proposal OR a cheap importance-reweighting / top-k hard-mining surrogate). Direction-attack variant:
    adversary also picks the sign/region in (Z,Y) maximizing D^2, emphasizing the de-invest corner where
    a_Z->0 and the strong residual is blind.
  - alternate: (i) planner step min over V,ctrl of [strong-resid (uniform) + beta*FOC-disc (adversarial)];
    (ii) adversary step max the FOC-disc weighting / proposal.

## Ranking
1. (c) FOC-consistency costate-attack  -- the ONLY one that injects non-residual (vZ) information. PICK.
2. (b1) adversarial-RAR-on-residual    -- redundant for the floor (re-weights the insufficient signal).
3. (a) saddle-point robust solver      -- redundant vs the exact closed-form tilt; inherits vZ flatness.

## Validation protocol (Phase 2)
Prototype (c) vs a UNIFORM-sampling baseline (same nets, strong residual only, no FOC adversary), short
training, 3 seeds. MEASURE both with stable_fd_eval.grade. A "win" REQUIRES di_err_vZ DOWN and the mild
de-invest depth (di_min_i_d ~ -0.025, di_match_depth down) CAPTURED -- vs the FD true costate, NOT vs the
residual (softplus lesson: validate the true quantity, not a gameable loss). If (c) does not move
di_err_vZ below the ~0.81 incumbent floor, the honest verdict is: adversarial methods are redundant here
because the worst case is closed-form and residual-based adversaries cannot see the under-identified vZ;
only a non-residual costate-information channel (FOC-consistency or FD supervision) can.
