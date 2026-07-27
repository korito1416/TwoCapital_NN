# Paper Equation Reference (authoritative)

Extracted verbatim-in-substance from Barnett–Brock–Hansen–Hu–Huang,
*A Deep Learning Analysis of Climate Change, Innovation, and Uncertainty* (2025 NBER tex).
Use together with `AGENTS.md`. Where the paper notes typos, implement value+slope continuity.

## States / controls
States: `logK=log(Kd+Kg)`, `Z=Kg/(Kd+Kg)`, `Y`, `logR`. Pseudo-states: `ξ` (logξ), `ℓ` (λ3).
Controls: `i^d, i^g, i^r`. Distortions: `h` (Brownian drift), `g^ℓ,g^{ℓ'},g^{ℓ''}` (jump).

## Damage function N(y) (curvature jump at ŷ; earlier jump ⇒ more severe)
- Pre-damage (0≤y≤ŷ):   `log N(y) = λ1 y + (λ2/2) y²`  ⇒ `(logN)_y = λ1 + λ2 y`.
- Post-damage (y≥ŷ):  `log N(y) = λ1 y + (λ2/2)ŷ² + (λ2/2)(y−ŷ+ȳ)² + (λ3(ℓ)/2)(y−ŷ)² − (λ2/2)λ2(ȳ)²`
  (paper flags the matching constants as typo-prone). The y-DERIVATIVE that matters is
  `(logN)_y = λ1 + λ2 (y−ŷ+ȳ) + λ3(ℓ)(y−ŷ)`  — it **contains the λ3 curvature term**.
  ⇒ Any term that uses `(logN)_y` (e.g. the climate drift distortion `h_y`, the `V_y` cash-flow)
  MUST use the regime-correct slope: pre-damage uses `λ1+λ2 y`; post-damage must include the λ3 piece.

## Value transform (why `(logN)_y` appears explicitly)
The solver parameterizes a transformed value `v` (≈ `V` shifted by `±log N`); the paper's commented
diffusion HJB shows the climate term as `( v_y − {λ1+λ2 Y} )(θ̄+ς·h)E_t + ½ λ2 ς² E_t²`, i.e.
the *true* `V_y` used inside `h_y` is `v_y − (logN)_y`. **`h_y` and the `V_y` cash-flow term must use the
same `(logN)_y` as the regime's damage function** (consistency check, independent of the typos).

## Climate / emissions
`E_t = η A_d (1−Z) K`. `dY = E_t(θ̄ dt + ς dW^y)`. Climate drift in HJB: `V_y(θ̄+ς·h)E_t + ½ ς² E_t² V_yy`.

## Capital drifts (φ) and FOCs
`φ_j(i) = α_j + Γ_j log(1+θ_j i)`, `φ_j'(i)=Γ_j θ_j/(1+θ_j i)`.
FOCs (MU of consumption = marginal benefit), with `C/K=(A_d−i^d)(1−Z)+(A_g−i^g)Z−i^r`:
- `δ/(C/K) = φ_d'(i^d)[V_logK − Z V_Z]`
- `δ/(C/K) = φ_g'(i^g)[V_logK + (1−Z) V_Z]`
- `δ/(C/K) = ψ_r'(i^r) V_logR`   (R&D-active regimes only)

## Knowledge stock / logR drift  (CRITICAL for σ_κ sign)
`dR = −ζR dt + ψ0 (I^r)^{ψ1} R^{1−ψ1} dt + R σ_r dW^r`, `i^r=I^r/(K^d+K^g)`.
Itô ⇒ logR drift = `−ζ + ψ0 (i^r)^{ψ1} exp(ψ1(logK−logR)) − ½ σ_r²`.
Paper's HJB carries `(ψ_r(i^r) − ½ σ_r²) V_logR + (σ_r²/2) V_{logR,logR}` with
`ψ_r(i^r) := −ζ + ψ0(i^r)^{ψ1} exp(ψ1(logK−logR))`.
⇒ **logR drift in simulation must be `ψ_r − ½σ_r²` (the Itô term is `−½σ_r²`, NOT `+½σ_r²`).**

## Diffusion block (pre-damage pre-tech Full HJB, substitute regime's V)
```
[(1−Z)φ_d + Z φ_g − (σ_d²(1−Z)²+σ_g²Z²)/2] V_logK + (σ_d²(1−Z)²+σ_g²Z²)/2 V_{logK,logK}
+ [φ_g − Zσ_g² − φ_d + (1−Z)σ_d²] Z(1−Z) V_Z + ½ Z²(1−Z)²(σ_g²+σ_d²) V_ZZ
+ [−Z(1−Z)²σ_d² + Z²(1−Z)σ_g²] V_{logK,Z}
+ V_y(θ̄+ς·h)E_t + ½ ς²E_t² V_yy
+ (ψ_r(i^r) − ½σ_r² + σ_r·h) V_logR + (σ_r²/2) V_{logR,logR} + ξ|h|²/2
```
Brownian distortion FOCs (independent shocks):
`h_d=−(1/ξ){V_logK−V_Z Z}(1−Z)σ_d`, `h_g=−(1/ξ){V_logK+V_Z(1−Z)}Z σ_g`,
`h_r=−(1/ξ)V_logR σ_r`, `h_y=−(1/ξ)V_y E_t ς`  (with the regime-correct `V_y`, see transform note).

## Jumps and jump distortion (CRITICAL for exp-overflow)
Damage: `J_n^ℓ(y)=(1/L)J_n(y)`, `J_n(y)=r1(exp((r2/2)(y−y̲)²)−1)·1_{y≥y̲}`.
Tech: `J_g^{ℓ'}(R)=(1−π)R`, `J_g^{ℓ''}(R)=π R` (R in arrival units = exp(logR)/varrho · scale).
Jump contribution `Σ_ℓ J^ℓ g^ℓ (V^ℓ−V) + ξ Σ_ℓ J^ℓ(1−g^ℓ+g^ℓ log g^ℓ)` with closed form
`g^{ℓ*} = exp(−(1/ξ)(V^ℓ − V))`.  ⇒ For small ξ and `V^ℓ<V` this exponent blows up; numerically
`exp(...)` overflows float32 (>~88). Paper math is exact; the *numerical* guard (clip) is an
implementation concern, not a spec change.

## Tech-jump tree (π = breakthrough prob | a tech jump)
pre-tech → with prob π to ℓ'' (breakthrough, absorbing), prob (1−π) to ℓ' (intermediate);
intermediate ℓ' → ℓ'' w.p. 1. `A_g(ℓ')=A_d`, `A_g(ℓ'')=λ^n A_d`. Baseline `π=0.04`.

## Key parameters
`δ=0.01`; `(α,Γ,θ,σ)_d=(α,Γ,θ,σ)_g=(−0.035,0.060,16.7,0.01)`;
`A_d=0.1303`, `{A_g,A_g',A_g''}={0.1085,0.1303,0.1567}`; `(ζ,ψ0,ψ1,σ_κ)=(0,0.10583,0.5,0.0078)`;
`varrho=746.67`; `θ̄=1.86/1000`; `η=0.291`; `ς=1.2·1.86/1000`;
`(λ1,λ2)=(0.00017675, 2·0.0022)`; `λ3(ℓ)=(1/3)(ℓ−1)/(L−1)`, L=5; `(r1,r2)=(1.5,0.36)`; `(y̲,ȳ)=(1.5,2.5)`.
NN: 4 hidden layers width 32; value swish, controls tanh/softplus, custom bounded output for i_d/i_g.

## Net-of-R&D consumption
`C = A_d K^d − i^d K^d + A_g K^g − i^g K^g − i^r(K^g+K^d)`;  in (logK,Z): `C/K=(A_d−i^d)(1−Z)+(A_g−i^g)Z−i^r`.
The `−i^r` term is present only in R&D-active (pre-breakthrough) regimes.
