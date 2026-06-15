# DGM HJB Models

This directory reuses the six existing HJB and policy-improvement equations
with a separate Deep Galerkin Method subnet. Each subnet contains:

1. An initial dense state embedding.
2. Repeated gated DGM layers driven by the original state and hidden state.
3. The regime-specific output transformation already used by the baseline.

The gated equations follow the architecture in:

- Sirignano and Spiliopoulos, *DGM: A deep learning algorithm for solving
  partial differential equations*.
- Al-Aradi et al., `alialaradi/DeepGalerkinMethod`.

These checkpoints are intentionally incompatible with the baseline MLP
checkpoints. The training script therefore uses sampled-output distillation
from a baseline checkpoint as initialization, followed by ordinary HJB/FOC
training. Train the regimes backward within one `output_dgm_001` folder.
