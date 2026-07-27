# JAX spike — parity + speed verdict (2026-07-14)

Port of the full PostDamagePostTech train step (value + control Adam updates, HJB residual
with per-sample gradient/Hessian, monotonicity penalty, LHS-style resampling every step)
to JAX 0.4.13 (already in anaconda-2021.05 — no environment work needed).
Script: `port_postpost.py`. Jobs: 52052121 / 52053710 (caslake, 4 CPU).

## Parity — EXACT

Same trained RUNA weights, same 8192-point sample, same λ3/ξ inputs, float32:

| quantity | TF | JAX | rel. diff |
|---|---|---|---|
| HJB residual RMS | 2.569676e-03 | 2.569676e-03 | **9.1e-08** |
| FOC_d RMS | 1.5128e-04 | 1.5132e-04 | 3e-04 |
| FOC_g RMS | 1.6279e-04 | 1.6279e-04 | <1e-04 |

Two port facts that made this possible: (1) BatchNorm is a frozen affine in this codebase
(pde_rhs always calls nets with training=False; moving stats verified exactly 0/1), and
(2) the subnet is bn0 → 4×[dense→act→bn] → **sum of the four bn outputs** → final dense.

Pitfall log: an earlier "34% parity failure" was the harness passing sampled λ3 to JAX but
default λ3=1/6 to TF — matched inputs → exact match. An earlier "275 steps/s at every batch
size" was a closure-over-global-BATCH bug (batch size not in the jit signature → cache hit
→ every bench ran batch 128). Batch is now an explicit static argument.

## Speed (4-CPU caslake node, 300 timed steps, compile excluded)

| batch | JAX steps/s | TF steps/s (graph / +XLA-cluster) | JAX points/s | TF points/s |
|---|---|---|---|---|
| 128 (production) | **262.8** | 157.8 / 168.5 | 33.6k | 20.2k / 21.6k |
| 512 | 69.0 | — | 35.3k | — |
| 2048 | 17.8 | 26.9 / 27.0 | 36.5k | 55.1k |

Compile: 16–19 s per batch shape, one-time (irrelevant over 300k–1M steps).

- **Production regime (batch 128): JAX = +60%** over TF graph mode, +56% over TF+XLA.
  The whole two-net double-backward step fuses into one XLA executable; dispatch overhead
  (TF's per-op executor) is what dominated at this size.
- **Large batch: TF wins (1.5×).** JAX point-throughput saturates at ~35k pts/s at every
  batch size because the port computes per-sample 3×3 Hessians (`vmap`∘`jax.hessian` =
  forward-over-reverse per sample), which does not fuse into large GEMMs. TF's batched
  `tf.gradients` formulation does. A batched-Hessian JAX formulation (3 basis-direction
  HVPs over the whole batch) should recover this — v2 design note, not fixed in the spike.

## Verdict

Unchanged from the pre-spike recommendation, now with numbers: **v2 should be born in JAX**
(exact parity is achievable and was achieved; +60% at the production batch size out of the
box; jit∘grad∘grad native, keyed PRNG, vmap seed-ensembles; same XLA engine as TF). Not a
migration of the current production solver — the RCT and all in-flight comparisons stay on
the TF trainer.
