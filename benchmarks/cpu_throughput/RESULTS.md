# CPU throughput benchmark — results & recommended defaults (2026-07-14)

Workload: real `train_step` of PostDamagePostTech (models_warmstart), caslake nodes,
anaconda-2021.05 TF, 300 timed steps after 50 warmup. Full matrix in job-outs/cpu_bench/.

## Findings

1. **Latency-bound confirmed**: 1 core ≈ 4 cores ≈ 8 cores at default threading
   (109 vs 108 vs 95 steps/s, batch 128). Intra-op parallelism is useless for 4×32 nets.
2. **inter_op = 2 is the one threading knob that pays**: +30–40% (2c 1:1 = 83 → 2c 1:2 = 138;
   4c best = 2:2 = 151). The train-step graph has independent branches (three nets' forwards,
   parallel gradient paths) that two executor threads exploit.
3. **Batch amortizes overhead ~3× in point-throughput**: at 4c 2:2, batch 128 → 19.3k pts/s,
   batch 2048 → 56.5k pts/s. (Changes the optimization recipe — future configs only.)
4. **XLA JIT unavailable on this stack**: `jit_compile=True` fails on the second-order-autodiff
   graph — `XlaDynamicUpdateSlice` has no registered gradient in this TF version (fixed in
   TF ≥ 2.12). torch on the cluster is 1.12 (no torch.compile). JIT is a v2-track item
   (JAX preferred; user-env TF 2.13 as a cheap test).

## Recommended sbatch defaults (apply AFTER the running RCT completes — do not
change the environment of an in-flight experiment)

| goal | request | env |
|---|---|---|
| fleet throughput (sweeps/RCTs) | `--cpus-per-task=2` | `OMP_NUM_THREADS=1`, TF `intra=1, inter=2` → 138 steps/s at 69/core |
| fastest single run | `--cpus-per-task=4` | TF `intra=2, inter=2` → 151 steps/s |
| never | `--cpus-per-task=8` | no configuration beats 4 cores |

TF threading must be set in code before the first op executes
(`tf.config.threading.set_{intra,inter}_op_parallelism_threads`) or via a small
sitecustomize; `OMP_NUM_THREADS` alone does not control the executor pools.

Memory: 8G is ample (21G requests slow backfill for nothing).

## XLA follow-up (compute-node A/B, 4c 2:2, 300 timed steps)

| config | batch 128 | batch 2048 |
|---|---|---|
| graph (baseline) | 157.8 steps/s | 26.9 |
| + `TF_XLA_FLAGS=--tf_xla_auto_jit=2` | **168.5 (+7%)** | 27.0 (+0%) |

Auto-clustering compiles only safe subgraphs → survives the second-order-autodiff graph that
strict `jit_compile=True` cannot (unregistered XlaDynamicUpdateSlice gradient), at the cost of
modest gains. Enabled in `sbatch/fast_stage.sbatch`.

Also verified by timing signature: `TF_NUM_INTRAOP_THREADS` / `TF_NUM_INTEROP_THREADS`
environment variables ARE honored by this TF build (env-forced 1:1 reproduces the slow-1:1
rate) — threading is controllable from sbatch alone, no code change.

## Final accelerated recipe (sbatch/fast_stage.sbatch)
`-c 2, --mem=8G, OMP=1, TF intra=1/inter=2 (env), TF_XLA_FLAGS=--tf_xla_auto_jit=2`
≈ 1.5× per-job speed × 2× jobs-per-core-budget ≈ **3× fleet throughput**, zero model-code change.
JIT beyond this (full fusion) requires TF≥2.12 / JAX — v2-track (see warmstart-rct-study notes).
