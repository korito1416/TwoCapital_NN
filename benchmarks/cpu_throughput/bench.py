"""CPU throughput benchmark for the DGM-PIA train step.

Measures train_step/s of PostDamagePostTech (terminal jump state, no downstream
nets — the lightest, most reproducible stage) under a given threading/batch
config. Threading must be fixed BEFORE TensorFlow executes any op, hence one
process per config; the sbatch wrapper loops configs.

Env in:  BENCH_INTRA, BENCH_INTER, BENCH_BATCH, BENCH_STEPS (timed), BENCH_WARMUP
Prints:  one CSV line  cpus,intra,inter,batch,steps_per_s
"""
import os, sys, time

INTRA = int(os.environ.get("BENCH_INTRA", "2"))
INTER = int(os.environ.get("BENCH_INTER", "2"))
BATCH = int(os.environ.get("BENCH_BATCH", "128"))
STEPS = int(os.environ.get("BENCH_STEPS", "300"))
WARM = int(os.environ.get("BENCH_WARMUP", "50"))
CPUS = os.environ.get("SLURM_CPUS_PER_TASK", "?")

import tensorflow as tf  # noqa: E402
if os.environ.get("BENCH_SKIP_THREAD_API", "0") != "1":
    tf.config.threading.set_intra_op_parallelism_threads(INTRA)
    tf.config.threading.set_inter_op_parallelism_threads(INTER)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "models_warmstart"))
from PostDamagePostTech import PostDamagePostTechModel   # noqa: E402
from params import PARAMS, investment_rate_activation    # noqa: E402

def cfg(act, fin, nm):
    return {"num_hiddens": [32] * 4, "use_bias": True, "activation": act,
            "dim": 1, "nn_name": nm, "final_activation": fin}

p = PARAMS.copy()
p.update({
    "π": 1.0, "tensorboard": False, "batch_size": BATCH,
    "learning_rates": [1e-5, 4e-4], "learning_rate_schedule_type": "warmup_cosine",
    "num_iterations": STEPS, "logging_frequency": 10**9, "gradient_clip_norm": 1.0,
    "export_folder": None, "verbose": False,
    "v_nn_config": cfg("swish", "softplus", "v_nn"),
    "i_g_nn_config": cfg("tanh", investment_rate_activation(PARAMS["θ_g"]), "i_g_nn"),
    "i_d_nn_config": cfg("tanh", investment_rate_activation(PARAMS["θ_d"]), "i_d_nn"),
    "i_r_nn_config": cfg("softplus", "softplus", "i_r_nn"),
})
p["optimizers"] = [tf.keras.optimizers.Adam(1e-5), tf.keras.optimizers.Adam(4e-4)]

m = PostDamagePostTechModel(p)
for nm, dim in [("v_nn", 7), ("i_g_nn", 7), ("i_d_nn", 7)]:
    getattr(m, nm)(tf.zeros([1, dim]))

JIT = os.environ.get("BENCH_JIT", "0") == "1"
if JIT:
    import functools
    raw = type(m).train_step.python_function          # undecorated method
    step = tf.function(functools.partial(raw, m), jit_compile=True)
else:
    step = m.train_step

tag = "jit" if JIT else "graph"
try:
    for _ in range(WARM):
        step()
    t0 = time.time()
    for _ in range(STEPS):
        step()
    dt = time.time() - t0
    print(f"RESULT,{CPUS},{INTRA},{INTER},{BATCH},{tag},{STEPS/dt:.1f}")
except Exception as e:
    print(f"RESULT,{CPUS},{INTRA},{INTER},{BATCH},{tag},COMPILE_FAIL:{type(e).__name__}")
