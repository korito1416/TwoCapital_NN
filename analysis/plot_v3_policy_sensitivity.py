"""
v3 POLICY-FUNCTION sensitivity plots (closed-form-control models_v3).

Loads a trained (or fresh-init, for tooling validation) value network v_nn for a
models_v3 regime and plots the SEMI-ANALYTIC controls (i_d, i_g, and i_r for the
pre-tech regimes) over 1-D state sweeps.

The controls are NOT re-derived here: we instantiate the regime's OWN model class
(so its InputNormalization + architecture match the trained net exactly) and call
its OWN `pde_rhs`, then read the controls straight out of pde_rhs's return tuple:
    i_g = (inside_log_i_g - 1) / theta_g
    i_d = (inside_log_i_d - 1) / theta_d
    i_r = c0 - c                       (resource constraint; pre-tech regimes only)
so the plotted controls are byte-identical to what training used (same q_d/q_g, same
c closed form, same i_r bisection).

USAGE (once a real checkpoint exists):
    cd /project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal
    module load python/anaconda-2021.05
    MODEL_GAMMA_D=0.12 MODEL_GAMMA_G=0.12 MODEL_THETA_D=8.35 MODEL_THETA_G=8.35 \
    python plot_v3_policy_sensitivity.py <REGIME> <RUN_FOLDER>
where RUN_FOLDER is the parent dir that contains <REGIME>/v_nn_checkpoint_<REGIME>,
e.g. output_v3/onejump_adjcost/<run>/  (the script appends <REGIME>/...).

The half-adjcost env vars MUST be set so the closed-form controls use the right
theta/Gamma (params.py applies them at import).

Set V3_POLICY_FRESH_INIT=1 to skip loading the checkpoint (fresh random weights) --
used only to validate the tooling end-to-end before a checkpoint exists.
"""

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_v3"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from params import PARAMS  # noqa: E402
from feedforward_subnet import regime_input_bounds  # noqa: E402

# ---- regime registry -------------------------------------------------------
_REGIMES = {
    "PostDamagePostTech": ("PostDamagePostTech", "PostDamagePostTechModel", 7, False),
    "PreDamagePostTech":  ("PreDamagePostTech",  "PreDamagePostTechModel",  6, False),
    "PostDamagePreTech":  ("PostDamagePreTech",  "PostDamagePreTechModel",  8, True),
    "PreDamagePreTech":   ("PreDamagePreTech",   "PreDamagePreTechModel",   7, True),
}

# Neighbor v_nn checkpoints each regime's __init__ tries to load. Controls do NOT
# depend on them (only the HJB jump terms do), so for POLICY plots we point any
# missing neighbor at a throwaway checkpoint built to the right input width.
_NEIGHBORS = {
    "PostDamagePostTech": [],
    "PreDamagePostTech":  [("v_PostDamagePostTech_nn_path", "PostDamagePostTech", 7)],
    "PostDamagePreTech":  [("v_PostDamagePostTech_nn_path", "PostDamagePostTech", 7),
                            ("v_PostDamageIntermTech_nn_path", "PostDamageIntermTech", 8)],
    "PreDamagePreTech":   [("v_PreDamagePostTech_nn_path", "PreDamagePostTech", 6),
                            ("v_PreDamageIntermTech_nn_path", "PreDamageIntermTech", 7),
                            ("v_PostDamagePreTech_nn_path", "PostDamagePreTech", 8)],
}


def _make_v_nn_config(regime, num_neurons=32, num_hidden_layers=4, seed=0):
    cfg = {
        "num_hiddens": [num_neurons] * num_hidden_layers,
        "use_bias": True,
        "activation": "swish",
        "dim": 1,
        "nn_name": "v_nn",
        "final_activation": "softplus",
        "input_bounds": regime_input_bounds(regime),
        "seed": seed,
    }
    return cfg


def _ensure_neighbor_checkpoints(regime, run_folder, params, dummy_dir):
    """For every neighbor net this regime loads, make sure a checkpoint exists.

    Prefer the real one under run_folder/<stage>/...; if absent, build a dummy
    v_nn of the right width and save a throwaway checkpoint (neighbor nets feed
    only the HJB jump terms, never the controls, so dummies are fine for policy
    plots). Returns the params dict with the *_nn_path keys filled in.
    """
    from feedforward_subnet import FeedForwardSubNet
    os.makedirs(dummy_dir, exist_ok=True)
    for key, stage, width in _NEIGHBORS[regime]:
        real = os.path.join(run_folder, stage, f"v_nn_checkpoint_{stage}")
        if os.path.exists(real + ".index"):
            params[key] = real
            continue
        # build + save a dummy checkpoint at the right width
        ckpt = os.path.join(dummy_dir, f"v_nn_checkpoint_{stage}")
        if not os.path.exists(ckpt + ".index"):
            cfg = _make_v_nn_config(stage)
            net = FeedForwardSubNet(cfg)
            net.build((None, width))
            net.save_weights(ckpt)
        params[key] = ckpt
    return params


def build_model(regime, run_folder, fresh_init=False):
    """Instantiate the regime model, load v_nn (unless fresh_init)."""
    import importlib
    mod_name, cls_name, n_inputs, is_pretech = _REGIMES[regime]
    mod = importlib.import_module(mod_name)
    Model = getattr(mod, cls_name)

    params = {
        "batch_size": 256,
        "learning_rates": [1e-4, 4e-3],
        "v_nn_config": _make_v_nn_config(regime),
        "num_iterations": 1,
        "logging_frequency": 1,
        "pretrained_path": None,
        "learning_rate_schedule_type": "warmup_cosine",
        "tensorboard": False,
        "train_from_scratch": True,
        "phase": "base",
        # pi=1.0 => intermediate-tech neighbor nets are skipped in __init__
        "π": 1.0,
        "export_folder": os.path.join(run_folder, regime),
    }
    dummy_dir = os.path.join(run_folder, regime, "_policy_dummy_neighbors")
    params = _ensure_neighbor_checkpoints(regime, run_folder, params, dummy_dir)

    model = Model(params)
    model.v_nn.build((None, n_inputs))
    ckpt = os.path.join(run_folder, regime, f"v_nn_checkpoint_{regime}")
    if fresh_init:
        print(f"[fresh-init] NOT loading checkpoint (random weights) for {regime}.")
    elif os.path.exists(ckpt + ".index"):
        model.v_nn.load_weights(ckpt)
        print(f"Loaded v_nn checkpoint: {ckpt}")
    else:
        raise FileNotFoundError(
            f"No v_nn checkpoint at {ckpt}.index . Set V3_POLICY_FRESH_INIT=1 to "
            f"validate the tooling on random weights."
        )
    return model, is_pretech


def controls_from_pde_rhs(model, logK, Z, Y, logR, lam3, logxi, is_pretech):
    """Call the regime's OWN pde_rhs and read controls out of its return tuple.

    Inputs are 1-D numpy arrays (same length). Returns dict of numpy control arrays.
    """
    def col(a):
        return tf.constant(np.asarray(a, np.float32).reshape(-1, 1))
    out = model.pde_rhs(col(logK), col(Z), col(Y), col(logR), col(lam3), col(logxi))
    # out layout (both): rhs, pv, dv_dY, c, inside_log_i_g, inside_log_i_d, FOC_d, FOC_g, [FOC_r, dv_dlogR,] precond_w
    c = out[3].numpy().ravel()
    inside_g = out[4].numpy().ravel()
    inside_d = out[5].numpy().ravel()
    theta_d = float(model.params["θ_d"]); theta_g = float(model.params["θ_g"])
    i_g = (inside_g - 1.0) / theta_g
    i_d = (inside_d - 1.0) / theta_d
    res = {"i_d": i_d, "i_g": i_g, "c": c}
    if is_pretech:
        # i_r recovered from the resource constraint c = (A_d-i_d)(1-Z)+(A_g-i_g)Z - i_r
        A_d = float(model.params["A_d"]); A_g = float(model.params["A_g"])
        Zv = np.asarray(Z, np.float32).ravel()
        c0 = (A_d - i_d) * (1.0 - Zv) + (A_g - i_g) * Zv
        res["i_r"] = c0 - c
        res["FOC_r"] = out[8].numpy().ravel()
        res["dv_dlogR"] = out[9].numpy().ravel()
    return res


# ---- sweep configuration ---------------------------------------------------
# central (held-fixed) values for the non-swept states
_CENTRAL = {
    "logK": 5.5,
    "Z": 0.5,
    "Y": 1.2,
    "logR": 3.5,
    "λ3": float(PARAMS.get("λ3_max", 1.0 / 3.0)) / 2.0,   # mid of [0, 1/3]
}
# logxi values to overlay (xi = 0.05 robust, xi = 148.6 ~ near risk-neutral)
_XI_OVERLAY = [0.05, 148.6]

# per-axis sweep ranges (the two most important are Y and Z)
_SWEEPS = {
    "Y":    ("Y",    np.linspace(0.0, 4.0, 121)),
    "Z":    ("Z",    np.linspace(0.05, 0.95, 121)),
    "logK": ("logK", np.linspace(4.0, 7.0, 121)),
    "logR": ("logR", np.linspace(1.0, 6.0, 121)),
}


def _state_grid(axis_name, axis_vals, logxi_val):
    """Build the 6 state columns for a sweep over `axis_name`, others central."""
    n = len(axis_vals)
    cols = {}
    for s in ("logK", "Z", "Y", "logR", "λ3"):
        cols[s] = np.full(n, _CENTRAL[s], np.float32)
    cols[axis_name if axis_name != "λ3" else "λ3"] = np.asarray(axis_vals, np.float32)
    logxi = np.full(n, np.log(logxi_val), np.float32)
    return cols["logK"], cols["Z"], cols["Y"], cols["logR"], cols["λ3"], logxi


def make_plots(regime, run_folder, fresh_init=False):
    model, is_pretech = build_model(regime, run_folder, fresh_init=fresh_init)
    out_dir = os.path.join(run_folder, regime, "policy_sensitivity")
    os.makedirs(out_dir, exist_ok=True)

    control_names = ["i_d", "i_g"] + (["i_r"] if is_pretech else [])
    x_axis_labels = {"Y": r"$Y$ (temperature)", "Z": r"$Z$ (green share)",
                     "logK": r"$\log K$", "logR": r"$\log R$"}

    summary = {}
    for axis_name, (state_key, axis_vals) in _SWEEPS.items():
        for ctrl in control_names:
            plt.figure()
            for xi in _XI_OVERLAY:
                gk = _state_grid(state_key, axis_vals, xi)
                res = controls_from_pde_rhs(model, *gk, is_pretech)
                yv = res[ctrl]
                plt.plot(axis_vals, yv, label=fr"$\xi={xi:g}$")
                summary[f"{axis_name}:{ctrl}:xi={xi:g}"] = (
                    float(np.nanmin(yv)), float(np.nanmax(yv)),
                    bool(np.isnan(yv).any() or np.isinf(yv).any()))
            plt.xlabel(x_axis_labels.get(axis_name, axis_name))
            plt.ylabel(ctrl)
            plt.title(f"{regime}: {ctrl} vs {axis_name}"
                      + ("  [FRESH-INIT random weights]" if fresh_init else ""))
            plt.legend()
            plt.grid(True, alpha=0.3)
            fname = os.path.join(out_dir, f"{ctrl}_vs_{axis_name}.png")
            plt.savefig(fname, dpi=110)
            plt.close()

    # also dump a small text summary of ranges
    with open(os.path.join(out_dir, "control_ranges.txt"), "w") as f:
        f.write(f"regime={regime} fresh_init={fresh_init}\n")
        f.write(f"theta_d={model.params['θ_d']} theta_g={model.params['θ_g']} "
                f"Gamma_d={model.params['Γ_d']} Gamma_g={model.params['Γ_g']}\n")
        for k in sorted(summary):
            lo, hi, bad = summary[k]
            f.write(f"{k}: min={lo:.5g} max={hi:.5g} nonfinite={bad}\n")
    print(f"Saved policy-sensitivity PNGs + control_ranges.txt to {out_dir}")
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python plot_v3_policy_sensitivity.py <REGIME> <RUN_FOLDER>")
        print("  REGIME in", list(_REGIMES))
        sys.exit(2)
    regime = sys.argv[1]
    run_folder = sys.argv[2]
    if regime not in _REGIMES:
        raise SystemExit(f"unknown regime {regime}; choose from {list(_REGIMES)}")
    fresh = os.environ.get("V3_POLICY_FRESH_INIT", "0").lower() in {"1", "true", "yes", "on"}
    summ = make_plots(regime, run_folder, fresh_init=fresh)
    # print a compact range report to stdout
    print("=== control range summary (min, max, nonfinite) ===")
    for k in sorted(summ):
        lo, hi, bad = summ[k]
        print(f"  {k}: min={lo:.5g} max={hi:.5g} nonfinite={bad}")
