"""
Jump-distortion g = exp(-(1/xi)(V^l - V)) diagnostic for PreDamagePreTech (pi=1).

Mike's question: the HJB residual stays ~1e-3 at all xi (see per_xi_residual.json), but is the
solution actually usable at low xi?  The failure channel is the JUMP DISTORTION g, not the residual:
a value-LEVEL error ~ (residual/delta) gets amplified by 1/xi inside the exponent of g.

This script recomputes, on the SAME box batch, the exponent  e = -(1/xi)(V^l - V)  for the
damage jump (worst offender) and the tech-breakthrough jump, and reports:
  - RMS / max |e|  (how far the exponent ranges)
  - g stats (median, p95, max)  -> g swings orders of magnitude at low xi
  - fraction of points where the production clamp [-700, 350] would bite.
It re-derives V^l - V exactly as models_float64/PreDamagePreTech.pde_rhs does.
"""
import os, sys, json
import numpy as np, tensorflow as tf
tf.keras.backend.set_floatx("float64")

REPO   = "/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
OUT    = os.path.join(REPO, "benchmarks/report_one_jump_lowxi_stable/mike_diagnostics")
sys.path.insert(0, OUT)
import eval_per_xi_residual as E   # reuse the faithful model builder + samplers

BATCH = 4096
SEED  = 20260701

def main():
    m = E.build_model("PreDamagePreTech")
    p = m.params
    A_g_pp = p["A_g_prime_prime"]; y_upper = p["y_upper"]; L = p["L"]; l3v = p["λ3_values"]
    rng = np.random.default_rng(SEED)
    logK, Z, Y, logR, _l3 = E.sample_box(rng, BATCH)

    results = {}
    for label, xi in E.XI_GRID:
        logxi = tf.constant(np.full((BATCH,1), np.log(xi)), dtype=tf.float64)
        X = tf.concat([logK, Z, Y, logR, logxi, logxi, logxi], 1)
        v = m.v_nn(X)
        # tech-breakthrough partner value (6-input net): [logK, Z, Y, A_g'', logxi, logxi]
        v_tech = m.v_PreDamagePostTech_nn(tf.concat(
            [logK, Z, Y, A_g_pp*tf.ones(tf.shape(Y),dtype=Y.dtype), logxi, logxi], 1))
        e_tech = (-1.0/np.exp(np.log(xi))) * (v_tech - v)     # exponent for g^{l''}
        # damage partner value (8-input net), evaluated per lambda3 realization
        e_dmg_all = []
        for l in range(L):
            v_dmg = m.v_PostDamagePreTech_nn(tf.concat(
                [logK, Z, tf.ones(tf.shape(Y),dtype=Y.dtype)*y_upper, logR,
                 l3v[l]*tf.ones(tf.shape(Y),dtype=Y.dtype), logxi, logxi, logxi], 1))
            e_dmg_all.append((-1.0/xi)*(v_dmg - v))
        e_dmg = tf.concat(e_dmg_all, 0)

        def stats(e, tag):
            e = e.numpy().ravel()
            g = np.exp(np.clip(e, -700, 350))
            return {
                f"{tag}_exp_rms": float(np.sqrt(np.mean(e**2))),
                f"{tag}_exp_absmax": float(np.max(np.abs(e))),
                f"{tag}_exp_p95": float(np.percentile(e, 95)),
                f"{tag}_g_median": float(np.median(g)),
                f"{tag}_g_p95": float(np.percentile(g, 95)),
                f"{tag}_g_max": float(np.max(g)),
                f"{tag}_clampbite_frac": float(np.mean((e < -700) | (e > 350))),
            }
        row = {"xi": xi}
        row.update(stats(e_tech, "tech"))
        row.update(stats(e_dmg,  "dmg"))
        # dV level spread proxy: how much V^l - V varies (independent of 1/xi amplification)
        dV_dmg = (e_dmg.numpy().ravel()) * (-xi)   # = V^l - V
        row["dmg_dV_rms"] = float(np.sqrt(np.mean(dV_dmg**2)))
        row["dmg_dV_absmax"] = float(np.max(np.abs(dV_dmg)))
        results[label] = row
        print(f"xi={xi:<9.4g} dmg: exp_rms={row['dmg_exp_rms']:.3e} exp_absmax={row['dmg_exp_absmax']:.3e} "
              f"g_max={row['dmg_g_max']:.3e} clampbite={row['dmg_clampbite_frac']:.4f} "
              f"| dV_rms={row['dmg_dV_rms']:.3e}", flush=True)
    with open(os.path.join(OUT, "per_xi_g_distortion.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", os.path.join(OUT, "per_xi_g_distortion.json"))

if __name__ == "__main__":
    main()
