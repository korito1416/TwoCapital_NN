"""Unified driver for the FD-anchor program: solve the terminal Post-Damage Post-Tech
HJB by a chosen method and (a) GRADE it against the stable-FD ground truth to VERIFY
independence, and/or (b) SAVE it as a frozen V^l for the damage-jump amplifier.

  --method pibys    : fd_pdpt_v5_stable.solve_stable  (the primary; the stored reference)
  --method adi2     : howard_adi2.solve  (independent monotone-upwind ADI Howard)
  --method implicit : howard_implicit.solve (independent implicit Howard)

  --lam3 <val>      : damage curvature (0, 1/12, 1/6, 1/4, 1/3)
  --grade           : print stable_fd_eval.grade metrics (costates + de-invest sign)
  --save <path>     : np.savez the {logK,Z,Y,v,i_d,i_g,c,vlK,vZ,vY} solution
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FDDIR = os.path.join(HERE, "..", "post_damage_post_tech")
sys.path.insert(0, FDDIR)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pibys", "adi2", "implicit"])
    ap.add_argument("--lam3", type=float, default=1/6.0)
    ap.add_argument("--xi", type=float, default=148.4)
    ap.add_argument("--nK", type=int, default=31)
    ap.add_argument("--nZ", type=int, default=61)
    ap.add_argument("--nY", type=int, default=31)
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--save", default=None)
    ap.add_argument("--iters", type=int, default=0, help="Howard/PIBYS iteration budget (0=method default)")
    ap.add_argument("--dtau", type=float, default=2.0, help="adi2/implicit false-transient step")
    ap.add_argument("--omega", type=float, default=1.0, help="adi2 Howard damping (<1 = under-relax)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="calibration override on fd_pdpt_v5.P (in place), e.g. --set delta=0.008")
    a = ap.parse_args()

    if a.set:
        import fd_pdpt_v5 as FDP     # P is shared by reference across all solver modules
        for kv in a.set:
            k, v = kv.split("=")
            assert k in FDP.P, f"unknown calibration key {k}; known: {list(FDP.P)}"
            old = FDP.P[k]; FDP.P[k] = float(v)
            print(f"calibration override: {k} = {float(v)} (baseline {old})", flush=True)

    if a.method == "pibys":
        import fd_pdpt_v5_stable as S
        kw = dict(lam3=a.lam3, xi=a.xi, nK=a.nK, nZ=a.nZ, nY=a.nY, verbose=True)
        if a.iters: kw["howard_max"] = a.iters
        out = S.solve_stable(**kw)
    elif a.method == "adi2":
        import howard_adi2 as H
        out = H.solve(lam3=a.lam3, xi=a.xi, nK=a.nK, nZ=a.nZ, nY=a.nY,
                      dtau=a.dtau, howard_iters=(a.iters or 600), qfloor=0.02, omega=a.omega)
    else:
        import howard_implicit as H   # implicit Howard has no false-transient dtau knob
        out = H.solve(lam3=a.lam3, xi=a.xi, nK=a.nK, nZ=a.nZ, nY=a.nY,
                      howard_iters=(a.iters or 200), qfloor=0.02)

    print(f"\n[{a.method} lam3={a.lam3:.4f}] iters={out.get('iters')} "
          f"time={out.get('time',0):.0f}s maxR={out.get('max_abs_residual'):.2e}", flush=True)

    if a.grade:
        from stable_fd_eval import grade, load_stable_fd
        m = grade(out, load_stable_fd())   # vs stable-FD ground truth (lam3=1/6)
        print(f"\n=== GRADE {a.method} vs stable-FD ground truth ===", flush=True)
        for k, v in m.items():
            print(f"    {k:18s}: {v:+.4e}", flush=True)
        print("VERDICT-KEYS: box_err_i_d (agreement), di_err_vZ (costate), "
              "di_min_i_d & di_frac_neg (de-invest SIGN)", flush=True)
        with open(os.path.join(HERE, f"grade_{a.method}_lam3_{a.lam3:.4f}.json"), "w") as f:
            json.dump(m, f, indent=1)

    if a.save:
        keys = ["logK", "Z", "Y", "v", "i_d", "i_g", "c", "vlK", "vZ", "vY"]
        np.savez(a.save, **{k: np.asarray(out[k]) for k in keys})
        # provenance sidecar: the full calibration + scheme, so every solution is traceable
        import fd_pdpt_v5 as FDP
        prov = dict(method=a.method, lam3=a.lam3, xi=a.xi, grid=[a.nK, a.nZ, a.nY],
                    iters=int(out.get("iters", -1)),
                    max_abs_residual=float(out.get("max_abs_residual", float("nan"))),
                    overrides=a.set, calibration={k: float(v) for k, v in FDP.P.items()})
        with open(a.save.replace(".npz", "_PROVENANCE.json"), "w") as f:
            json.dump(prov, f, indent=1)
        print(f"saved -> {a.save} (+ provenance json)", flush=True)

if __name__ == "__main__":
    main()
