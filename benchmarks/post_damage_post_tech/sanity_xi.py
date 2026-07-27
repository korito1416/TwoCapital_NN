"""
Sanity check: WHY is the xi (robustness) effect so small? Decompose, on the validated
network, the worst-case distortions h and the robustness drag -1/(2xi) sum E^2, and compare
the drag to the HJB scale (flow, delta*v). Confirms the O(sigma^2/xi) story.
"""
import numpy as np
import plot_pretrained_climate as C
from params import PARAMS

sd, sg, vars_ = PARAMS["σ_d"], PARAMS["σ_g"], PARAMS["ϛ"]
eta, A_d, delta, thbar = PARAMS["η"], PARAMS["A_d"], PARAMS["δ"], PARAMS["θ_bar"]


def diag(net, Z, Y, logxi, lam3=1/6.0):
    o = C.evaluate(*net, np.array([Z]), np.array([Y]), logxi=logxi, lam3=lam3)
    xi = np.exp(logxi)
    K = np.exp(C.logK_fix); E = eta * A_d * (1 - Z) * K
    qd, qg, VY = o["q_d"][0], o["q_g"][0], o["V_Y"][0]
    E_d = (1 - Z) * sd * qd; E_g = Z * sg * qg; E_y = vars_ * E * VY
    h_d, h_g, h_y = -E_d / xi, -E_g / xi, -E_y / xi
    drag = -0.5 / xi * (E_d**2 + E_g**2 + E_y**2)
    flow = delta * (np.log(max(o["c"][0], 1e-9)) + C.logK_fix)
    dv = delta * o["v"][0]
    return dict(xi=xi, i_d=o["i_d"][0], i_g=o["i_g"][0], VY=VY, qd=qd,
                h_d=h_d, h_g=h_g, h_y=h_y, drag=drag, flow=flow, dv=dv, E=E,
                drag_pct=abs(drag) / abs(flow) * 100)


def main():
    net = C.build_and_load()
    print(f"sigma_d=sigma_g={sd}, varsigma={vars_}  (E~{eta*A_d*0.5*np.exp(C.logK_fix):.1f} at Z=0.5)\n")
    Z, Y = 0.5, 3.0
    print(f"At Z={Z}, Y={Y}, logK={C.logK_fix:.2f}, lambda3=1/6:")
    print(f"{'xi':>8} {'i_d':>9} {'i_g':>9} {'V_Y':>9} | {'h_d':>8} {'h_g':>8} {'h_y':>8} | "
          f"{'drag':>10} {'flow':>8} {'drag/flow%':>10}")
    rows = []
    for lx in (5.0, np.log(0.1), np.log(0.05)):
        d = diag(net, Z, Y, lx)
        rows.append(d)
        print(f"{d['xi']:>8.2f} {d['i_d']:>+9.5f} {d['i_g']:>+9.5f} {d['VY']:>+9.4f} | "
              f"{d['h_d']:>+8.4f} {d['h_g']:>+8.4f} {d['h_y']:>+8.4f} | "
              f"{d['drag']:>10.2e} {d['flow']:>8.4f} {d['drag_pct']:>9.2f}%")
    # observed change in controls/value from xi=148.4 -> 0.05
    b, s = rows[0], rows[-1]
    print(f"\nObserved change (xi 148.4 -> 0.05): "
          f"d(i_d)={s['i_d']-b['i_d']:+.2e}  d(i_g)={s['i_g']-b['i_g']:+.2e}  d(V_Y)={s['VY']-b['VY']:+.2e}")
    print(f"=> worst-case drifts h are O(0.05-0.12) (real distortion of DYNAMICS), but the")
    print(f"   value-level drag is ~{s['drag_pct']:.1f}% of flow, so allocations barely move: O(sigma^2/xi).")
    # what sigma would make it a 10% effect?  drag ~ sigma^2/xi ; scale sigma so drag_pct~10%
    scale = np.sqrt(10.0 / s['drag_pct'])
    print(f"   (To get a ~10% drag at xi=0.05 you'd need sigma ~ {sd*scale:.3f}, i.e. ~{scale:.0f}x larger.)")


if __name__ == "__main__":
    main()
