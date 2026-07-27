import numpy as np, os, json
ROOT="/project/lhansen/Cap_damage/TwoStageTechJump_FOCIr_orignal"
FULL=os.path.join(ROOT,"output_lowxi_float64","OneTechJump_Pi_1p0_TechIntensityScale_1p0_AdjustmentCostFull_logximin_m5p30_LR_warmup_cosine_10e-6,40e-5_128_neurons_32_#HiddenLayer_4_num_iterations50000")
XIS=["148.600","0.040","0.030","0.025","0.020","0.010","0.005"]
def L(xi,nm):
    p=os.path.join(FULL,"SimulationDeterministic",f"SimulationOutputs_ξ_{xi}",nm+".txt")
    return np.atleast_1d(np.loadtxt(p)) if os.path.exists(p) else None

print("="*70)
print("AUDIT 1 (figA investment): year-5 i_d/i_g/i_r  (report: i_d 10.27->8.89)")
for xi in XIS:
    print(f"  xi={xi:>8}: i_d={L(xi,'i_d')[60]*100:.3f}%  i_g={L(xi,'i_g')[60]*100:.3f}%  i_r={L(xi,'i_r')[60]*100:.4f}%")

print("="*70)
print("AUDIT 2 (figCURV): distorted lambda3 weights -- sum, xi=inf uniform?")
for xi in XIS:
    w=L(xi,"lambda3_weights_distorted")
    print(f"  xi={xi:>8}: sum={w.sum():.4f}  w=[{' '.join(f'{x:.3f}' for x in w)}]")

print("="*70)
print("AUDIT 3 (figD drift): h at t=0 and h*xi (should be ~const if h~1/xi)")
print(f"  {'xi':>8} | {'h_y0':>9} {'h_y*xi':>9} | {'h_d0':>9} {'h_g0':>9} {'h_r0':>9}")
for xi in XIS:
    xf=148.6 if xi=='148.600' else float(xi)
    hy,hd,hg,hr=L(xi,'h_y')[0],L(xi,'h_d')[0],L(xi,'h_g')[0],L(xi,'h_r')[0]
    print(f"  {xi:>8} | {hy:9.5f} {hy*xf:9.5f} | {hd:9.5f} {hg:9.5f} {hr:9.5f}")

print("="*70)
print("AUDIT 4 (figC/figDENS): dmg_jump_prob(t60), density peak & mass")
for xi in XIS:
    p=L(xi,'dmg_jump_prob'); d=L(xi,'dmg_jump_density')
    print(f"  xi={xi:>8}: P(t60)={p[-1]:.4f}  dens_peak={d.max():.4g}  dens_sum={d.sum():.4g}")

print("="*70)
print("AUDIT 5 (JENSEN BOUND): gbar = distorted_intensity/baseline_J_n  vs  exp(|dV|/xi)")
r1,r2,yl=1.5,0.36,1.5
for xi in XIS:
    Y=L(xi,'Y'); di=L(xi,'dmg_jump_intensity')
    Jn=r1*(np.exp(r2/2*(Y-yl)**2)-1)*(Y>=yl)   # baseline intensity along path
    mask=Jn>1e-9
    gbar=np.full_like(Jn,np.nan); gbar[mask]=di[mask]/Jn[mask]
    gbar_mean=np.nanmean(gbar[mask]) if mask.any() else np.nan
    gbar_max=np.nanmax(gbar[mask]) if mask.any() else np.nan
    print(f"  xi={xi:>8}: gbar_mean={gbar_mean:8.4f} gbar_max={gbar_max:8.3f}  (admissible gbar>=1? {'YES' if gbar_mean>=1 else 'NO -- VIOLATION'})")

print("="*70)
print("AUDIT 5b (Jensen numeric bound from measured value gap dV_rms):")
g=json.load(open('per_xi_g_distortion.json'))
for k in ['0.05','0.04','0.03','0.025','0.02','0.01','0.005']:
    if k in g:
        dv=g[k]['dmg_dV_rms']; xf=float(k)
        print(f"  xi={k:>6}: |dV|_rms={dv:.4f}  Jensen bound exp(|dV|/xi)={np.exp(dv/xf):.3e}  (gbar MUST exceed this)")
