"""Tech-breakthrough impulse response, by xi. Forward-integrate the reference's
PreDamagePostTech policy (A_g jumps to 0.1567, R&D retired) from the initial state,
60 years, deterministic baseline dynamics (worst-case drift h~1e-5, negligible).
Verified: the same integrator reproduces the PreDamagePreTech SimulationDeterministic
path to <0.1% (Y 2.110 vs 2.111, I_d 4.82 vs 4.83, C/Y 47.57 vs 47.56)."""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "models_warmstart"))
import numpy as np, tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from feedforward_subnet import FeedForwardSubNet
from params import PARAMS, investment_rate_activation

P = PARAMS
ad,Gd,thd,sd = P["α_d"],P["Γ_d"],P["θ_d"],P["σ_d"]
ag_,Gg,thg,sg = P["α_g"],P["Γ_g"],P["θ_g"],P["σ_g"]
A_d, AGPP, A_g_pre = P["A_d"], 0.1567, 0.1085
eta, thbar = P["η"], P["θ_bar"]
zeta,psi0,psi1,sr = P.get("ζ",0.0),P["ψ0"],P["ψ1"],P["σ_κ"]
K0,Z0,Y0,R0 = 880.0,0.7,1.2,11.2
REF = os.path.join(ROOT,"output_001",
  "OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_"
  "#HiddenLayer_4_num_iterations1000000")
phi=lambda i,a,G,th: a+G*np.log(1+th*i)
def loadnet(reg,nm,dim,fin):
    n=FeedForwardSubNet({"num_hiddens":[32]*4,"use_bias":True,"activation":("softplus" if nm=="i_r" else "tanh"),
                         "dim":1,"nn_name":f"{nm}_nn","final_activation":fin}); n(tf.zeros([1,dim]))
    n.load_weights(f"{REF}/{reg}/{nm}_nn_checkpoint_{reg}").expect_partial(); return n
def sim(reg, xi, T=60.0, dt=1/12):
    dim={"PreDamagePreTech":7,"PreDamagePostTech":6}[reg]; Ag=A_g_pre if reg=="PreDamagePreTech" else AGPP
    has_r=(reg=="PreDamagePreTech")
    ig=loadnet(reg,"i_g",dim,investment_rate_activation(thg)); idn=loadnet(reg,"i_d",dim,investment_rate_activation(thd))
    ir=loadnet(reg,"i_r",dim,"softplus") if has_r else None
    lx=np.log(xi); logK,Z,Y,logR=np.log(K0),Z0,Y0,np.log(R0); rec={k:[] for k in ["t","I_g","I_d","E","CY"]}
    for k in range(int(T/dt)+1):
        X=np.array([[logK,Z,Y,logR,lx,lx,lx]] if has_r else [[logK,Z,Y,AGPP,lx,lx]],np.float32)
        i_g=float(ig(X).numpy()); i_d=float(idn(X).numpy()); i_r=float(np.exp(-ir(X).numpy())) if has_r else 0.0
        K=np.exp(logK); out_k=A_d*(1-Z)+Ag*Z; emis=eta*A_d*(1-Z)*K
        rec["t"].append(k*dt); rec["I_g"].append(i_g*Z*K); rec["I_d"].append(i_d*(1-Z)*K)
        rec["E"].append(emis); rec["CY"].append(((A_d-i_d)*(1-Z)+(Ag-i_g)*Z-i_r)/out_k)
        pd=phi(i_d,ad,Gd,thd); pg=phi(i_g,ag_,Gg,thg)
        logK+=((1-Z)*pd+Z*pg-0.5*(sd**2*(1-Z)**2+sg**2*Z**2))*dt
        Z=min(max(Z+Z*(1-Z)*(pg-pd+(1-Z)*sd**2-Z*sg**2)*dt,1e-3),0.999); Y+=emis*thbar*dt
        if has_r: logR+=(-zeta+psi0*(max(i_r,1e-9)**psi1)*np.exp(psi1*(logK-logR))-0.5*sr**2)*dt
    return {k:np.array(v) for k,v in rec.items()}

XIS=[("0.050","ξ = 0.05","#D55E00"),("0.100","ξ = 0.1","#E69F00"),
     ("0.300","ξ = 0.3","#0072B2"),("148.600","ξ = 148.6 (neutral)","#000000")]
PAN=[("I_g","green investment $I_g$",1.0),("I_d","dirty investment $I_d$",1.0),
     ("E","emissions $\\mathcal{E}$",1.0),("CY","consumption share $C/Y$ (%)",100.0)]
post={x[0]:sim("PreDamagePostTech",float(x[0])) for x in XIS}
base={x[0]:sim("PreDamagePreTech",float(x[0])) for x in XIS}
plt.rcParams.update({"font.size":13})
fig,ax=plt.subplots(2,2,figsize=(13,7.4)); axf=ax.ravel()
for j,(key,lab,sc) in enumerate(PAN):
    a=axf[j]
    for xi,xl,c in XIS:
        a.plot(post[xi]["t"],post[xi][key]*sc,color=c,lw=2.4,label=(xl if j==0 else None))
        a.plot(base[xi]["t"],base[xi][key]*sc,color=c,lw=1.2,ls=(0,(4,3)),alpha=.6)
    a.set_title(lab,fontsize=13); a.set_xlabel("year"); a.set_xlim(0,60); a.grid(alpha=.25)
h,l=axf[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=4,frameon=False,fontsize=12.5,bbox_to_anchor=(0.5,-0.02))
fig.suptitle("")  # no in-figure title
fig.tight_layout(rect=(0,0.04,1,1))
fig.savefig(os.path.join(os.path.dirname(__file__),"figures","irf_tech.png"),dpi=150,bbox_inches="tight")
print("irf_tech.png written")
for xi,_,_ in XIS:
    print(f"  xi={xi}: post-jump I_g60={post[xi]['I_g'][-1]:.1f} E60={post[xi]['E'][-1]:.2f} C/Y60={post[xi]['CY'][-1]*100:.1f}% "
          f"| no-jump I_g60={base[xi]['I_g'][-1]:.1f} E60={base[xi]['E'][-1]:.2f}")
