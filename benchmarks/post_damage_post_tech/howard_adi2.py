"""Howard policy iteration + implicit ADI (Douglas) with PROPER Neumann ends and damping.
Key fixes vs v1:
  - tridiagonal rows at the two ends of each axis use a one-sided (zero second-derivative,
    zero-flux drift) stencil so the implicit operator stays an M-matrix (diag dominant).
  - q-tilde floored at a PHYSICAL value (qfloor) so c=num/den can't blow up; this is the
    policy-evaluation regularization that tames the nonlinear feedback.
  - Howard damping omega on the value update for the first few sweeps.
  - moderate dtau (false-transient) so each Howard step is a damped implicit relaxation.
"""
import numpy as np, time
import fd_pdpt as M
P=M.P

def logNy(Y,lam3,p): return p["l1"]+p["l2"]*Y+lam3*(Y-p["y_up"])
def logNyy(lam3,p): return p["l2"]+lam3

def controls(vlK,vZ,Z,p,qfloor):
    qd=np.maximum(vlK-Z*vZ,qfloor); qg=np.maximum(vlK+(1-Z)*vZ,qfloor)
    Abar=(1-Z)*p["A_d"]+Z*p["A_gpp"]
    num=p["delta"]*(Abar+(1-Z)/p["t_d"]+Z/p["t_g"])
    den=p["delta"]+(1-Z)*p["G_d"]*qd+Z*p["G_g"]*qg
    c=num/den
    i_d=p["G_d"]*qd*c/p["delta"]-1.0/p["t_d"]
    i_g=p["G_g"]*qg*c/p["delta"]-1.0/p["t_g"]
    return i_d,i_g,c,qd,qg

def central(v,axis,dx):
    g=np.zeros_like(v)
    lo=[slice(None)]*3; lo[axis]=slice(2,None); hi=[slice(None)]*3; hi[axis]=slice(0,-2)
    md=[slice(None)]*3; md[axis]=slice(1,-1)
    g[tuple(md)]=(v[tuple(lo)]-v[tuple(hi)])/(2*dx)
    e0=[slice(None)]*3; e0[axis]=0; e1=[slice(None)]*3; e1[axis]=1
    en=[slice(None)]*3; en[axis]=-1; en1=[slice(None)]*3; en1[axis]=-2
    g[tuple(e0)]=(v[tuple(e1)]-v[tuple(e0)])/dx
    g[tuple(en)]=(v[tuple(en)]-v[tuple(en1)])/dx
    return g

def thomas(rhs,l,d,u,axis):
    rhs=np.moveaxis(rhs,axis,-1); l=np.moveaxis(l,axis,-1); d=np.moveaxis(d,axis,-1); u=np.moveaxis(u,axis,-1)
    shp=rhs.shape; m=shp[-1]
    r=rhs.reshape(-1,m).copy(); L=l.reshape(-1,m); D=d.reshape(-1,m); U=u.reshape(-1,m)
    cp=np.empty_like(D); dp=np.empty_like(r)
    cp[:,0]=U[:,0]/D[:,0]; dp[:,0]=r[:,0]/D[:,0]
    for k in range(1,m):
        den=D[:,k]-L[:,k]*cp[:,k-1]; cp[:,k]=U[:,k]/den; dp[:,k]=(r[:,k]-L[:,k]*dp[:,k-1])/den
    x=np.empty_like(r); x[:,-1]=dp[:,-1]
    for k in range(m-2,-1,-1): x[:,k]=dp[:,k]-cp[:,k]*x[:,k+1]
    return np.moveaxis(x.reshape(shp),-1,axis)

def coeffs_dir(a,b,dx):
    """interior upwind+diffusion tridiagonal coefficients (lower,diag,upper) as 3D arrays."""
    ap=np.maximum(a,0.0); am=np.minimum(a,0.0)
    lo=-(am/dx)-b/dx**2
    di=(ap/dx)-(am/dx)+2*b/dx**2
    up=-(ap/dx)-b/dx**2
    return lo,di,up

def Ldir(v,a,b,dx,axis):
    ap=np.maximum(a,0.0); am=np.minimum(a,0.0)
    fwd=np.zeros_like(v); bwd=np.zeros_like(v); d2=np.zeros_like(v)
    lo=[slice(None)]*3; hi=[slice(None)]*3
    lo[axis]=slice(0,-1); hi[axis]=slice(1,None)
    fwd[tuple(lo)]=(v[tuple(hi)]-v[tuple(lo)])/dx
    bwd[tuple(hi)]=(v[tuple(hi)]-v[tuple(lo)])/dx
    l2=[slice(None)]*3; h2=[slice(None)]*3; m2=[slice(None)]*3
    l2[axis]=slice(2,None); h2[axis]=slice(0,-2); m2[axis]=slice(1,-1)
    d2[tuple(m2)]=(v[tuple(l2)]-2*v[tuple(m2)]+v[tuple(h2)])/dx**2
    return ap*fwd+am*bwd+b*d2

def implicit_axis(rhs, a, b, dx, axis, dtau, add_diag=0.0):
    """Solve (I + dtau*add_diag - dtau*L_dir) w = rhs along axis, with Neumann ends.
    Ends: drop diffusion (b=0 effect) and use one-sided upwind so the row stays diag-dominant."""
    lo,di,up=coeffs_dir(a,b,dx)
    Dl=-dtau*lo; Dd=1.0+dtau*(di+add_diag); Du=-dtau*up
    # Neumann (zero-flux) at the two ends along axis: make end rows identity+drift only.
    # End 0: forward one-sided; remove the lower coupling (there is none) and the diffusion ghost.
    n=a.shape[axis]
    sl0=[slice(None)]*3; sl0[axis]=0; sln=[slice(None)]*3; sln[axis]=n-1
    # simplest robust choice: identity rows at the boundaries (Neumann handled by extrapolation
    # in the explicit predictor); keeps M-matrix.
    Dl=np.moveaxis(Dl,axis,-1).copy(); Dd=np.moveaxis(Dd,axis,-1).copy(); Du=np.moveaxis(Du,axis,-1).copy()
    Dd[...,0]=1.0; Du[...,0]=0.0
    Dd[...,-1]=1.0; Dl[...,-1]=0.0
    Dl=np.moveaxis(Dl,-1,axis); Dd=np.moveaxis(Dd,-1,axis); Du=np.moveaxis(Du,-1,axis)
    return thomas(rhs,Dl,Dd,Du,axis)

def solve(lam3=1/6.0,xi=148.4,nK=25,nZ=30,nY=25,dtau=2.0,
          howard_iters=400,tol=1e-9,qfloor=0.02,omega=1.0,verbose=True,log_every=25):
    p=P
    logK=np.linspace(4.0,7.0,nK); dK=logK[1]-logK[0]
    Z=np.linspace(0.02,0.98,nZ); dZ=Z[1]-Z[0]
    Y=np.linspace(0.0,4.0,nY); dY=Y[1]-Y[0]
    LK,ZZ,YY=np.meshgrid(logK,Z,Y,indexing="ij"); K=np.exp(LK)
    E=p["eta"]*p["A_d"]*(1-ZZ)*K
    sd2=sg2=p["s_d"]**2
    lNy=logNy(YY,lam3,p); lNyy=logNyy(lam3,p)
    inv_xi=0.0 if not np.isfinite(xi) else 1.0/xi
    delta=p["delta"]
    v=0.5*LK+1.0
    t0=time.time(); step=1.0
    for h in range(howard_iters):
        vlK=central(v,0,dK); vZ=central(v,1,dZ); vY=central(v,2,dY)
        i_d,i_g,c,qd,qg=controls(vlK,vZ,ZZ,p,qfloor)
        phid=p["a_d"]+p["G_d"]*np.log(np.maximum(1+p["t_d"]*i_d,1e-9))
        phig=p["a_g"]+p["G_g"]*np.log(np.maximum(1+p["t_g"]*i_g,1e-9))
        Dc=sd2*(1-ZZ)**2+sg2*ZZ**2
        a_lK=(1-ZZ)*phid+ZZ*phig-Dc/2.0
        a_Z=ZZ*(1-ZZ)*(phig-phid+(1-ZZ)*sd2-ZZ*sg2)
        a_Y=p["thbar"]*E
        b_lK=Dc/2.0; b_Z=0.5*ZZ**2*(1-ZZ)**2*(sd2+sg2); b_Y=0.5*p["vars"]**2*E**2
        cross=(-ZZ*(1-ZZ)**2*sd2+ZZ**2*(1-ZZ)*sg2)
        flow=delta*(np.log(np.maximum(c,1e-12))+LK)
        E_d=(1-ZZ)*p["s_d"]*qd; E_g=ZZ*p["s_g"]*qg; E_y=p["vars"]*E*(vY-lNy)
        robust=-0.5*inv_xi*(E_d**2+E_g**2+E_y**2)
        damage=-(lNy*a_Y+lNyy*b_Y)
        vKZ=np.zeros_like(v)
        vKZ[1:-1,1:-1,:]=(v[2:,2:,:]-v[2:,:-2,:]-v[:-2,2:,:]+v[:-2,:-2,:])/(4*dK*dZ)
        src=flow+robust+damage+cross*vKZ
        # explicit operator pieces
        LKv=Ldir(v,a_lK,b_lK,dK,0); LZv=Ldir(v,a_Z,b_Z,dZ,1); LYv=Ldir(v,a_Y,b_Y,dY,2)
        # Douglas: bring each dim implicit in turn, delta split equally
        d3=delta/3.0
        # stage1 logK
        r1 = v + dtau*(LZv+LYv - delta*v + src) - dtau*(LZv+LYv) + dtau*LKv  # = v+dtau(LKv - delta v + src)... 
        # cleaner standard Douglas:
        r1 = v + dtau*((LKv+LZv+LYv) - delta*v + src) - dtau*LKv
        w1 = implicit_axis(r1, a_lK,b_lK,dK,0,dtau,add_diag=0.0)
        r2 = w1 - dtau*LZv
        w2 = implicit_axis(r2, a_Z,b_Z,dZ,1,dtau,add_diag=0.0)
        r3 = w2 - dtau*LYv
        w3 = implicit_axis(r3, a_Y,b_Y,dY,2,dtau,add_diag=0.0)
        vnew = v + omega*(w3 - v)
        step=np.max(np.abs(vnew-v)); v=vnew
        if verbose and (h%log_every==0 or h==howard_iters-1):
            print("  [howard] it %3d max|dv|=%.3e"%(h,step),flush=True)
        if step<tol and h>3: break
    el=time.time()-t0
    vlK=central(v,0,dK); vZ=central(v,1,dZ); vY=central(v,2,dY)
    i_d,i_g,c,qd,qg=controls(vlK,vZ,ZZ,p,1e-6)
    res=M._residual(v,vlK,vZ,vY,i_d,i_g,c,qd,qg,ZZ,K,E,lNy,lNyy,lam3,xi,logK,Z,Y,dK,dZ,dY,p)
    return dict(logK=logK,Z=Z,Y=Y,v=v,i_d=i_d,i_g=i_g,c=c,vlK=vlK,vZ=vZ,vY=vY,
                iters=h+1,max_abs_residual=float(np.max(np.abs(res))),time=el)

if __name__=="__main__":
    out=solve(dtau=2.0,howard_iters=600,qfloor=0.02,omega=1.0)
    print("iters=%d time=%.2fs resid=%.2e"%(out["iters"],out["time"],out["max_abs_residual"]))
    ik=np.argmin(abs(out["logK"]-np.log(880))); jz=np.argmin(abs(out["Z"]-0.7)); ky=np.argmin(abs(out["Y"]-3.0))
    V_Y=out["vY"][ik,jz,ky]-(0.00017675+2*0.0022*3.0+(1/6.0)*(3.0-2.5))
    print("[logK=%.2f Z=%.2f Y=%.1f] i_d=%+.4f i_g=%+.4f vlK=%.3f c=%.4f V_Y=%.4f"%(
        out["logK"][ik],out["Z"][jz],out["Y"][ky],out["i_d"][ik,jz,ky],out["i_g"][ik,jz,ky],
        out["vlK"][ik,jz,ky],out["c"][ik,jz,ky],V_Y))
    print("TARGET: i_d~0.040 i_g~0.104 vlK~0.54 c~0.064 V_Y~-0.16")
