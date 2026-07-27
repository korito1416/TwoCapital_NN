"""Howard policy iteration + implicit ADI (Douglas) linear solve per policy step.
Cross term v_logKZ and robust/damage SOURCE are explicit; the three diagonal
(upwind drift + central diffusion) operators are implicit, split into 3 tridiagonal
sweeps. Outer Howard loop re-evaluates closed-form controls."""
import numpy as np, time
import fd_pdpt as M
P=M.P

def logNy(Y,lam3,p): return p["l1"]+p["l2"]*Y+lam3*(Y-p["y_up"])
def logNyy(lam3,p): return p["l2"]+lam3

def controls(vlK,vZ,Z,p):
    eps=1e-6
    qd=np.maximum(vlK-Z*vZ,eps); qg=np.maximum(vlK+(1-Z)*vZ,eps)
    Abar=(1-Z)*p["A_d"]+Z*p["A_gpp"]
    num=p["delta"]*(Abar+(1-Z)/p["t_d"]+Z/p["t_g"])
    den=p["delta"]+(1-Z)*p["G_d"]*qd+Z*p["G_g"]*qg
    c=num/den
    i_d=p["G_d"]*qd*c/p["delta"]-1.0/p["t_d"]
    i_g=p["G_g"]*qg*c/p["delta"]-1.0/p["t_g"]
    return i_d,i_g,c,qd,qg

def central(v,axis,dx):
    g=np.zeros_like(v)
    lo=[slice(None)]*3; lo[axis]=slice(2,None)
    hi=[slice(None)]*3; hi[axis]=slice(0,-2)
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
    r=rhs.reshape(-1,m).copy(); L=l.reshape(-1,m); D=d.reshape(-1,m).copy(); U=u.reshape(-1,m)
    cp=np.empty_like(D); dp=np.empty_like(r)
    cp[:,0]=U[:,0]/D[:,0]; dp[:,0]=r[:,0]/D[:,0]
    for k in range(1,m):
        den=D[:,k]-L[:,k]*cp[:,k-1]
        cp[:,k]=U[:,k]/den
        dp[:,k]=(r[:,k]-L[:,k]*dp[:,k-1])/den
    x=np.empty_like(r); x[:,-1]=dp[:,-1]
    for k in range(m-2,-1,-1):
        x[:,k]=dp[:,k]-cp[:,k]*x[:,k+1]
    return np.moveaxis(x.reshape(shp),-1,axis)

def build_tridiag(a,b,dx,n,axis,frac):
    """Implicit operator (I - frac*dtau*[upwind a*d/dx + b*d2/dx2]) tridiagonal coeffs.
    frac splits the I/dtau evenly across the 3 sweeps in Douglas. Returns l,d,u (full 3D),
    and the explicit operator value L_dir*v for the predictor."""
    ap=np.maximum(a,0.0); am=np.minimum(a,0.0)
    # upwind: a>0 uses (v[i+1]-v[i])/dx (couples i,i+1); a<0 uses (v[i]-v[i-1])/dx (couples i-1,i)
    lo = -(am/dx) - b/dx**2                 # coef of v[i-1]
    di =  (ap/dx) - (am/dx) + 2*b/dx**2      # coef of v[i]
    up = -(ap/dx) - b/dx**2                  # coef of v[i+1]
    return lo,di,up

def Ldir(v,a,b,dx,axis):
    """explicit value of upwind-drift + central-diffusion along axis (for residual/predictor)."""
    ap=np.maximum(a,0.0); am=np.minimum(a,0.0)
    fwd=np.zeros_like(v); bwd=np.zeros_like(v); d2=np.zeros_like(v)
    lo=[slice(None)]*3; hi=[slice(None)]*3; md=[slice(None)]*3
    lo[axis]=slice(0,-1); hi[axis]=slice(1,None)
    fwd[tuple(lo)]=(v[tuple(hi)]-v[tuple(lo)])/dx
    bwd[tuple(hi)]=(v[tuple(hi)]-v[tuple(lo)])/dx
    l2=[slice(None)]*3; h2=[slice(None)]*3; m2=[slice(None)]*3
    l2[axis]=slice(2,None); h2[axis]=slice(0,-2); m2[axis]=slice(1,-1)
    d2[tuple(m2)]=(v[tuple(l2)]-2*v[tuple(m2)]+v[tuple(h2)])/dx**2
    return ap*fwd+am*bwd+b*d2

def solve(lam3=1/6.0,xi=148.4,nK=25,nZ=30,nY=25,dtau=20.0,
          howard_iters=60,tol=1e-9,verbose=True):
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
    t0=time.time()
    for h in range(howard_iters):
        vlK=central(v,0,dK); vZ=central(v,1,dZ); vY=central(v,2,dY)
        i_d,i_g,c,qd,qg=controls(vlK,vZ,ZZ,p)
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
        # cross term explicit
        vKZ=np.zeros_like(v)
        vKZ[1:-1,1:-1,:]=(v[2:,2:,:]-v[2:,:-2,:]-v[:-2,2:,:]+v[:-2,:-2,:])/(4*dK*dZ)
        src=flow+robust+damage+cross*vKZ      # explicit source S (frozen at policy)

        # ---- Douglas ADI for (I/dtau + delta - L_K - L_Z - L_Y) v_new = v/dtau + S ----
        # Predictor: explicit full operator
        LKv=Ldir(v,a_lK,b_lK,dK,0); LZv=Ldir(v,a_Z,b_Z,dZ,1); LYv=Ldir(v,a_Y,b_Y,dY,2)
        Av = LKv+LZv+LYv - delta*v + src
        v0=v
        rhs = v0 + dtau*Av        # explicit predictor residual form
        # Sweep 1 (logK): (I + dtau*(delta/3 ... )) -- we use Douglas: implicit one dim at a time,
        # subtracting that dim's explicit contribution.
        # (I - dtau L_K) Y1 = v0 + dtau(Av) - dtau L_K v0   ... standard Douglas first stage
        loK,diK,upK=build_tridiag(a_lK,b_lK,dK,nK,0,1.0)
        r1 = rhs - dtau*LKv + dtau*0  # move L_K to implicit: rhs1 = v0+dtau*Av - dtau*L_K v0
        # Actually Douglas stage1: (I - dtau L_K) w1 = v0 + dtau*(L_K+L_Z+L_Y-delta)v0 + dtau S - dtau L_K v0
        #                                            = v0 + dtau*(L_Z+L_Y-delta)v0 + dtau S
        # include delta in diagonal: bring -delta*v implicit-ish? keep delta on diagonal of stage1.
        r1 = v0 + dtau*(LZv+LYv - delta*v0 + src)
        dK_l = -dtau*loK; dK_d = 1.0+dtau*(diK+delta); dK_u=-dtau*upK
        # neumann ends: fold boundary (one-sided) -> set first/last rows to identity-ish via clamping
        w1=thomas(r1, dK_l, dK_d, dK_u, 0)
        # Stage2 (Z): (I - dtau L_Z) w2 = w1 - dtau L_Z v0
        loZ,diZ,upZ=build_tridiag(a_Z,b_Z,dZ,nZ,1,1.0)
        r2 = w1 - dtau*LZv
        dZ_l=-dtau*loZ; dZ_d=1.0+dtau*diZ; dZ_u=-dtau*upZ
        w2=thomas(r2,dZ_l,dZ_d,dZ_u,1)
        # Stage3 (Y): (I - dtau L_Y) w3 = w2 - dtau L_Y v0
        loY,diY,upY=build_tridiag(a_Y,b_Y,dY,nY,2,1.0)
        r3 = w2 - dtau*LYv
        dY_l=-dtau*loY; dY_d=1.0+dtau*diY; dY_u=-dtau*upY
        w3=thomas(r3,dY_l,dY_d,dY_u,2)
        vnew=w3
        step=np.max(np.abs(vnew-v)); v=vnew
        if verbose and (h%5==0 or h==howard_iters-1):
            print("  [howard] it %3d max|dv|=%.3e"%(h,step),flush=True)
        if step<tol: break
    el=time.time()-t0
    vlK=central(v,0,dK); vZ=central(v,1,dZ); vY=central(v,2,dY)
    i_d,i_g,c,qd,qg=controls(vlK,vZ,ZZ,p)
    res=M._residual(v,vlK,vZ,vY,i_d,i_g,c,qd,qg,ZZ,K,E,lNy,lNyy,lam3,xi,logK,Z,Y,dK,dZ,dY,p)
    return dict(logK=logK,Z=Z,Y=Y,v=v,i_d=i_d,i_g=i_g,c=c,vlK=vlK,vZ=vZ,vY=vY,
                iters=h+1,max_abs_residual=float(np.max(np.abs(res))),time=el)

if __name__=="__main__":
    out=solve()
    print("iters=%d time=%.2fs resid=%.2e"%(out["iters"],out["time"],out["max_abs_residual"]))
    ik=np.argmin(abs(out["logK"]-np.log(880))); jz=np.argmin(abs(out["Z"]-0.7)); ky=np.argmin(abs(out["Y"]-3.0))
    print("[at logK=%.2f Z=%.2f Y=%.1f] i_d=%+.4f i_g=%+.4f vlK=%.3f c=%.4f vY=%.4f V_Y=%.4f"%(
        out["logK"][ik],out["Z"][jz],out["Y"][ky],out["i_d"][ik,jz,ky],out["i_g"][ik,jz,ky],
        out["vlK"][ik,jz,ky],out["c"][ik,jz,ky],out["vY"][ik,jz,ky],
        out["vY"][ik,jz,ky]-(0.00017675+2*0.0022*3.0+(1/6.0)*(3.0-2.5))))
