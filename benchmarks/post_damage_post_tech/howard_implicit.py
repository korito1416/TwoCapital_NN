"""Howard policy iteration with ONE fully-implicit sparse linear solve per policy step.
Per Howard step the controls (hence all coefficients a_*,b_*,phi,flow,robust,damage) are FROZEN,
so the PDE delta*v = L v + S is LINEAR in v. We assemble the 7-point M-matrix (upwind drift +
central diffusion, diagonal dims) PLUS the explicit cross term folded into the RHS source, and
solve A v = b once with scipy spsolve. Outer loop re-evaluates controls until v converges.

This is policy iteration: each linear solve is the exact value of the frozen policy; Howard
converges monotonically and (with M-matrix) is unconditionally stable -- no dtau, no CFL.
"""
import numpy as np, time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, splu
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

def solve(lam3=1/6.0,xi=148.4,nK=25,nZ=30,nY=25,howard_iters=80,tol=1e-9,
          qfloor=0.02,verbose=True,use_lu=True):
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
    N=nK*nZ*nY
    idx=np.arange(N).reshape(nK,nZ,nY)
    v=(0.5*LK+1.0)
    t0=time.time()
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
        S = flow+robust+damage+cross*vKZ      # explicit source (cross frozen)

        # assemble A (N x N): delta*v - L v = S  -> (delta I - L) v = S
        rows=[]; cols=[]; vals=[]
        def add(i,j,val): rows.append(i); cols.append(j); vals.append(val)
        I=idx
        # diagonal base: delta
        diag=np.full((nK,nZ,nY),delta)
        # ---- logK ----
        ap=np.maximum(a_lK,0.0); am=np.minimum(a_lK,0.0)
        # interior (1..nK-2): upwind + central diff
        # coefficients (operator L, so we SUBTRACT in A): A gets -L
        # L v_i = ap*(v_{i+1}-v_i)/dK + am*(v_i-v_{i-1})/dK + b*(v_{i+1}-2v_i+v_{i-1})/dK^2
        cup = ap/dK + b_lK/dK**2          # coupling to i+1
        clo = -am/dK + b_lK/dK**2         # coupling to i-1
        cdi = -(ap-am)/dK - 2*b_lK/dK**2  # self (this is L_ii)
        # add -L to A
        sl=slice(1,nK-1)
        diag[sl,:,:]+= -cdi[sl,:,:]
        add_arr=lambda a:a.ravel()
        # build via flat loops over interior using vectorized index arrays
        def couple(axis, c_self, c_lo, c_hi):
            pass
        # We'll just build with explicit COO via meshgrid masks for each axis.
        # logK couplings:
        for (offset,carr,name) in [(+1,cup,"up"),(-1,clo,"lo")]:
            ii,jj,kk=np.meshgrid(np.arange(1,nK-1),np.arange(nZ),np.arange(nY),indexing="ij")
            src_i=I[ii,jj,kk]; tgt=I[ii+offset,jj,kk]
            rows.append(src_i.ravel()); cols.append(tgt.ravel()); vals.append((-carr[ii,jj,kk]).ravel())
        # logK Neumann ends (rows 0 and nK-1): one-sided, zero second deriv. Use drift only.
        # end 0: a*(v1-v0)/dK ; end n-1: a*(vn-1 - vn-2)/dK -- fold into diag/off
        # (kept simple: zero-flux => identity contribution from diffusion, drift one-sided)
        # ---- Z ----
        apZ=np.maximum(a_Z,0.0); amZ=np.minimum(a_Z,0.0)
        cupZ=apZ/dZ + b_Z/dZ**2; cloZ=-amZ/dZ + b_Z/dZ**2; cdiZ=-(apZ-amZ)/dZ-2*b_Z/dZ**2
        diag[:,1:nZ-1,:]+= -cdiZ[:,1:nZ-1,:]
        for (offset,carr) in [(+1,cupZ),(-1,cloZ)]:
            ii,jj,kk=np.meshgrid(np.arange(nK),np.arange(1,nZ-1),np.arange(nY),indexing="ij")
            src_i=I[ii,jj,kk]; tgt=I[ii,jj+offset,kk]
            rows.append(src_i.ravel()); cols.append(tgt.ravel()); vals.append((-carr[ii,jj,kk]).ravel())
        # ---- Y ----
        apY=np.maximum(a_Y,0.0); amY=np.minimum(a_Y,0.0)
        cupY=apY/dY + b_Y/dY**2; cloY=-amY/dY + b_Y/dY**2; cdiY=-(apY-amY)/dY-2*b_Y/dY**2
        diag[:,:,1:nY-1]+= -cdiY[:,:,1:nY-1]
        for (offset,carr) in [(+1,cupY),(-1,cloY)]:
            ii,jj,kk=np.meshgrid(np.arange(nK),np.arange(nZ),np.arange(1,nY-1),indexing="ij")
            src_i=I[ii,jj,kk]; tgt=I[ii,jj,kk+offset]
            rows.append(src_i.ravel()); cols.append(tgt.ravel()); vals.append((-carr[ii,jj,kk]).ravel())
        # diagonal entries
        rows.append(I.ravel()); cols.append(I.ravel()); vals.append(diag.ravel())
        R=np.concatenate(rows); C=np.concatenate(cols); V=np.concatenate(vals)
        A=csr_matrix((V,(R,C)),shape=(N,N))
        b=S.ravel().copy()
        # boundary rows: Neumann v_end = v_neighbor (zero-flux). Overwrite those rows.
        # logK ends
        bnd=[]
        b0=I[0,:,:].ravel();  bn=I[nK-1,:,:].ravel()
        # We'll instead solve with the simple approach: set boundary rows to (v_end - v_in)=0
        # Build a lil for row overrides
        A=A.tolil()
        def neumann(end_idx,in_idx):
            for e,n_ in zip(end_idx,in_idx):
                A.rows[e]=[e,n_]; A.data[e]=[1.0,-1.0]; b[e]=0.0
        neumann(I[0,:,:].ravel(), I[1,:,:].ravel())
        neumann(I[nK-1,:,:].ravel(), I[nK-2,:,:].ravel())
        neumann(I[:,0,:].ravel(), I[:,1,:].ravel())
        neumann(I[:,nZ-1,:].ravel(), I[:,nZ-2,:].ravel())
        neumann(I[:,:,0].ravel(), I[:,:,1].ravel())
        neumann(I[:,:,nY-1].ravel(), I[:,:,nY-2].ravel())
        A=A.tocsc()
        vnew=spsolve(A,b).reshape(nK,nZ,nY)
        step=np.max(np.abs(vnew-v)); v=vnew
        if verbose and (h%5==0 or h==howard_iters-1):
            print("  [howard] it %3d max|dv|=%.3e"%(h,step),flush=True)
        if step<tol and h>2: break
    el=time.time()-t0
    vlK=central(v,0,dK); vZ=central(v,1,dZ); vY=central(v,2,dY)
    i_d,i_g,c,qd,qg=controls(vlK,vZ,ZZ,p,1e-6)
    res=M._residual(v,vlK,vZ,vY,i_d,i_g,c,qd,qg,ZZ,K,E,lNy,lNyy,lam3,xi,logK,Z,Y,dK,dZ,dY,p)
    return dict(logK=logK,Z=Z,Y=Y,v=v,i_d=i_d,i_g=i_g,c=c,vlK=vlK,vZ=vZ,vY=vY,
                iters=h+1,max_abs_residual=float(np.max(np.abs(res))),time=el)

if __name__=="__main__":
    out=solve(howard_iters=80,qfloor=0.02)
    print("iters=%d time=%.2fs resid=%.2e"%(out["iters"],out["time"],out["max_abs_residual"]))
    ik=np.argmin(abs(out["logK"]-np.log(880))); jz=np.argmin(abs(out["Z"]-0.7)); ky=np.argmin(abs(out["Y"]-3.0))
    V_Y=out["vY"][ik,jz,ky]-(0.00017675+2*0.0022*3.0+(1/6.0)*(3.0-2.5))
    print("[logK=%.2f Z=%.2f Y=%.1f] i_d=%+.4f i_g=%+.4f vlK=%.3f c=%.4f V_Y=%.4f"%(
        out["logK"][ik],out["Z"][jz],out["Y"][ky],out["i_d"][ik,jz,ky],out["i_g"][ik,jz,ky],
        out["vlK"][ik,jz,ky],out["c"][ik,jz,ky],V_Y))
    print("TARGET: i_d~0.040 i_g~0.104 vlK~0.54 c~0.064 V_Y~-0.16")
