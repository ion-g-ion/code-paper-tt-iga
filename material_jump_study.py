
import torch as tn
import torchtt as tntt
import matplotlib.pyplot as plt
import tt_iga
import iga_fem
import numpy as np
import datetime
import matplotlib.colors
import scipy.sparse.linalg 
import pandas as pd

tn.set_default_dtype(tn.float64)

def iga_solve(deg, n, nl):
    
    print()
    print('#'*32)
    print('deg ',deg,' n ',n,' nl ',nl)
    print('#'*32)
    print()
    
    results = {'n' : n ,'deg' : deg ,'nl' : nl}
    
    Ns = np.array(3*[n])-deg+1
    baza1 = tt_iga.BSplineBasis(np.concatenate((np.linspace(0,0.5,Ns[0]//2),np.linspace(0.5,1,Ns[0]//2))),deg)
    baza2 = tt_iga.BSplineBasis(np.linspace(0,1,Ns[1]),deg)
    baza3 = tt_iga.BSplineBasis(np.concatenate((np.linspace(0,0.3,Ns[2]//3),np.linspace(0.3,0.7,Ns[2]//3),np.linspace(0.7,1,Ns[2]//3))),deg)

    Basis = [baza1,baza2,baza3]
    N = [baza1.N,baza2.N,baza3.N]

    nl = 12
    Basis_param = [tt_iga.LagrangeLeg(nl,[-0.05,0.05])]*4

    # square to circle transformation
    xc = lambda u,v: u*tn.sqrt(1-v**2/2)
    yc = lambda u,v: v*tn.sqrt(1-u**2/2)
    # scale [0,1] to an inteval [a,b]
    line = lambda t,a,b: t*(b-a)+a
    # aux function needed for mapping along the length of the cylinder
    def scaling(z,theta1,theta2):
        a = 0.3
        b = 0.7
        s = (z<a)*line(z/a,0,a+theta1)
        s+= tn.logical_and(z>=a,z<=b)*line((z-a)/(b-a),a+theta1,b+theta2)
        s+= tn.logical_and(z>b,z<=1)*line((z-b)/(1-b),b+theta2,1)
        return s

    # create the components of the parametrization
    angle_mult = 1.0
    xparam = lambda t : xc(t[:,0]*2-1,t[:,1]*2-1)
    yparam = lambda t : yc(t[:,0]*2-1,t[:,1]*2-1)
    zparam = lambda t : scaling(t[:,2],t[:,6],t[:,5]+xparam(t)*angle_mult*t[:,4]+yparam(t)*0*t[:,4])
    # create the material coeffiecient (defined on the reference domain)
    sigma_ref = lambda x:  0.0*x[:,2]+(5.0+x[:,3]*5.0)*tn.logical_and(x[:,0]>=0.0,x[:,0]<0.5)*tn.logical_and(x[:,2]>0.3,x[:,2]<0.7)+1

    # Instantiate the Geometry object and do some plots
    geom = tt_iga.Geometry(Basis+Basis_param)
    geom.interpolate([xparam, yparam, zparam])


    # Construct the mass and stiffness TT operators and the zeros rhs
    tme = datetime.datetime.now() 
    Mass_tt = geom.mass_interp(eps=1e-11)
    tme = datetime.datetime.now() -tme
    print('Time mass matrix ',tme.total_seconds())
    results['time mass'] = tme.total_seconds()

    tme = datetime.datetime.now() 
    Stt = geom.stiffness_interp( func=None, func_reference = sigma_ref, qtt = False, verb=False)
    tme = datetime.datetime.now() -tme
    print('Time stiffness matrix ',tme.total_seconds())
    results['time stiff'] = tme.total_seconds()
    
    f_tt = tntt.zeros(Stt.N)

    # incorporate the boundary conditions and construct the system tensor operator
    Pin_tt,Pbd_tt = tt_iga.get_projectors(N,[[1,1],[1,1],[0,0]])
    # Pbd_tt = (1/N[0]) * Pbd_tt
    U0 = 10

    Pin_tt = Pin_tt ** tntt.eye([nl]*4)
    Pbd_tt = Pbd_tt ** tntt.eye([nl]*4)


    tmp = tn.zeros(N, dtype = tn.float64)
    tmp[:,:,0] = U0 

    g_tt = Pbd_tt @ (tntt.TT(tmp) ** tntt.ones([nl]*4))


    # M_tt = Pin_tt@Stt@Pin_tt + Pbd_tt
    # rhs_tt = Pin_tt @ (Mass_tt @ f_tt - Stt@Pbd_tt@g_tt).round(1e-12) + g_tt
    M_tt = Pin_tt@Stt+Pbd_tt
    rhs_tt = (g_tt).round(1e-11)
    M_tt = M_tt.round(1e-9)
    # print(M_tt,rhs_tt)


    # solve in the TT format
    eps_solver = 1e-7

    print('Solving in TT...')
    tme_amen = datetime.datetime.now() 
    # dofs_tt = tntt.solvers.amen_solve(M_tt.cuda(), rhs_tt.cuda(), x0 = tntt.ones(rhs_tt.N).cuda(), eps = eps_solver, nswp=40, kickrank=4).cpu()
    dofs_tt = tntt.solvers.amen_solve(M_tt, rhs_tt, x0 = tntt.ones(rhs_tt.N), eps = eps_solver, preconditioner = 'c', local_iterations = 20, resets = 8, nswp=60, kickrank=4)
    tme_amen = (datetime.datetime.now() -tme_amen).total_seconds() 
    print('Time system solve in TT ',tme_amen)

    # print(dofs_tt)

    # fspace = Function(Basis+Basis_param)
    # fspace.dofs = dofs_tt

    # fval = fspace([tn.linspace(0,1,128),tn.tensor([0.5]),tn.linspace(0,1,128),tn.tensor([0.05]),tn.tensor([0.05]),tn.tensor([0.05]),tn.tensor([0.05])])
    # x,y,z = geom([tn.linspace(0,1,128),tn.tensor([0.5]),tn.linspace(0,1,128),tn.tensor([0.05]),tn.tensor([0.05]),tn.tensor([0.05]),tn.tensor([0.05])])
   
    results['storage matrix'] = tntt.numel(M_tt)*8/1e6
    results['storage dofs'] = tntt.numel(dofs_tt)*8/1e6    
    results['rank solution'] = dofs_tt.R
    results['rank system'] = M_tt.R
    results['mean rank solution'] = np.mean(dofs_tt.R)
    results['mean rank system'] = np.mean(M_tt.R)
    results['time solver'] = tme_amen
    
    tme_stiff_classic = datetime.datetime.now()
    # stiff_sparse = construct_stiff_sparse([baza1, baza2, baza3],[Xk, Yk, Zk], kappa_ref = lambda y1, y2, y3: 0.0*y2+(5.0+theta1*5.0)*np.logical_and(y1>=0.0,y1<0.5)*np.logical_and(y3>0.3,y3<0.7)+1)
    stiff_sparse = iga_fem.construct_sparse_from_tt(Basis+Basis_param,Stt,[0,0,0,0])
    tme_stiff_classic = (datetime.datetime.now() - tme_stiff_classic).total_seconds() 

    print('Stiff time (conventional) ',tme_stiff_classic)
    results['time stiff classic'] = tme_stiff_classic

    Pin, Pbd = iga_fem.boundary_matrices([baza1, baza2, baza3], opened = [[1,1],[1,1],[0,0]])
    
    M = Pin @ stiff_sparse + Pbd
    rhs = Pbd @ g_tt[:,:,:,0,0,0,0].full().reshape([-1,1])
    tme_gmres_classic = datetime.datetime.now()
    dofs,status = scipy.sparse.linalg.gmres(M,rhs,tol=eps_solver)
    print('GMRES status', status)
    
    tme_gmres_classic = (datetime.datetime.now() - tme_gmres_classic).total_seconds() 
    print('GMRES time ',tme_gmres_classic)
    results['time gmres'] = tme_gmres_classic
    print('ERROR RHS ',np.linalg.norm(rhs.reshape(N)-rhs_tt[:,:,:,0,0,0,0].numpy())) 

    err = np.linalg.norm(dofs_tt[:,:,:,0,0,0,0].numpy()-dofs.reshape(N))/np.linalg.norm(dofs)
    print('ERROR ',err)
    results['error'] = err
    
    return results
        
    
if __name__ == "__main__":   
    
    Ncv = 100
    params = np.random.rand(Ncv,4)*0.1-0.05
    
    results = []
    
    
    degs = [2]
    ns = [20,30,40,50,60,80]
    nls = [12]
    for deg in degs:
        for n in ns:
            for nl in nls:
                res = iga_solve(deg, n, nl)
                results.append(res)
    
    df = pd.DataFrame([[el for el in v.values() ] for v in results], columns = [k for k in results[0]])
    print(df)


    
