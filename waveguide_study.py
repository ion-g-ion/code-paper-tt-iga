
import torch as tn
import torchtt as tntt
import matplotlib.pyplot as plt
from tt_iga import *
import iga_fem
import numpy as np
import datetime
import matplotlib.colors
import scipy.sparse
import scipy.sparse.linalg

tn.set_default_dtype(tn.float64)

def iga_solve(deg, n, nl):
    """
    Solve the waveguide problem for different combinations of parameters

    Args:
        deg (int): degree of Bsplines.
        n (int): size of Bspline basis. The 3d size will be (n,n,2*n).
        nl (int): size of univariate parameter grid.

    Returns:
        dict: dictionary containing results.
    """

    print()
    print('#'*32)
    print('(n,deg,nl)=',(n,deg,nl))
    print('#'*32)
    print()

    results = {} # dictionary fopr the results
    
    Ns = np.array([n]*2+[n*2])-deg+1
    baza1 = BSplineBasis(np.linspace(0,1,Ns[0]),deg)
    baza2 = BSplineBasis(np.linspace(0,1,Ns[1]),deg)
    baza3 = BSplineBasis(np.concatenate((np.linspace(0,0.25,Ns[2]//4),np.linspace(0.25,0.5,Ns[2]//4),np.linspace(0.5,0.75,Ns[2]//4),np.linspace(0.75,1,Ns[2]//4-1))),deg)

    Basis = [baza1,baza2,baza3]
    N = [baza1.N,baza2.N,baza3.N]

    Basis_param = [LagrangeLeg(nl,[-0.2,0.2])]*2+[LagrangeLeg(nl,[-0.3,0.3])]


    xc = lambda u,v: u*tn.sqrt(1-v**2/2)
    yc = lambda u,v: v*tn.sqrt(1-u**2/2)

    line = lambda t,a,b: t*(b-a)+a

    def plane_spanner(P1,P2,P3,t1,t2):
        x = (P1[:,0]-P2[:,0])*t1+(P3[:,0]-P2[:,0])*t2+P2[:,0]
        y = (P1[:,1]-P2[:,1])*t1+(P3[:,1]-P2[:,1])*t2+P2[:,1]
        z = (P1[:,2]-P2[:,2])*t1+(P3[:,2]-P2[:,2])*t2+P2[:,2]
        return x,y,z

    def curve2(t,a,b,c,d):
        w2 = 1.5
        ry = 1.5
        h = 0.5
        rz = 1.5+c
        x = -1*(t<0.25)+tn.logical_and(t>=0.25,t<0.5)*line((t-0.25)/0.25,-1,-w2)+tn.logical_and(t>=0.5,t<=1)*(-w2)
        y = 0*(t<=0.75) + (t>0.75) * ( -ry*tn.cos((t-0.75)/0.25*np.pi/2) + ry )
        z = line(t/0.25,-3,-2+a)*(t<=0.25) + tn.logical_and(t>0.25,t<=0.5)*line((t-0.25)/0.25,-2+a,-1+b) + tn.logical_and(t>0.5,t<=0.75)*line((t-0.5)/0.25,-1+b,-0) + (t>0.75) * ( rz*tn.sin((t-0.75)/0.25*np.pi/2) )
        return tn.hstack((tn.reshape(x,[-1,1]),tn.reshape(y,[-1,1]),tn.reshape(z,[-1,1])))

    def curve1(t,a,b,c,d):
        w2 = 1.5
        ry = 1.5
        h = 0.5
        rz = 1.5+c
        x = 1*(t<0.25)+tn.logical_and(t>=0.25,t<0.5)*line((t-0.25)/0.25,1,w2)+tn.logical_and(t>=0.5,t<=1)*(w2)
        y = 0*(t<=0.75) + (t>0.75) * ( -ry*tn.cos((t-0.75)/0.25*np.pi/2) + ry )
        z = line(t/0.25,-3,-2+a)*(t<=0.25) + tn.logical_and(t>0.25,t<=0.5)*line((t-0.25)/0.25,-2+a,-1+b) + tn.logical_and(t>0.5,t<=0.75)*line((t-0.5)/0.25,-1+b,-0) + (t>0.75) * ( rz*tn.sin((t-0.75)/0.25*np.pi/2) )
        return tn.hstack((tn.reshape(x,[-1,1]),tn.reshape(y,[-1,1]),tn.reshape(z,[-1,1])))
    
    def curve3(t,a,b,c,d):
        w2 = 1.5
        ry = 1.5
        h = 0.5
        rz = 1.5+c
        x = -1*(t<0.25)+tn.logical_and(t>=0.25,t<0.5)*line((t-0.25)/0.25,-1,-w2)+tn.logical_and(t>=0.5,t<=1)*(-w2)
        y = h*(t<=0.75) + (t>0.75) * ( -(ry-h)*tn.cos((t-0.75)/0.25*np.pi/2) + ry )
        z = line(t/0.25,-3,-2+a)*(t<=0.25) + tn.logical_and(t>0.25,t<=0.5)*line((t-0.25)/0.25,-2+a,-1+b) + tn.logical_and(t>0.5,t<=0.75)*line((t-0.5)/0.25,-1+b,-0) + (t>0.75) * ( (rz-h)*tn.sin((t-0.75)/0.25*np.pi/2) )
        return tn.hstack((tn.reshape(x,[-1,1]),tn.reshape(y,[-1,1]),tn.reshape(z,[-1,1])))
        
    scale_mult = 1
    xparam = lambda t : plane_spanner(curve1(t[:,2],scale_mult*t[:,3],scale_mult*t[:,4],scale_mult*t[:,5],0),curve2(t[:,2],scale_mult*t[:,3],scale_mult*t[:,4],scale_mult*t[:,5],0),curve3(t[:,2],scale_mult*t[:,3],scale_mult*t[:,4],scale_mult*t[:,5],0),t[:,0],t[:,1])[0]
    yparam = lambda t : plane_spanner(curve1(t[:,2],scale_mult*t[:,3],scale_mult*t[:,4],scale_mult*t[:,5],0),curve2(t[:,2],scale_mult*t[:,3],scale_mult*t[:,4],scale_mult*t[:,5],0),curve3(t[:,2],scale_mult*t[:,3],scale_mult*t[:,4],scale_mult*t[:,5],0),t[:,0],t[:,1])[1]
    zparam = lambda t : plane_spanner(curve1(t[:,2],scale_mult*t[:,3],scale_mult*t[:,4],scale_mult*t[:,5],0),curve2(t[:,2],scale_mult*t[:,3],scale_mult*t[:,4],scale_mult*t[:,5],0),curve3(t[:,2],scale_mult*t[:,3],scale_mult*t[:,4],scale_mult*t[:,5],0),t[:,0],t[:,1])[2]

    # interpolate the geometry parametrization
    geom = Geometry(Basis+Basis_param)
    geom.interpolate([xparam, yparam, zparam])


    # Construct matrices
    tme = datetime.datetime.now() 
    Mass_tt = geom.mass_interp(eps=1e-11)
    tme = datetime.datetime.now() -tme
    print('Time mass matrix ',tme.total_seconds())
    results['time mass'] = tme
    
    tme = datetime.datetime.now() 
    Stt = geom.stiffness_interp( func=None, func_reference = None, qtt = False, verb=False)
    tme = datetime.datetime.now() -tme
    print('Time stiffness matrix ',tme.total_seconds())
    results['time stiff'] = tme


    Pin_tt,Pbd_tt = get_projectors(N,[[0,0],[0,0],[0,0]])
    # Pbd_tt = (1/N[0]) * Pbd_tt
    U0 = 10

    Pin_tt = Pin_tt ** tntt.eye([nl]*3)
    Pbd_tt = Pbd_tt ** tntt.eye([nl]*3)

    f_tt = tntt.zeros(Stt.N)

    excitation_dofs = Function(Basis).interpolate(lambda t: tn.sin(t[:,0]*np.pi)*tn.sin(t[:,1]*np.pi))
    tmp = tn.zeros(N)
    tmp[:,:,0] = excitation_dofs[:,:,0].full()
    g_tt = Pbd_tt@ (tntt.TT(tmp) ** tntt.ones([nl]*3))


    k = 49

    eps_solver = 1e-8
    M_tt = (Pin_tt@(Stt-k*Mass_tt)+Pbd_tt).round(1e-12)
    rhs_tt = (Pbd_tt @ g_tt).round(1e-12)

    M_tt = M_tt.round(1e-11)

    # M_qtt = ttm2qttm(M_tt).round(1e-9)
    # rhs_qtt = tt2qtt(rhs_tt)

    print('Solving in TT...')
    tme_amen = datetime.datetime.now() 
    # dofs_tt = tntt.solvers.amen_solve(M_tt.cuda(), rhs_tt.cuda(), x0 = tntt.ones(rhs_tt.N).cuda(), eps = eps_solver, nswp=40, kickrank=2, verbose=True, preconditioner = 'c', local_iterations=24, resets=10).cpu()
    dofs_tt = tntt.solvers.amen_solve(M_tt, rhs_tt, x0 = tntt.ones(rhs_tt.N), eps = eps_solver, nswp=40, kickrank=2, verbose=False, preconditioner = 'c', local_iterations=24, resets=10).cpu()
    tme_amen = (datetime.datetime.now() -tme_amen).total_seconds() 
    print('',flush=True)
    print('Time system solve in TT ',tme_amen)
    print('Relative residual ', (M_tt@dofs_tt-rhs_tt).norm()/rhs_tt.norm())

    if False:
        M_qtt = ttm2qttm(M_tt)
        rhs_qtt = tt2qtt(rhs_tt)
        print('Rank Mtt ',M_tt.tt.r)
        print('Rank rhstt ',rhs_tt.r)
        print('TT solution 2 QTT rank ',tt2qtt(dofs_tt).round(1e-10))
        tme_solve_qtt = datetime.datetime.now()
        dofs_qtt = tt.amen.amen_solve(M_qtt,rhs_qtt, tt.ones(rhs_qtt.n), eps_solver,verb=1,nswp = 80,kickrank=6)
        tme_solve_qtt = datetime.datetime.now() - tme_solve_qtt

    print('N ',N)
    print('Rank Mtt ',M_tt.R)
    print('Rank rhstt ',rhs_tt.R)
    print('Rank solution ',dofs_tt.R)
    print('size stiff ',tntt.numel(Stt)*8/1e6,' MB')
    print('size mass ',tntt.numel(Mass_tt)*8/1e6,' MB')
    print('size system mat ',tntt.numel(M_tt)*8/1e6,' MB')
    print('size rhstt ',tntt.numel(rhs_tt)*8/1e6,' MB')
    print('size solution ',tntt.numel(dofs_tt)*8/1e6,' MB, one full solution: ',np.prod(N)*8/1e6,' MB')

    results['time solver'] = tme_amen
    results['storage matrix'] = tntt.numel(M_tt)*8/1e6
    results['storage dofs'] = tntt.numel(dofs_tt)*8/1e6    
    results['rank solution'] = dofs_tt.R
    results['rank system'] = M_tt.R

    fspace = Function(Basis+Basis_param)
    fspace.dofs = dofs_tt
    
    # conventional solver for ONE parameter
    tme_stiff_classic = datetime.datetime.now()
    stiff_sparse = iga_fem.construct_sparse_from_tt(Basis+Basis_param,Stt,[0,0,0])
    mass_sparse = iga_fem.construct_sparse_from_tt(Basis+Basis_param,Mass_tt,[0,0,0])
    tme_stiff_classic = (datetime.datetime.now() - tme_stiff_classic).total_seconds() 

    print('Stiff time ',tme_stiff_classic)
    results['time stiff classic'] = tme_stiff_classic

    Pin, Pbd = iga_fem.boundary_matrices(Basis, opened = [[0,0],[0,0],[0,0]])

    M = (Pin@(stiff_sparse-k*mass_sparse)+Pbd)
    rhs = Pbd @ g_tt[:,:,:,0,0,0].numpy().reshape([-1,1])

    tme_gmres_classic = datetime.datetime.now()
    dofs,status = scipy.sparse.linalg.gmres(M,rhs,tol=eps_solver)
    print('GMRES status', status)
    
    tme_gmres_classic = (datetime.datetime.now() - tme_gmres_classic).total_seconds() 
    print('GMRES time ',tme_gmres_classic)
    results['time gmres'] = tme_gmres_classic
    print('ERROR RHS ',np.linalg.norm(rhs.reshape(N)-rhs_tt[:,:,:,0,0,0].numpy()))
    print('RESIDUAL GMRES ',np.linalg.norm(M.dot(dofs)-rhs)/np.linalg.norm(dofs)) 
    # plt.figure()
    # plt.imshow(dofs_tt[:,nl//2,:,0,0,0,0].full())
    # plt.figure()
    # plt.imshow(dofs.reshape(N)[:,nl//2,:])
    # plt.figure() 
    # plt.imshow(dofs_tt[:,nl//2,:,0,0,0,0].full()-dofs.reshape(N)[:,nl//2,:])
    # plt.colorbar()
    # plt.show()


    err = np.linalg.norm(dofs_tt[:,:,:,0,0,0,].numpy()-dofs.reshape(N))/np.linalg.norm(dofs)
    print('ERROR ',err)
    results['error'] = err
    
    return results


if __name__ == "__main__":   
    
    dct_results =dict()
    
    degs = [2]
    ns = [20,30,40,50,60,70,80,90,100]
    nls = [12]
    for deg in degs:
        for n in ns:
            for nl in nls:
                res = iga_solve(deg, n, nl)
                dct_results[(deg,n,nl)] = res

        