import torch as tn
import torchtt as tntt
import matplotlib.pyplot as plt
from tt_iga import *
import numpy as np
import datetime
import matplotlib.colors
import pandas as pd

tn.set_default_dtype(tn.float64)


def solve(Np, Ns = [40,20,80], deg = 2 ,nl = 8):
    print()
    print('number of parameters ',Np)
    # used to return the results
    dct = {}

    # B-splines
    Ns = np.array([40,20,80])-deg+1
    baza1 = BSplineBasis(np.linspace(0,1,Ns[0]),deg)
    baza2 = BSplineBasis(np.linspace(0,1,Ns[1]),deg)
    baza3 = BSplineBasis(np.linspace(0,1,Ns[2]),deg)
    Basis = [baza1,baza2,baza3]
    N = [baza1.N,baza2.N,baza3.N]

    # Parameter space basis
    var = 0.05
    Basis_param = [LagrangeLeg(nl,[-var,var])]*Np

    # B-spline basis for the radius perturbation
    bspl = BSplineBasis(np.linspace(0,1,Np-2+3),2)
    def interface_func(t1,tp):
        return tn.einsum('ij,ji->j',tn.tensor(bspl(t1)[1:-1,:]),tn.tensor(tp))
    line = lambda t,a,b: t*(b-a)+a
    damp = lambda x: 1 # -4*x*(x-1) 

    # parametrization
    w = 1
    h = 0.5
    r = 2 
    xparam = lambda t : (2-h+line(t[:,1],0,interface_func(t[:,2],t[:,3:])*damp(t[:,2])+h))*tn.cos(1.5*np.pi+0.25*np.pi*t[:,2]) 
    yparam = lambda t : (2-h+line(t[:,1],0,interface_func(t[:,2],t[:,3:])*damp(t[:,2])+h))*tn.sin(1.5*np.pi+0.25*np.pi*t[:,2]) 
    zparam = lambda t : w*t[:,0]

    # instantiate the GeometryMapping object. It is used for intepolating, evaluating and computing the discrete operators corresponding to a parameter dependent geometry
    geom = Geometry(Basis+Basis_param)
    # interpolate the geometry parametrization
    geom.interpolate([xparam, yparam, zparam])

    # compute the mass matrix in TT
    tme = datetime.datetime.now() 
    Mass_tt = geom.mass_interp(eps=1e-11)
    tme = datetime.datetime.now() -tme
    print('Time mass matrix ',tme.total_seconds())
    
    tme = datetime.datetime.now() 
    Stt = geom.stiffness_interp( eps = 1e-9, qtt = True, verb=True)
    tme = datetime.datetime.now() -tme
    print('Time stiffness matrix ',tme.total_seconds())
    dct['time stiff'] = tme.total_seconds()

    # projection operators for enforcing the BCs
    Pin_tt, Pbd_tt = get_projectors(N,[[1,1],[0,0],[1,1]])
    Pin_tt = Pin_tt ** tntt.eye([nl]*Np)
    Pbd_tt = Pbd_tt ** tntt.eye([nl]*Np)

    # right hand side. Zero since we solve the homogenous equation.
    f_tt = tntt.zeros(Stt.N)

    # interpoalte the excitation and compute the correspinding tensor
    u0 = 1
    extitation_dofs = Function(Basis).interpolate(lambda t: t[:,0]*0+u0)
    tmp = np.zeros(N)
    tmp[:,-1,:] = extitation_dofs[:,-1,:].full()
    g_tt =  Pbd_tt @ (tntt.TT(tmp) ** tntt.ones([nl]*Np))

    # assemble the system matrix
    M_tt = Pin_tt@Stt@Pin_tt + Pbd_tt
    rhs_tt = Pin_tt @ (Mass_tt @ f_tt - Stt @ Pbd_tt @ g_tt) + g_tt
    M_tt = M_tt.round(1e-11)

    # solve the system
    eps_solver = 1e-7
    tme_amen = datetime.datetime.now() 
    # dofs_tt = tntt.solvers.amen_solve(M_tt.cuda(), rhs_tt.cuda(), x0 = tntt.ones(rhs_tt.N).cuda(), eps = eps_solver, nswp = 40, preconditioner = 'c', verbose = False).cpu()
    dofs_tt = tntt.solvers.amen_solve(M_tt, rhs_tt, x0 = tntt.ones(rhs_tt.N), eps = eps_solver, nswp = 40, preconditioner = 'c',  verbose = False)
    tme_amen = (datetime.datetime.now() -tme_amen).total_seconds() 
    dct['time solver'] = tme_amen

    print('Time solver', tme_amen)
    
    # save stats in the dictionary
    dct['rank matrix'] = np.mean(M_tt.R)
    dct['rank rhs'] = np.mean(rhs_tt.R)
    dct['rank solution'] = np.mean(dofs_tt.R)
    dct['memory stiff'] = tntt.numel(Stt)*8/1e6
    dct['memory mass'] = tntt.numel(Mass_tt)*8/1e6
    dct['memory system mat'] = tntt.numel(M_tt)*8/1e6
    dct['memory rhs'] = tntt.numel(rhs_tt)*8/1e6
    dct['memory solution'] = tntt.numel(dofs_tt)*8/1e6
    dct['np'] = Np
    # check the error for the case Theta = 0 (cylinder capacitor)
    fspace = Function(Basis+Basis_param)
    fspace.dofs = dofs_tt

    u_val = fspace([tn.linspace(0,1,8),tn.linspace(0,1,128),tn.linspace(0,1,128)]+[tn.tensor([0.0]) for i in range(Np)]).full()
    x,y,z = geom([tn.linspace(0,1,8),tn.linspace(0,1,128),tn.linspace(0,1,128)]+[tn.tensor([0.0]) for i in range(Np)])
    r = tn.sqrt(x.full()**2+y.full()**2)

    a = u0/np.log(2/(2-h))
    b = u0-a*np.log(2)

    u_ref = a*tn.log(r)+b    

    err = tn.max(tn.abs(u_val-u_ref))

    dct['max_err'] = err
    print('\nMax err %e\n\n'%(err))
    return dct


if __name__ == '__main__':

    Nps = [2,3,4,5,6,7,8,9,10,11,12]

    dct_results = dict()

    for Np in Nps:
        dct = solve(Np)
        dct_results[Np] = dct
        
    # print header 
    df = pd.DataFrame([[el for el in v.values() ] for v in dct_results.values()], columns = [k for k in dct_results[2]])
    print(df)

    eoc = lambda x,y: np.log(y[1:]/y[:-1])/np.log(x[1:]/x[:-1])
