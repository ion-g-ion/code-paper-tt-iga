import torch as tn
import torchtt as tntt
import matplotlib.pyplot as plt
import tt_iga
import numpy as np
import datetime
import matplotlib.colors
import scipy.sparse
import scipy.sparse.linalg
import iga_fem
import pandas as pd

tn.set_default_dtype(tn.float64)

def solve(Ns,deg,nl,alpha=1/4,eps_solver = 10*1e-9,eps_construction=1e-11,qtt = True,conventional = True, gpu_solver = False):
   
    baza1 = tt_iga.BSplineBasis(np.linspace(0,1,Ns[0]-deg+1),deg)
    baza2 = tt_iga.BSplineBasis(np.linspace(0,1,Ns[1]-deg+1),deg)
    baza3 = tt_iga.BSplineBasis(np.linspace(0,1,Ns[2]-deg+1),deg)
    
    Basis = [baza1,baza2,baza3]
    N = [baza1.N,baza2.N,baza3.N]
    
    Basis_param = [tt_iga.LagrangeLeg(nl,[0,1])]
    
    xc = lambda u,v: u*np.sqrt(1-v**2/2)
    yc = lambda u,v: v*np.sqrt(1-u**2/2)

    xparam = lambda t : xc(t[:,0]*2-1,t[:,1]*2-1)*((1+np.cos((t[:,2]*2-1)*np.pi))*alpha*t[:,3]+1)
    yparam = lambda t : yc(t[:,0]*2-1,t[:,1]*2-1)*((1+np.cos((t[:,2]*2-1)*np.pi))*alpha*t[:,3]+1)
    zparam = lambda t : t[:,2]*2-1

    geom = tt_iga.Geometry(Basis+Basis_param)
    geom.interpolate([xparam, yparam, zparam])      


    tme = datetime.datetime.now() 
    Mass_tt = geom.mass_interp(eps=1e-12)
    tme = datetime.datetime.now() -tme
    print('Time mass matrix ',tme.total_seconds())
    tme_mass = tme.total_seconds()
    
    
    tme = datetime.datetime.now() 
    Stiff_tt = geom.stiffness_interp( eps = eps_construction, qtt = False, verb=False)
    tme = datetime.datetime.now() -tme
    print('Time stiffness matrix ',tme.total_seconds())
    tme_stiff = tme.total_seconds()
    
    N = [baza1.N,baza2.N,baza3.N]


    # interpolate rhs and reference solution
    sigma = 0.5
    uref = lambda x: np.exp(-((x[:,0]-0.0)**2+(x[:,1]-0.0)**2+(x[:,2]-0)**2)/sigma)
    ffun = lambda x: -np.exp(-((x[:,0]-0.0)**2+(x[:,1]-0.0)**2+(x[:,2]-0)**2)/sigma)*(-6*sigma+4*((x[:,0]-0.0)**2+(x[:,1]-0.0)**2+(x[:,2]-0)**2))/sigma/sigma
    # uref =lambda x: np.sin(np.pi*2*x[:,0])*np.sin(np.pi*2*x[:,1])*np.sin(np.pi*2*x[:,2])
    # ffun= lambda x: 3*4*np.pi**2*np.sin(np.pi*2*x[:,0])*np.sin(np.pi*2*x[:,1])*np.sin(np.pi*2*x[:,2])
    # gfun= lambda x: x[:,0]*0+1.0
    kx = 2
    ky = 3
    uref = lambda x: np.sin(kx*x[:,0])*np.cos(ky*x[:,1])*np.exp(-np.sqrt(kx*kx+ky*ky)*x[:,2])
    ffun = lambda x: x[:,0]*0
    
    uref_fun = tt_iga.Function(Basis+Basis_param)
    uref_fun.interpolate(uref, geometry = geom, eps = 1e-14)
    
    f_fun = tt_iga.Function(Basis+Basis_param)
    f_fun.dofs = tntt.zeros(uref_fun.dofs.N)
    

    Pin_tt, Pbd_tt = tt_iga.get_projectors(N,[[0,0],[0,0],[0,0]]) 
   
    
    Pin_tt = Pin_tt ** tntt.eye(Stiff_tt.N[3:])
    Pbd_tt = Pbd_tt ** tntt.eye(Stiff_tt.N[3:])
    
    
    Pbd_tt = (N[0]**-1) * Pbd_tt
    M_tt = (Pin_tt@Stiff_tt+Pbd_tt).round(eps_construction)
    rhs_tt = (Pin_tt @ Mass_tt @ f_fun.dofs + Pbd_tt @ uref_fun.dofs ).round(eps_construction)
    
    print('System matrix... ',flush=True)


    print('Rank Mtt ',M_tt.R)
    print('Rank rhstt ',rhs_tt.R)
    print('Rank uref TT',uref_fun.dofs.R)
    
    tme = datetime.datetime.now() 
    print('eps solver ',eps_solver,flush=True)
    # dofs_tt = tntt.solvers.amen_solve(M_tt.cuda(), rhs_tt.cuda(), x0 = tntt.ones(rhs_tt.N).cuda(), eps = eps_solver, nswp = 50, kickrank = 4, preconditioner = 'c', verbose = False).cpu()
    dofs_tt = tntt.solvers.amen_solve(M_tt, rhs_tt, x0 = tntt.ones(rhs_tt.N), eps = eps_solver, nswp = 50, kickrank = 4, preconditioner = 'c', verbose = False).cpu()
    tme = datetime.datetime.now() -tme
    print('Time system solve ',tme,flush=True)
    tme_solve = tme.total_seconds()
   
    if qtt:
        M_qtt = M_tt.to_qtt().round(eps_construction)
        rhs_qtt = rhs_tt.to_qtt().round(eps_construction)
        print('Rank Mqtt ',M_qtt.R)
        print('Rank rhsqtt ',rhs_qtt.R)
    
    tme_solve_qtt = datetime.datetime.now()
    # if qtt: dofs_qtt = tntt.solvers.amen_solve(M_qtt.cuda(), rhs_qtt.cuda(), x0 = rhs_qtt.round(1e-10,1).cuda(), eps = eps_solver, nswp = 80, kickrank = 6, preconditioner='c').cpu()
    if qtt: dofs_qtt = tntt.solvers.amen_solve(M_qtt, rhs_qtt, x0 = rhs_qtt.round(1e-10,1), eps = eps_solver, nswp = 80, kickrank = 6, preconditioner='c').cpu()
    tme_solve_qtt = datetime.datetime.now() - tme_solve_qtt
    print('Time in QTT ',tme_solve_qtt,flush=True)
    tme_solve_qtt = tme_solve_qtt.total_seconds()
    
    print('residual TT ',(M_tt@dofs_tt-rhs_tt).norm()/rhs_tt.norm())
    print('residual ref TT ',(M_tt@uref_fun.dofs-rhs_tt).norm()/rhs_tt.norm())
    print('residual BD ' , (Pbd_tt@dofs_tt-Pbd_tt@uref_fun.dofs).norm()/Pbd_tt.norm())
    print('error tens tt', (dofs_tt - uref_fun.dofs).norm()/dofs_tt.norm())

    
    
    if qtt:
        print('TT size ',tntt.numel(dofs_tt)*8/1e6,' [MB], QTT ',tntt.numel(dofs_qtt.round(1e-10))*8/1e6,' [MB]')
        storageqtt = tntt.numel(dofs_qtt.round(1e-10))*8/1e6
    else:
        storageqtt = 0
        
    storage = tntt.numel(dofs_tt.round(1e-10))*8/1e6
    Ns = 1000
    
    if qtt: dofs_tt = tntt.reshape(dofs_qtt,dofs_tt.N)
    
    
    
    # classic solver for only one  parameter
    if np.prod(N)<=64**3 and conventional:
        tme_stiff_classic = datetime.datetime.now()
        # stiff_sparse = iga_fem.construct_stiff_sparse([baza1, baza2, baza3],[geom.Xs[0][:,:,:,nl//2].numpy(), geom.Xs[1][:,:,:,nl//2].full(), geom.Xs[2][:,:,:,nl//2].full()])
        stiff_sparse = iga_fem.construct_sparse_from_tt(Basis+Basis_param, Stiff_tt, [nl//2])
        tme_stiff_classic = (datetime.datetime.now() - tme_stiff_classic).total_seconds() 
        print('Stiff time ',tme_stiff_classic)

        Pin, Pbd = iga_fem.boundary_matrices([baza1, baza2, baza3])
        
        M = Pin @ stiff_sparse + Pbd
        rhs = Pin @ f_fun.dofs[:,:,:,nl//2].numpy().reshape([-1,1]) + Pbd @ uref_fun.dofs[:,:,:,nl//2].numpy().reshape([-1,1])
        tme_gmres_classic = datetime.datetime.now()
        dofs = scipy.sparse.linalg.gmres(M,rhs,tol=eps_solver)[0]
        tme_gmres_classic = (datetime.datetime.now() - tme_gmres_classic).total_seconds() 
        print('GMRES time ',tme_gmres_classic)
        err = np.linalg.norm(dofs_tt[:,:,:,nl//2].full()-dofs.reshape(N))/np.linalg.norm(dofs)
        print('ERROR ',err)
    else:
        tme_gmres_classic = np.nan
        tme_stiff_classic = np.nan

    solution = tt_iga.Function(Basis+Basis_param)
    solution.dofs = dofs_tt
    err_L2 = solution.L2error(uref, geometry_map = geom, level=100) #L2_error(uref, dofs_tt, Basis+Basis_param, [Xk,Yk,Zk],level=100)
    
    err_Linf = err_L2
    
    print('Computed L2 ',err_Linf)
    # print('Energy approx ',tt.dot(uref_fun.dofs,uref_fun.dofs*int_tt))
    dct = {'err_L2' : err_L2, 'err_inf' : err_Linf, 'rank_tt': dofs_tt.R, 'rank_mat': M_tt.R, 'time_stiff': tme_stiff, 'time_solve': tme_solve}
    dct['time_solve_qtt'] = tme_solve_qtt
    dct['storage_tt'] = storage
    dct['storage_qtt'] = storageqtt
    dct['time_gmres'] = tme_gmres_classic
    dct['time_classic'] = tme_stiff_classic
    
    return dct
    
degs = [1,2,3]
ns = [6,8,10,16,20,25,32,40,60,80,90,100,120,140,160,180,256]
ns = [8,16,32,64,128]

results1 = []

for d in degs:
    for n in ns:
        print()
        print('N ',n,' , deg ',d)
        Ns = np.array([n,n,n])
        dct = solve(Ns,d,8,conventional=1, eps_solver=1e-9 if (d==3 and n>100) else 1e-8,qtt = True)
        dct['n'] = n
        dct['deg'] = d
        results1.append(dct)


df = pd.DataFrame([[el for el in v.values() ] for v in results1], columns = [k for k in results1[0]])
# dict_keys(['err_L2', 'err_inf', 'rank_tt', 'rank_mat', 'time_stiff', 'time_solve', 'time_solve_qtt', 'storage_tt', 'storage_qtt', 'time_gmres', 'time_classic', 'n', 'deg'])

# Plots 
import tikzplotlib
plt.figure()
plt.loglog(ns,df[df['deg']==1]['err_L2'].to_numpy(),'r')
plt.loglog(ns,df[df['deg']==2]['err_L2'].to_numpy(),'g')
plt.loglog(ns,df[df['deg']==3]['err_L2'].to_numpy(),'b')
plt.loglog(ns,df[df['deg']==1]['err_L2'].to_numpy()[-2]*ns[-2]**2*1/np.array(ns)**2,'k:')
plt.loglog(ns,df[df['deg']==2]['err_L2'].to_numpy()[-2]*ns[-2]**3*1/np.array(ns)**3,'k:')
plt.loglog(ns,df[df['deg']==3]['err_L2'].to_numpy()[-2]*ns[-2]**4*1/np.array(ns)**4,'k:')
plt.legend([r'linear',r'quadratic',r'cubic'])
plt.gca().set_xlabel(r'Size of univariate B-spline basis $n$')
plt.gca().set_ylabel(r'relative errors')
plt.grid(True, which="both", ls="-")
tikzplotlib.save('data/conv_err.tex')

plt.figure()
plt.loglog(ns,df[df['deg']==2]['storage_tt'].to_numpy(),'r')
plt.loglog(ns,df[df['deg']==2]['storage_qtt'].to_numpy(),'g')
plt.legend([r'TT',r'QTT '])
plt.gca().set_xlabel(r'Size of univariate B-spline basis $n$')
plt.gca().set_ylabel(r'storage [MB]')
plt.grid(True, which="both", ls="-")
tikzplotlib.save('data/conv_storage.tex')

plt.figure()
plt.loglog(ns,df[df['deg']==2]['time_solve'].to_numpy(),'r')
plt.loglog(ns,df[df['deg']==2]['time_solve_qtt'].to_numpy(),'g')
plt.loglog(ns,df[df['deg']==2]['time_gmres'].to_numpy(),'b')
plt.loglog(np.array(ns),np.array(ns)**2/200,'k:')
plt.ylabel('time [s]')
plt.xlabel(r'Size of univariate B-spline basis $n$')
plt.legend(['TT (entire parameter grid)','QTT (entire parameter grid)','GMRES (one parameter)',r'$\mathcal{O}(n^{2})$ line'])
plt.grid(True, which="both", ls="-")
tikzplotlib.save('data/conv_time.tex')

plt.figure()
plt.loglog(ns,df[df['deg']==2]['time_stiff'].to_numpy(),'r')
plt.loglog(ns,df[df['deg']==2]['time_classic'].to_numpy(),'g')
plt.ylabel('time [s]')
plt.xlabel(r'Size of univariate B-spline basis $n$')
plt.legend(['TT (entire parameter grid)','conventional (one parameter)'])
plt.grid(True, which="both", ls="-")
tikzplotlib.save('data/stiff_time.tex')



#%% interpolation level
nls = [3,4,5,6,7,8,9]
ns = [10,11,12,13,14,15,17,19,20,22,24,26,28,30,40,50,60,70,80,90,100,110,120,130,140,150]


results2 = []
for nl in nls:
    for n in ns:
        print()
        print('#####################')
        print('nl ',nl, ' n ',n)
        dct = solve(np.array([n]*3),3,nl,eps_solver=1e-10,qtt=False,conventional = False)
        dct['nl'] = nl
        dct['n'] = n
        results2.append(dct)

df2 = pd.DataFrame([[el for el in v.values() ] for v in results2], columns = [k for k in results2[0]])
    

plt.figure()
for nl in nls: plt.loglog(np.array(ns),df2[df2['nl']==nl]['err_L2'].to_numpy())
plt.legend([str(tmp) for tmp in nls])
plt.gca().set_xlabel(r'$n$')
plt.gca().set_ylabel(r'relative error')
plt.grid(True, which="both", ls="-")
tikzplotlib.save('data/conv_N_ell.tex')

plt.figure()
plt.semilogy(np.array(nls),df2[df2['n']==ns[-1]]['err_L2'].to_numpy())
plt.gca().set_xlabel(r'$#collocation points \ell$')
plt.gca().set_ylabel(r'relative error')
plt.grid(True, which="both", ls="-")
tikzplotlib.save('data/conv_ell.tex')

# plt.figure()
# plt.loglog(np.array(nls),time_stiff)
# plt.legend([r'$N='+str(tmp)+'$' for tmp in ns])
# plt.gca().set_xlabel(r'$\ell$')
# plt.gca().set_ylabel(r'time [s]')
# plt.grid(True, which="both", ls="-")

# plt.figure()
# plt.loglog(np.array(nls),time_solve)
# plt.legend([r'$N='+str(tmp)+'$' for tmp in ns])
# plt.gca().set_xlabel(r'$\ell$')
# plt.gca().set_ylabel(r'time [s]')
# plt.grid(True, which="both", ls="-")

#%% interpolation level
epss = [12-4,1e-5,1e-6,1e-7,1e-8]
ns = [16,32,64,128]
nl = 8


results3 = []
for eps in epss:
    for n in ns:
        print()
        print('#####################')
        print('eps ',eps, ' N ',n)
        dct = solve(np.array([n]*3),2,nl,eps_solver=eps,conventional = False,qtt=False)
        dct['n'] = n
        dct['eps'] = eps


df3 = pd.DataFrame([[el for el in v.values() ] for v in results3], columns = [k for k in results3[0]])

plt.figure()
for eps in epss: plt.loglog(np.array(ns),df2[df2['eps']==eps]['err_L2'].to_numpy())
plt.gca().set_xlabel(r'$n$')
plt.gca().set_ylabel(r'relative error')
plt.legend([r'$\epsilon=10^{'+str(int(np.log10(tmp)))+r'}$' for tmp in epss])
plt.grid()
tikzplotlib.save('data/conv_eps.tex')