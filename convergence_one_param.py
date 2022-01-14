import torch as tn
import torchtt as tntt
import matplotlib.pyplot as plt
from tt_iga import *
import numpy as np
import datetime
import matplotlib.colors
import scipy.sparse
from iga_fem import *

tn.set_default_dtype(tn.float64)

def solve(Ns,deg,nl,alpha=1/4,eps_solver = 10*1e-9,eps_construction=1e-11,qtt = True,conventional = True, gpu_solver = False):
   
    baza1 = BSplineBasis(np.linspace(0,1,Ns[0]-deg+1),deg)
    baza2 = BSplineBasis(np.linspace(0,1,Ns[1]-deg+1),deg)
    baza3 = BSplineBasis(np.linspace(0,1,Ns[2]-deg+1),deg)
    
    Basis = [baza1,baza2,baza3]
    N = [baza1.N,baza2.N,baza3.N]
    
    Basis_param = [LagrangeLeg(nl,[0,1])]
    
    xc = lambda u,v: u*np.sqrt(1-v**2/2)
    yc = lambda u,v: v*np.sqrt(1-u**2/2)

    xparam = lambda t : xc(t[:,0]*2-1,t[:,1]*2-1)*((1+np.cos((t[:,2]*2-1)*np.pi))*alpha*t[:,3]+1)
    yparam = lambda t : yc(t[:,0]*2-1,t[:,1]*2-1)*((1+np.cos((t[:,2]*2-1)*np.pi))*alpha*t[:,3]+1)
    zparam = lambda t : t[:,2]*2-1

    geom = Geometry(Basis+Basis_param)
    geom.interpolate([xparam, yparam, zparam])      


    tme = datetime.datetime.now() 
    Mass_tt = geom.mass_interp(eps=1e-12)
    tme = datetime.datetime.now() -tme
    print('Time mass matrix ',tme.total_seconds())
    tme_mass = tme.total_seconds()
    
    
    tme = datetime.datetime.now() 
    Stiff_tt = geom.stiffness_interp( eps = eps_construction, qtt = False, verb=True)
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
    
    uref_fun = Function(Basis+Basis_param)
    uref_fun.interpolate(uref, geometry = geom, eps = 1e-14)
    
    f_fun = Function(Basis+Basis_param)
    f_fun.interpolate(ffun, geometry = geom , eps = 1e-14)
    

    Pin_tt, Pbd_tt = get_projectors(N,[[0,0],[0,0],[0,0]]) 
   
    
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
    dofs_tt = tntt.solvers.amen_solve(M_tt, rhs_tt, x0 = tntt.ones(rhs_tt.N), eps = eps_solver, nswp = 50, kickrank = 4, preconditioner = 'c')
    tme = datetime.datetime.now() -tme
    print('Time system solve ',tme,flush=True)
    tme_solve = tme.total_seconds()
   
    if qtt:
        M_qtt = M_tt.to_qtt()
        rhs_qtt = rhs_tt.to_qtt()
        print('Rank Mqtt ',M_qtt.R)
        print('Rank rhsqtt ',rhs_qtt.R)
    
    tme_solve_qtt = datetime.datetime.now()
    if qtt: dofs_qtt = tntt.solvers.amen_solve(M_qtt, rhs_qtt, x0 = rhs_qtt.round(1e-10,1), eps = eps_solver, nswp = 80, kickrank = 6, preconditioner='c')
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
        stiff_sparse = construct_stiff_sparse([baza1, baza2, baza3],[geom.Xs[0][:,:,:,nl//2].numpy(), geom.Xs[1][:,:,:,nl//2].full(), geom.Xs[2][:,:,:,nl//2].full()])
        tme_stiff_classic = (datetime.datetime.now() - tme_stiff_classic).total_seconds() 
        print('Stiff time ',tme_stiff_classic)
        
        Pin, Pbd = boundary_matrices([baza1, baza2, baza3])
        
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

    solution = Function(Basis+Basis_param)
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
    
    
#%% First test
degs = [1,2,3]
ns = [6,8,10,16,20,25,32,40,60,80,90,100,120,140,160,180,256]
ns = [8,16,32,64,128]
# ns = [8]
errz_L2 = []
times_S = []
times_solve =[]
times_qtt =[]
errz_Linf = []
storage_S = []
storage_solution = []
storage_solutionqtt = []
time_S_classic = []
times_gmres = []
for d in degs:
    tmp = []
    tmp2 = []
    tmp3 = []
    tmp4 = []
    tmp5 = []
    tmp6 = []
    tmp7 = []
    tmp8 = []
    tmp9 = []
    for n in ns:
        print()
        print('N ',n,' , deg ',d)
        Ns = np.array([n,n,n])
        dct = solve(Ns,d,8,conventional=0, eps_solver=1e-9 if (d==3 and n>100) else 1e-8,qtt = 1  )
        tmp2.append(dct['err_L2'])
        tmp.append(dct['err_inf'])
        tmp3.append(dct['time_stiff'])
        tmp4.append(dct['time_solve'])
        tmp5.append(dct['time_solve_qtt'])
        tmp6.append(dct['storage_tt'])
        tmp7.append(dct['storage_qtt'])
        tmp8.append(dct['time_classic'])
        tmp9.append(dct['time_gmres'])
        
    errz_Linf.append(tmp)
    errz_L2.append(tmp2)
    times_S.append(tmp3)
    times_solve.append(tmp4)
    times_qtt.append(tmp5)
    storage_solution.append(tmp6)
    storage_solutionqtt.append(tmp7)
    time_S_classic.append(tmp8)
    times_gmres.append(tmp9)
    
errz_Linf = np.array(errz_Linf)
errz_L2 = np.array(errz_L2)
times_solve = np.array(times_solve)
times_S = np.array(times_S)
times_qtt = np.array(times_qtt)
storage_solution = np.array(storage_solution)
storage_solutionqtt = np.array(storage_solutionqtt)
time_S_classic = np.array(time_S_classic)
times_gmres = np.array(times_gmres)

# plt.figure()
# plt.loglog(ns,errz_L2[0,:],'r')
# plt.loglog(ns,errz_L2[1,:],'m')
# plt.loglog(ns,errz_L2[2,:],'c')
# plt.loglog(ns,errz_L2[0,-1]*ns[-1]**2*1/np.array(ns)**2,'k:')
# plt.loglog(ns,errz_L2[1,-1]*ns[-1]**3*1/np.array(ns)**3,'k:')
# plt.loglog(ns,errz_L2[2,-1]*ns[-1]**4*1/np.array(ns)**4,'k:')

plt.figure()
plt.loglog(ns,errz_Linf[0,:],'r')
plt.loglog(ns,errz_Linf[1,:],'g')
plt.loglog(ns,errz_Linf[2,:],'b')
plt.loglog(ns,errz_Linf[0,-2]*ns[-2]**2*1/np.array(ns)**2,'k:')
plt.loglog(ns,errz_Linf[1,-2]*ns[-2]**3*1/np.array(ns)**3,'k:')
plt.loglog(ns,errz_Linf[2,-2]*ns[-2]**4*1/np.array(ns)**4,'k:')
plt.legend([r'deg 1',r'deg 2',r'deg 3'])
plt.gca().set_xlabel(r'$N$')
plt.gca().set_ylabel(r'relative errors')
plt.grid(True, which="both", ls="-")

plt.figure()
plt.loglog(ns,times_qtt[0,:],'r')
plt.loglog(ns,times_qtt[1,:],'g')
plt.loglog(ns,times_qtt[2,:],'b')
plt.loglog(ns,times_solve[0,:],'r:')
plt.loglog(ns,times_solve[1,:],'g:')
plt.loglog(ns,times_solve[2,:],'b:')
# plt.loglog(np.array(ns),np.array(ns),'k')
plt.loglog(np.array(ns),np.array(ns)**2,'k:')
# plt.loglog(np.array(ns),np.array(ns)**3,'k')
plt.grid(True, which="both", ls="-")
plt.legend([r'QTT deg 1',r'QTT deg 2',r'QTT deg 3',r'TT deg 1',r'TT deg 2',r'TT deg 3',r'$\mathcal{O}(N^{2})$ line'])
plt.gca().set_xlabel(r'$N$')
plt.gca().set_ylabel(r'time [s]')

plt.figure()
plt.loglog(ns,times_qtt[1,:],'g')
plt.loglog(ns,times_solve[1,:],'r')
plt.loglog(np.array(ns),np.array(ns)**2/100,'k:')
plt.grid(True, which="both", ls="-")
plt.legend([r'QTT deg 2',r'TT deg 2',r'$\mathcal{O}(N^{2})$ line'])
plt.gca().set_xlabel(r'$N$')
plt.gca().set_ylabel(r'time [s]')

plt.figure()
plt.loglog(ns,storage_solution[0,:],'r')
plt.loglog(ns,storage_solution[1,:],'g')
plt.loglog(ns,storage_solution[2,:],'b')
plt.legend([r'deg 1',r'deg 2',r'deg 3'])
plt.gca().set_xlabel(r'$N$')
plt.gca().set_ylabel(r'storage [MB]')
plt.grid()


plt.figure()
plt.loglog(ns,storage_solution[1,:],'r')
plt.loglog(ns,storage_solutionqtt[1,:],'g')
plt.legend([r'TT',r'QTT'])
plt.gca().set_xlabel(r'$N$')
plt.gca().set_ylabel(r'storage [MB]')
plt.grid(True, which="both", ls="-")

plt.figure()
plt.loglog(ns,times_solve[1,:],'r')
plt.loglog(ns,times_qtt[1,:],'g')
plt.loglog(ns,times_gmres[1,:],'b')
plt.loglog(np.array(ns),np.array(ns)**2/200,'k:')
plt.ylabel('time [s]')
plt.xlabel(r'$N$')
plt.legend(['TT','QTT','GMRES',r'$\mathcal{O}(N^{2})$ line'])
plt.grid(True, which="both", ls="-")

plt.figure()
plt.loglog(ns,times_S[1,:],'r')
plt.loglog(ns,time_S_classic[1,:],'g')
plt.ylabel('time [s]')
plt.xlabel(r'$N$')
plt.legend(['TT','conventional'])
plt.grid(True, which="both", ls="-")


import sys
sys.exit()


#%% interpolation level
nls = [4,8,16,32]
nls = [3,4,5,6,7,8,9]
ns = [10,11,12,13,14,15,17,19,20,22,24,26,28,30,40,50,60,70,80,90,100,110,120,130,140,150]

errz_Linf =[]
time_stiff = []
time_solve = []
for nl in nls:
    tmp = []
    tmp2 = []
    tmp3 = []
    for n in ns:
        print()
        print('#####################')
        print('nl ',nl, ' N ',n)
        dct = solve(np.array([n]*3),3,nl,eps_solver=1e-10,qtt=False,conventional = False)
        tmp.append(dct['err_inf'])
        tmp2.append(dct['time_stiff'])
        tmp3.append(dct['time_solve'])
    errz_Linf.append(tmp)
    time_stiff.append(tmp2)
    time_solve.append(tmp3)
    
time_stiff = np.array(time_stiff)
time_solve = np.array(time_solve)
errz_Linf = np.array(errz_Linf)

plt.figure()
plt.loglog(np.array(ns),errz_Linf[:,:].transpose())
plt.legend([str(tmp) for tmp in nls])
plt.gca().set_xlabel(r'$N$')
plt.gca().set_ylabel(r'relative error')
plt.grid(True, which="both", ls="-")

plt.figure()
plt.semilogy(np.array(nls),errz_Linf[:,-1])
# plt.legend([str(tmp) for tmp in nls])
plt.gca().set_xlabel(r'$\ell$')
plt.gca().set_ylabel(r'relative error')
plt.grid(True, which="both", ls="-")


plt.figure()
plt.loglog(np.array(nls),time_stiff)
plt.legend([r'$N='+str(tmp)+'$' for tmp in ns])
plt.gca().set_xlabel(r'$\ell$')
plt.gca().set_ylabel(r'time [s]')
plt.grid(True, which="both", ls="-")

plt.figure()
plt.loglog(np.array(nls),time_solve)
plt.legend([r'$N='+str(tmp)+'$' for tmp in ns])
plt.gca().set_xlabel(r'$\ell$')
plt.gca().set_ylabel(r'time [s]')
plt.grid(True, which="both", ls="-")

#%% interpolation level
epss = [10**-4]
ns = [16,32,64,128]
nl = 8
errz_Linf =[]

for eps in epss:
    tmp = []
    for n in ns:
        print()
        print('#####################')
        print('eps ',eps, ' N ',n)
        dct = solve(np.array([n]*3),2,nl,eps_solver=eps,conventional = False,qtt=False)
        tmp.append(dct['err_inf'])
    errz_Linf.append(tmp)
    
errz_Linf = np.array(errz_Linf)

plt.figure()
plt.loglog(np.array(ns),errz_Linf.T)
plt.gca().set_xlabel(r'$N$')
plt.gca().set_ylabel(r'relative error')
plt.legend([r'$\epsilon=10^{'+str(int(np.log10(tmp)))+r'}$' for tmp in epss])
plt.grid()