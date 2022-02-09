import torch as tn
import torchtt as tntt
import matplotlib.pyplot as plt
from tt_iga import *
import numpy as np
import datetime
import matplotlib.colors
import pandas as pd

tn.set_default_dtype(tn.float64)


Np = 8
Ns = [40,20,80]
deg = 2
nl = 8
qtt = True

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

# if tn.cuda.is_available(): 
#     tme = datetime.datetime.now() 
#     Stt = geom.stiffness_interp( eps = 1e-9, qtt = True, verb=True, device = tn.device('cuda:0'))
#     tme = datetime.datetime.now() -tme
#     print('Time stiffness matrix GPU',tme.total_seconds())
#     dct['time stiff GPU'] = tme.total_seconds()

tme = datetime.datetime.now() 
Stt = geom.stiffness_interp( eps = 1e-9, qtt =  qtt, verb=True, device = None)
tme = datetime.datetime.now() -tme
print('Time stiffness matrix ',tme.total_seconds())

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
dofs_tt = tntt.solvers.amen_solve(M_tt, rhs_tt, x0 = tntt.ones(rhs_tt.N), eps = eps_solver, nswp = 50, preconditioner = 'c',  verbose = False)
tme_amen = (datetime.datetime.now() -tme_amen).total_seconds() 
print('Time solver', tme_amen)

if tn.cuda.is_available():
    tme_amen_gpu = datetime.datetime.now()
    dofs_tt = tntt.solvers.amen_solve(M_tt.cuda(), rhs_tt.cuda(), x0 = tntt.ones(rhs_tt.N).cuda(), eps = eps_solver, nswp = 50, preconditioner = 'c', verbose = False).cpu()
    tme_amen_gpu = (datetime.datetime.now() -tme_amen_gpu).total_seconds() 
    print('Time solver GPU', tme_amen_gpu)

# save stats in the dictionary
print('Rank matrix',np.mean(M_tt.R))
print('Rank rhs',np.mean(rhs_tt.R))
print('Rank solution',np.mean(dofs_tt.R))
print('Memory stiff [MB]',tntt.numel(Stt)*8/1e6)
print('Memory mass [MB]',tntt.numel(Mass_tt)*8/1e6)
print('Memory system mat [MB]',tntt.numel(M_tt)*8/1e6)
print('Memory rhs [MB]',tntt.numel(rhs_tt)*8/1e6)
print('Memory solution [MB]',tntt.numel(dofs_tt)*8/1e6)

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

print('\nMax err %e\n\n'%(err))

random_params4plot = [2*var*(tn.rand((1))-0.5) for i in range(Np)]
u_val = fspace([tn.tensor([0.5]), tn.linspace(0,1,128), tn.linspace(0,1,128)]+random_params4plot).full()
x,y,z = geom([tn.tensor([0.5]), tn.linspace(0,1,128), tn.linspace(0,1,128)]+random_params4plot)

plt.figure()
fig = geom.plot_domain(random_params4plot, [(0,1),(0,1),(0.0,1)], surface_color=None, wireframe = False, alpha=0.1, n=64, frame_color = 'k')
ax = fig.gca()
C = u_val.numpy().squeeze()
norm = matplotlib.colors.Normalize(vmin=C.min(),vmax=C.max())
C = plt.cm.jet(norm(C))
C[:,:,-1] = 1
ax.plot_surface(x.numpy().squeeze(), y.numpy().squeeze(), z.numpy().squeeze(), edgecolors=None, linewidth=0, facecolors = C, antialiased=True, rcount=256, ccount=256, alpha=0.5)
fig.gca().set_xlabel(r'$x_1$', fontsize=14)
fig.gca().set_ylabel(r'$x_2$', fontsize=14)
fig.gca().set_zlabel(r'$x_3$', fontsize=14)
fig.gca().view_init(45, -60)
fig.gca().zaxis.set_rotate_label(False)
fig.gca().set_xticks([0,0.5,1,1.5])
fig.gca().set_yticks([-2,-1.5,-1])
fig.gca().set_zticks([0,0.5,1])
fig.gca().tick_params(axis='both', labelsize=14)
fig.gca().set_box_aspect(aspect = (1.5,1,1))

plt.figure()
fig = geom.plot_domain([tn.tensor([0.0])]*Np, [(0,1),(0,1),(0.0,1)], surface_color='blue', wireframe = False, alpha=0.1, n=64, frame_color = 'k')
for i in range(5): geom.plot_domain([2*var*(tn.rand((1))-0.5) for i in range(Np)],[(0,1),(0,1),(0.0,1)], fig = fig, surface_color=None, wireframe = False, alpha=0.1, n=64, frame_color = 'r')
fig.gca().set_xlabel(r'$x_1$', fontsize=14)
fig.gca().set_ylabel(r'$x_2$', fontsize=14)
fig.gca().set_zlabel(r'$x_3$', fontsize=14)
fig.gca().view_init(45, -60)
fig.gca().zaxis.set_rotate_label(False)
fig.gca().set_xticks([0,0.5,1,1.5])
fig.gca().set_yticks([-2,-1.5,-1])
fig.gca().set_zticks([0,0.5,1])
fig.gca().tick_params(axis='both', labelsize=14)
fig.gca().set_box_aspect(aspect = (1.5,1,1))