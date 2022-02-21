import torch as tn
import torchtt as tntt
import matplotlib.pyplot as plt
from tt_iga import *
import numpy as np
import datetime
import matplotlib.colors

tn.set_default_dtype(tn.float64)

#%% Create the bases for the space and parameters
deg = 2
Ns = np.array(3*[64])-deg+1
baza1 = BSplineBasis(np.concatenate((np.linspace(0,0.5,Ns[0]//2),np.linspace(0.5,1,Ns[0]//2))),deg)
baza2 = BSplineBasis(np.linspace(0,1,Ns[1]),deg)
baza3 = BSplineBasis(np.concatenate((np.linspace(0,0.3,Ns[2]//3),np.linspace(0.3,0.7,Ns[2]//3),np.linspace(0.7,1,Ns[2]//3))),deg)

Basis = [baza1,baza2,baza3]
N = [baza1.N,baza2.N,baza3.N]

nl = 12
Basis_param = [LagrangeLeg(nl,[-0.05,0.05])]*4

#%% Create the parametrization 

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

#%% Instantiate the Geometry object and do some plots
geom = Geometry(Basis+Basis_param)
geom.interpolate([xparam, yparam, zparam])

# plots

fig = geom.plot_domain([tn.tensor([0.05]),tn.tensor([-0.05]),tn.tensor([0.05]),tn.tensor([0.05])],[(0,1),(0,1),(0.0,1)],surface_color='blue', wireframe = False,alpha=0.1)
geom.plot_domain([tn.tensor([0.05]),tn.tensor([-0.05]),tn.tensor([0.05]),tn.tensor([0.05])],[(0.0,0.5),(0.0,1),(0.3,0.7)],fig = fig,surface_color='green',wireframe = False)
fig.gca().zaxis.set_rotate_label(False)
fig.gca().set_xlabel(r'$x_1$', fontsize=14)
fig.gca().set_ylabel(r'$x_2$', fontsize=14)
fig.gca().set_zlabel(r'$x_3$', fontsize=14)
fig.gca().set_xticks([-1, 0, 1])
fig.gca().set_yticks([-1, 0, 1])
fig.gca().set_zticks([0, 0.5, 1])
fig.gca().view_init(15, -60)
fig.gca().tick_params(axis='both', labelsize=14)
plt.savefig('./data/cylinder_material1.pdf')

fig = geom.plot_domain([tn.tensor([0.05]),tn.tensor([0.05]),tn.tensor([0.05]),tn.tensor([0.05])],[(0,1),(0,1),(0.0,1)],surface_color='blue', wireframe = False,alpha=0.1)
geom.plot_domain([tn.tensor([0.05]),tn.tensor([0.05]),tn.tensor([0.05]),tn.tensor([0.05])],[(0.0,0.5),(0.0,1),(0.3,0.7)],fig = fig,surface_color='green',wireframe = False)
fig.gca().zaxis.set_rotate_label(False)
fig.gca().set_xlabel(r'$x_1$', fontsize=14)
fig.gca().set_ylabel(r'$x_2$', fontsize=14)
fig.gca().set_zlabel(r'$x_3$', fontsize=14)
fig.gca().set_xticks([-1, 0, 1])
fig.gca().set_yticks([-1, 0, 1])
fig.gca().set_zticks([0, 0.5, 1])
fig.gca().view_init(15, -60)
fig.gca().tick_params(axis='both', labelsize=14)
plt.savefig('./data/cylinder_material2.pdf')

#%% Construct the mass and stiffness TT operators and the zeros rhs
tme = datetime.datetime.now() 
Mass_tt = geom.mass_interp(eps=1e-11)
tme = datetime.datetime.now() -tme
print('Time mass matrix ',tme.total_seconds())

tme = datetime.datetime.now() 
Stt = geom.stiffness_interp( func=None, func_reference = sigma_ref, qtt = False, verb=True)
tme = datetime.datetime.now() -tme
print('Time stiffness matrix ',tme.total_seconds())

f_tt = tntt.zeros(Stt.N)

# incorporate the boundary conditions and construct the system tensor operator
Pin_tt,Pbd_tt = get_projectors(N,[[1,1],[1,1],[0,0]])
# Pbd_tt = (1/N[0]) * Pbd_tt
U0 = 10

Pin_tt = Pin_tt ** tntt.eye([nl]*4)
Pbd_tt = Pbd_tt ** tntt.eye([nl]*4)


tmp = tn.zeros(N, dtype = tn.float64)
tmp[:,:,0] = U0 

g_tt = Pbd_tt @ (tntt.TT(tmp) ** tntt.ones([nl]*4))


M_tt = Pin_tt@Stt@Pin_tt + Pbd_tt
rhs_tt = Pin_tt @ (Mass_tt @ f_tt - Stt@Pbd_tt@g_tt).round(1e-12) + g_tt
M_tt = M_tt.round(1e-9)
# print(M_tt,rhs_tt)


#%% solve in the TT format
eps_solver = 1e-6

print('Solving in TT...')
tme_amen = datetime.datetime.now() 
dofs_tt = tntt.solvers.amen_solve(M_tt.cuda(), rhs_tt.cuda(), x0 = tntt.ones(rhs_tt.N).cuda(), eps = eps_solver, nswp=40, kickrank=4).cpu()
tme_amen = (datetime.datetime.now() -tme_amen).total_seconds() 
print('Time system solve in TT ',tme_amen)

# print(dofs_tt)

#%% plots
fspace = Function(Basis+Basis_param)
fspace.dofs = dofs_tt

fval = fspace([tn.linspace(0,1,128),tn.tensor([0.5]),tn.linspace(0,1,128),tn.tensor([0.05]),tn.tensor([0.05]),tn.tensor([0.05]),tn.tensor([0.05])])
x,y,z = geom([tn.linspace(0,1,128),tn.tensor([0.5]),tn.linspace(0,1,128),tn.tensor([0.05]),tn.tensor([0.05]),tn.tensor([0.05]),tn.tensor([0.05])])

plt.figure()
plt.contour(x.full().numpy().squeeze(),z.full().numpy().squeeze(),fval.full().numpy().squeeze(), levels = 128)
plt.colorbar()

plt.figure()
plt.contourf(x.full().numpy().squeeze(),z.full().numpy().squeeze(),fval.full().numpy().squeeze(), levels = 128)
plt.colorbar()

from matplotlib import cm
fig = geom.plot_domain([tn.tensor([0.05])]*4,[(0,1),(0,1),(0.0,1)],surface_color=None, wireframe = False,frame_color='k')
geom.plot_domain([tn.tensor([0.05])]*4,[(0.0,0.5),(0.0,1),(0.3,0.7)],fig = fig,surface_color=None,wireframe = False,frame_color='k')

ax = fig.gca()
C = fval.full().numpy().squeeze()
norm = matplotlib.colors.Normalize(vmin=C.min(),vmax=C.max())
C = plt.cm.jet(norm(C))
C[:,:,-1] = 1
ax.plot_surface(x.full().numpy().squeeze(),y.full().numpy().squeeze(),z.full().numpy().squeeze(),facecolors = C, antialiased=True,rcount=256,ccount=256,alpha=0.1)

fig.gca().set_xlabel(r'$x_1$')
fig.gca().set_ylabel(r'$x_2$')
fig.gca().set_zlabel(r'$x_3$')

# fig = plt.figure(figsize = (14, 9))
# ax = plt.axes(projection = '3d')
# ax.plot_surface(x.full().squeeze(), z.full().squeeze(), fval.full().squeeze(), facecolors = C)

