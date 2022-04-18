"""
    
"""
#%% Imports
import torch as tn
import numpy as np
import torchtt as tntt
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
from tt_iga import Function, Geometry
from tt_iga.bspline import BSplineBasis
from tt_iga.lagrange import LagrangeLeg

tn.set_default_dtype(tn.float64)
#%% Univariate bases and space definition
N = [64]*3      # dimension fo the B-splines
deg = 2         # degree of the bsplines
nl = 8          # number of polynomials in the parameter space

baza1 = BSplineBasis(np.linspace(0,1,N[0]-deg+1),deg)
baza2 = BSplineBasis(np.linspace(0,1,N[1]-deg+1),deg)
baza3 = BSplineBasis(np.linspace(0,1,N[2]-deg+1),deg)
Basis = [baza1,baza2,baza3]
Basis_param = [LagrangeLeg(nl,[0,1])]

#%% Geometry parametrization
xc = lambda u,v: u*np.sqrt(1-v**2/2)
yc = lambda u,v: v*np.sqrt(1-u**2/2)

alpha = 0.5
xparam = lambda t : xc(t[:,0]*2-1,t[:,1]*2-1)*((1+np.cos((t[:,2]*2-1)*np.pi))*alpha*t[:,3]+1)
yparam = lambda t : yc(t[:,0]*2-1,t[:,1]*2-1)*((1+np.cos((t[:,2]*2-1)*np.pi))*alpha*t[:,3]+1)
zparam = lambda t : t[:,2]*2-1

# interpolate the geometry
geom = Geometry(Basis+Basis_param)
geom.interpolate([xparam, yparam, zparam])      

#%% Interpolate the functions
# define the functions to interpolate on the grid
sigma = 0.5
function = lambda x: tn.exp(-((x[:,0]-0.0)**2+(x[:,1]-0.0)**2+(x[:,2]-0)**2)/sigma)

kx = 2
ky = 3
function = lambda x: tn.sin(kx*x[:,0])*tn.cos(ky*x[:,1])*tn.cos(2*np.pi*x[:,2])

fun = Function(Basis+Basis_param)
fun.interpolate(function, geometry=geom)

#%% Plots

# plot geometry for the parameters {0,1}
fig = geom.plot_domain([tn.tensor([0.0])],[(0,1),(0,1),(0.0,1)],surface_color='blue', wireframe = False,alpha=0.1)
fig.gca().set_xlabel(r'$x_1$')
fig.gca().set_ylabel(r'$x_2$')
fig.gca().set_zlabel(r'$x_3$')
fig.gca().view_init(15, -60)

fig = geom.plot_domain([tn.tensor([1])],[(0,1),(0,1),(0.0,1)],surface_color='blue', wireframe = False,alpha=0.1)
fig.gca().set_xlabel(r'$x_1$')
fig.gca().set_ylabel(r'$x_2$')
fig.gca().set_zlabel(r'$x_3$')
fig.gca().view_init(15, -60)

# plot the function 
fval = fun([tn.linspace(0,1,128),tn.tensor([0.5]),tn.linspace(0,1,128),tn.tensor([1.0])]).full().numpy().squeeze()
x, y, z = geom([tn.linspace(0,1,128),tn.tensor([0.5]),tn.linspace(0,1,128),tn.tensor([1.0])])
fig = geom.plot_domain([tn.tensor([1])],[(0,1),(0,1),(0.0,1)],surface_color=None, wireframe = False,alpha=0.1)
ax = fig.gca()
C = plt.cm.jet(matplotlib.colors.Normalize(vmin=fval.min(),vmax=fval.max())(fval))
C[:,:,-1] = 1
ax.plot_surface(x.full().numpy().squeeze(),y.full().numpy().squeeze(),z.full().numpy().squeeze(),facecolors = C, antialiased=True,rcount=256,ccount=256,alpha=0.1)
fig.gca().set_xlabel(r'$x_1$')
fig.gca().set_ylabel(r'$x_2$')
fig.gca().set_zlabel(r'$x_3$')

fval = fun([tn.linspace(0,1,128),tn.linspace(0,1,128),tn.tensor([0.38]),tn.tensor([1.0])]).full().numpy().squeeze()
x, y, z = geom([tn.linspace(0,1,128),tn.linspace(0,1,128),tn.tensor([0.38]),tn.tensor([1.0])])
fig = geom.plot_domain([tn.tensor([1])],[(0,1),(0,1),(0.0,1)],surface_color=None, wireframe = False,alpha=0.1)
ax = fig.gca()
C = plt.cm.jet(matplotlib.colors.Normalize(vmin=fval.min(),vmax=fval.max())(fval))
C[:,:,-1] = 1
ax.plot_surface(x.full().numpy().squeeze(),y.full().numpy().squeeze(),z.full().numpy().squeeze(),facecolors = C, antialiased=True,rcount=256,ccount=256,alpha=0.1)
fig.gca().set_xlabel(r'$x_1$')
fig.gca().set_ylabel(r'$x_2$')
fig.gca().set_zlabel(r'$x_3$')


#%% Compute the L2 error

err = fun.L2error(function, geometry_map = geom, level = 64)
print('L2 error',err)