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

#%% cylinder woth material in physical domain
deg = 2
Ns = np.array(3*[64])-deg+1
baza1 = tt_iga.BSplineBasis(np.concatenate((np.linspace(0.0, 0.5, Ns[0]//2), np.linspace(0.5, 1, Ns[0]//2))), deg)
baza2 = tt_iga.BSplineBasis(np.concatenate((np.linspace(0.0, 0.5, Ns[1]//2), np.linspace(0.5, 1, Ns[1]//2))), deg)
baza3 = tt_iga.BSplineBasis(np.concatenate((np.linspace(0.0, 0.5, Ns[2]//2), np.linspace(0.5, 1, Ns[2]//2))), deg)

Basis = [baza1, baza2, baza3]
N = [baza1.N, baza2.N, baza3.N]

nl = 8
Basis_param = [tt_iga.LagrangeLeg(nl, [-0.2, 0.2])]


def xc(u, v): return u*np.sqrt(1-v**2/2)
def yc(u, v): return v*np.sqrt(1-u**2/2)

def line(t, a, b): return t*(b-a)+a

def scaling(z, theta1):
    a = 0.5
    s = (z < a)*line(z/a, 0, a+theta1)
    s += np.logical_and(z >= a, z <= 1)*line((z-a)/(1-a), a+theta1, 1)
    return s


angle_mult = 1/1
def xparam(t): return xc(t[:, 0]*2-1, t[:, 1]*2-1)
def yparam(t): return yc(t[:, 0]*2-1, t[:, 1]*2-1)
def zparam(t): return scaling(t[:, 2], t[:, 3])


geom = tt_iga.Geometry(Basis+Basis_param)
geom.interpolate([xparam, yparam, zparam])


fig = geom.plot_domain([tn.tensor([0.2])], [(0, 1), (0, 1), (0.0, 1)], surface_color=None, wireframe=False, frame_color='r', alpha=1, n=32)
geom.plot_domain([tn.tensor([0.2])], [(0, 0.5), (0, 0.5), (0.0, 0.5)], fig=fig, surface_color='yellow', wireframe=False, frame_color='r', alpha=.1, n=32)
fig.gca().set_xlabel(r'$x_1$', fontsize=14)
fig.gca().set_ylabel(r'$x_2$', fontsize=14)
fig.gca().set_zlabel(r'$x_3$', fontsize=14)
fig.gca().view_init(15, -60)
fig.gca().zaxis.set_rotate_label(False)
fig.gca().set_xticks([-1, 0, 1])
fig.gca().set_yticks([-1, 0, 1])
fig.gca().set_zticks([0, 0.5, 1])
fig.gca().tick_params(axis='both', labelsize=14)

fig = geom.plot_domain([tn.tensor([-0.2])], [(0, 1), (0, 1), (0.0, 1)], surface_color=None, wireframe=False, frame_color='r', alpha=1, n=32)
geom.plot_domain([tn.tensor([-0.2])], [(0, 0.5), (0, 0.5), (0.0, 0.5)], fig=fig, surface_color='yellow', wireframe=False, frame_color='r', alpha=.1, n=32)
fig.gca().set_xlabel(r'$x_1$', fontsize=14)
fig.gca().set_ylabel(r'$x_2$', fontsize=14)
fig.gca().set_zlabel(r'$x_3$', fontsize=14)
fig.gca().view_init(15, -60)
fig.gca().zaxis.set_rotate_label(False)
fig.gca().set_xticks([-1, 0, 1])
fig.gca().set_yticks([-1, 0, 1])
fig.gca().set_zticks([0, 0.5, 1])
fig.gca().tick_params(axis='both', labelsize=14)
plt.savefig('./data/jump_domain2.pdf')

def xparam(t): return t[:, 0]
def yparam(t): return t[:, 1]
def zparam(t): return t[:, 2]


geom = tt_iga.Geometry(Basis+Basis_param)
geom.interpolate([xparam, yparam, zparam])


fig = geom.plot_domain([tn.tensor([0.2])], [(0, 1), (0, 1), (0.0, 1)], surface_color=None, wireframe=False, frame_color='r', alpha=1, n=32)
geom.plot_domain([tn.tensor([0.2])], [(0, 0.5), (0, 0.5), (0.0, 0.5)], fig=fig, surface_color='yellow', wireframe=False, frame_color='r', alpha=.1, n=32)
fig.gca().set_xlabel(r'$y_1$', fontsize=14)
fig.gca().set_ylabel(r'$y_2$', fontsize=14)
fig.gca().set_zlabel(r'$y_3$', fontsize=14)
fig.gca().view_init(15, -60)
fig.gca().zaxis.set_rotate_label(False)
fig.gca().set_xticks([0, 0.5, 1])
fig.gca().set_yticks([0, 0.5, 1])
fig.gca().set_zticks([0, 0.5, 1])
fig.gca().tick_params(axis='both', labelsize=14)
plt.savefig('./data/jump_reference.pdf')

#%% Convergence shape 
baza1 = tt_iga.BSplineBasis(np.linspace(0,1,Ns[0]-deg+1),deg)
baza2 = tt_iga.BSplineBasis(np.linspace(0,1,Ns[1]-deg+1),deg)
baza3 = tt_iga.BSplineBasis(np.linspace(0,1,Ns[2]-deg+1),deg)

Basis = [baza1,baza2,baza3]
N = [baza1.N,baza2.N,baza3.N]

Basis_param = [tt_iga.LagrangeLeg(nl,[0,1])]

xc = lambda u,v: u*np.sqrt(1-v**2/2)
yc = lambda u,v: v*np.sqrt(1-u**2/2)

xparam = lambda t : xc(t[:,0]*2-1,t[:,1]*2-1)*((1+np.cos((t[:,2]*2-1)*np.pi))*1/4*t[:,3]+1)
yparam = lambda t : yc(t[:,0]*2-1,t[:,1]*2-1)*((1+np.cos((t[:,2]*2-1)*np.pi))*1/4*t[:,3]+1)
zparam = lambda t : t[:,2]*2-1

geom = tt_iga.Geometry(Basis+Basis_param)
geom.interpolate([xparam, yparam, zparam])      

fig = geom.plot_domain([tn.tensor([0.0])], [(0, 1), (0, 1), (0.0, 1)], surface_color='blue', wireframe=False, frame_color='r', alpha=0.1, n=32)
fig.gca().set_xlabel(r'$x_1$', fontsize=14)
fig.gca().set_ylabel(r'$x_2$', fontsize=14)
fig.gca().set_zlabel(r'$x_3$', fontsize=14)
fig.gca().view_init(15, -60)
fig.gca().zaxis.set_rotate_label(False)
fig.gca().set_xticks([-1, 0, 1])
fig.gca().set_yticks([-1, 0, 1])
fig.gca().set_zticks([-1, 0, 1])
fig.gca().tick_params(axis='both', labelsize=14)
plt.savefig('./data/cylinder.pdf')

fig = geom.plot_domain([tn.tensor([1.0])], [(0, 1), (0, 1), (0.0, 1)], surface_color='blue', wireframe=False, frame_color='r', alpha=0.1, n=32)
fig.gca().set_xlabel(r'$x_1$', fontsize=14)
fig.gca().set_ylabel(r'$x_2$', fontsize=14)
fig.gca().set_zlabel(r'$x_3$', fontsize=14)
fig.gca().view_init(15, -60)
fig.gca().zaxis.set_rotate_label(False)
fig.gca().set_xticks([-1.5, 0, 1.5])
fig.gca().set_yticks([-1.5, 0, 1.5])
fig.gca().set_zticks([-1, 0, 1])
fig.gca().tick_params(axis='both', labelsize=14)
plt.savefig('./data/cylinder_deformed.pdf')