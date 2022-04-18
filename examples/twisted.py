import torch as tn
import torchtt as tntt 
import tt_iga
import numpy as np
import matplotlib.pyplot as plt

#%% Geomtery 
def plane_spanner(P1,P2,P3,t1,t2):
    x = (P1[:,0]-P2[:,0])*t1+(P3[:,0]-P2[:,0])*t2+P2[:,0]
    y = (P1[:,1]-P2[:,1])*t1+(P3[:,1]-P2[:,1])*t2+P2[:,1]
    z = (P1[:,2]-P2[:,2])*t1+(P3[:,2]-P2[:,2])*t2+P2[:,2]
    return x,y,z

def curve2(t,w = tn.pi,v = 3):
    phi = 0
    r = 0.5
    x = r*tn.cos(w*t+phi)
    y = r*tn.sin(w*t+phi)
    z = v * t
    return tn.hstack((tn.reshape(x,[-1,1]),tn.reshape(y,[-1,1]),tn.reshape(z,[-1,1])))

def curve1(t,w = tn.pi, v = 3):
    phi = -tn.pi/2
    r = 0.5
    x = r*tn.cos(w*t+phi)
    y = r*tn.sin(w*t+phi)
    z = v * t
    return tn.hstack((tn.reshape(x,[-1,1]),tn.reshape(y,[-1,1]),tn.reshape(z,[-1,1])))
  
def curve3(t,w = tn.pi, v = 3):
    phi = tn.pi/2
    r = 0.5
    x = r*tn.cos(w*t+phi)
    y = r*tn.sin(w*t+phi)
    z = v * t
    return tn.hstack((tn.reshape(x,[-1,1]),tn.reshape(y,[-1,1]),tn.reshape(z,[-1,1])))
  
#%% Basis
deg = 2
Ns = np.array([60,60,120])-deg+1
Ns = np.array([40,40,82])-deg+1
baza1 = tt_iga.BSplineBasis(np.linspace(0,1,Ns[0]),deg)
baza2 = tt_iga.BSplineBasis(np.linspace(0,1,Ns[1]),deg)
baza3 = tt_iga.BSplineBasis(np.concatenate((np.linspace(0,0.25,Ns[2]//4),np.linspace(0.25,0.5,Ns[2]//4),np.linspace(0.5,0.75,Ns[2]//4),np.linspace(0.75,1,Ns[2]//4-1))),deg)

Basis = [baza1,baza2,baza3]
N = [baza1.N,baza2.N,baza3.N]

#%% Interpolate the geometry parametrization
scale_mult = 1
xparam = lambda t : plane_spanner(curve1(t[:,2]),curve2(t[:,2]),curve3(t[:,2]),t[:,0],t[:,1])[0]
yparam = lambda t : plane_spanner(curve1(t[:,2]),curve2(t[:,2]),curve3(t[:,2]),t[:,0],t[:,1])[1]
zparam = lambda t : plane_spanner(curve1(t[:,2]),curve2(t[:,2]),curve3(t[:,2]),t[:,0],t[:,1])[2]

geom = tt_iga.Geometry(Basis)
geom.interpolate([xparam, yparam, zparam])

print(geom.Xs)

#%% Plot
fig = geom.plot_domain([],[(0,1),(0,1),(0.0,1)],surface_color='blue', wireframe = False,alpha=0.1,n=64)
fig.gca().set_xlabel(r'$x_1$')
fig.gca().set_ylabel(r'$x_2$')
fig.gca().set_zlabel(r'$x_3$')
fig.gca().view_init(15, -60)
fig.gca().set_box_aspect(aspect = (1,1,3))
plt.show()

