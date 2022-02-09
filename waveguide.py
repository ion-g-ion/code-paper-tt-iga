import torch as tn
import torchtt as tntt
import matplotlib.pyplot as plt
import tt_iga
import numpy as np
import datetime
import matplotlib.colors

tn.set_default_dtype(tn.float64)


deg = 2
Ns = np.array([60,60,120])-deg+1
Ns = np.array([40,40,82])-deg+1
baza1 = tt_iga.bspline.BSplineBasis(np.linspace(0,1,Ns[0]),deg)
baza2 = tt_iga.bspline.BSplineBasis(np.linspace(0,1,Ns[1]),deg)
baza3 = tt_iga.bspline.BSplineBasis(np.concatenate((np.linspace(0,0.25,Ns[2]//4),np.linspace(0.25,0.5,Ns[2]//4),np.linspace(0.5,0.75,Ns[2]//4),np.linspace(0.75,1,Ns[2]//4-1))),deg)

Basis = [baza1,baza2,baza3]
N = [baza1.N,baza2.N,baza3.N]

nl = 8
Basis_param = [tt_iga.lagrange.LagrangeLeg(nl,[-0.2,0.2])]*2+[tt_iga.lagrange.LagrangeLeg(nl,[-0.3,0.3])]


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
geom = tt_iga.Geometry(Basis+Basis_param)
geom.interpolate([xparam, yparam, zparam])

# plot
fig = geom.plot_domain([tn.tensor([0.0]),tn.tensor([-0.0]),tn.tensor([0.0]),tn.tensor([0.1])],[(0,1),(0,1),(0.0,1)],surface_color='blue', wireframe = False,alpha=0.1,n=64)
fig.gca().set_xlabel(r'$x_1$', fontsize=14)
fig.gca().set_ylabel(r'$x_2$', fontsize=14)
fig.gca().set_zlabel(r'$x_3$', fontsize=14)
fig.gca().view_init(15, -60)
fig.gca().zaxis.set_rotate_label(False)
fig.gca().set_xticks([-1.5, 0, 1.5])
fig.gca().set_yticks([0,1])
fig.gca().set_zticks([-3,-1.5,0,1.5])
fig.gca().tick_params(axis='both', labelsize=14)
fig.gca().set_box_aspect(aspect = (3,1.5,4.5))

fig = geom.plot_domain([tn.tensor([0.0]), tn.tensor([-0.0]), tn.tensor([0.0])],[(0,1),(0,1),(0.0,1)],surface_color=None, wireframe = False, alpha = 0.1, n = 64, frame_color = 'k')
geom.plot_domain([tn.tensor([Basis_param[0].interval[0]]),tn.tensor([Basis_param[1].interval[0]]), tn.tensor([Basis_param[2].interval[0]]) ],[(0,1),(0,1),(0.0,1)],fig = fig,surface_color='blue', wireframe = False,alpha=0.01,n=64)
geom.plot_domain([tn.tensor([Basis_param[0].interval[0]]),tn.tensor([Basis_param[1].interval[0]]), tn.tensor([Basis_param[2].interval[1]]) ],[(0,1),(0,1),(0.0,1)],fig = fig,surface_color='blue', wireframe = False,alpha=0.01,n=64)
geom.plot_domain([tn.tensor([Basis_param[0].interval[0]]),tn.tensor([Basis_param[1].interval[1]]), tn.tensor([Basis_param[2].interval[0]]) ],[(0,1),(0,1),(0.0,1)],fig = fig,surface_color='blue', wireframe = False,alpha=0.01,n=64)
geom.plot_domain([tn.tensor([Basis_param[0].interval[0]]),tn.tensor([Basis_param[1].interval[1]]), tn.tensor([Basis_param[2].interval[1]]) ],[(0,1),(0,1),(0.0,1)],fig = fig,surface_color='blue', wireframe = False,alpha=0.01,n=64)
geom.plot_domain([tn.tensor([Basis_param[0].interval[1]]),tn.tensor([Basis_param[1].interval[0]]), tn.tensor([Basis_param[2].interval[0]]) ],[(0,1),(0,1),(0.0,1)],fig = fig,surface_color='blue', wireframe = False,alpha=0.01,n=64)
geom.plot_domain([tn.tensor([Basis_param[0].interval[1]]),tn.tensor([Basis_param[1].interval[0]]), tn.tensor([Basis_param[2].interval[1]]) ],[(0,1),(0,1),(0.0,1)],fig = fig,surface_color='blue', wireframe = False,alpha=0.01,n=64)
geom.plot_domain([tn.tensor([Basis_param[0].interval[1]]),tn.tensor([Basis_param[1].interval[1]]), tn.tensor([Basis_param[2].interval[0]]) ],[(0,1),(0,1),(0.0,1)],fig = fig,surface_color='blue', wireframe = False,alpha=0.01,n=64)
geom.plot_domain([tn.tensor([Basis_param[0].interval[1]]),tn.tensor([Basis_param[1].interval[1]]), tn.tensor([Basis_param[2].interval[1]]) ],[(0,1),(0,1),(0.0,1)],fig = fig,surface_color='blue', wireframe = False,alpha=0.01,n=64)
fig.gca().set_xlabel(r'$x_1$', fontsize=14)
fig.gca().set_ylabel(r'$x_2$', fontsize=14)
fig.gca().set_zlabel(r'$x_3$', fontsize=14)
fig.gca().view_init(15, -60)
fig.gca().zaxis.set_rotate_label(False)
fig.gca().set_xticks([-1.5, 0, 1.5])
fig.gca().set_yticks([0,1])
fig.gca().set_zticks([-3,-1.5,0,1.5])
fig.gca().tick_params(axis='both', labelsize=14)
fig.gca().set_box_aspect(aspect = (3,1.5,4.5))
plt.savefig('./data/wg_params.pdf')

#%% Construct matrices
tme = datetime.datetime.now() 
Mass_tt = geom.mass_interp(eps=1e-11)
tme = datetime.datetime.now() -tme
print('Time mass matrix ',tme.total_seconds())

tme = datetime.datetime.now() 
Stt = geom.stiffness_interp( func=None, func_reference = None, qtt = False, verb=False)
tme = datetime.datetime.now() -tme
print('Time stiffness matrix ',tme.total_seconds())


Pin_tt,Pbd_tt = tt_iga.projectors.get_projectors(N,[[0,0],[0,0],[0,0]])
# Pbd_tt = (1/N[0]) * Pbd_tt

Pin_tt = Pin_tt ** tntt.eye([nl]*3)
Pbd_tt = Pbd_tt ** tntt.eye([nl]*3)


f_tt = tntt.zeros(Stt.N)

excitation_dofs = tt_iga.Function(Basis).interpolate(lambda t: tn.sin(t[:,0]*np.pi)*tn.sin(t[:,1]*np.pi))
tmp = tn.zeros(N)
tmp[:,:,0] = excitation_dofs[:,:,0].full()
g_tt = Pbd_tt@ (tntt.TT(tmp) ** tntt.ones([nl]*3))


k = 49

eps_solver = 1e-7
M_tt = (Pin_tt@(Stt-k*Mass_tt)+Pbd_tt).round(1e-12)
rhs_tt = (Pbd_tt @ g_tt).round(1e-12)

M_tt = M_tt.round(1e-11)

# M_qtt = ttm2qttm(M_tt).round(1e-9)
# rhs_qtt = tt2qtt(rhs_tt)
cuda = True
print('Solving in TT...')
tme_amen = datetime.datetime.now() 
if cuda and tn.cuda.is_available():
    dofs_tt = tntt.solvers.amen_solve(M_tt.cuda(), rhs_tt.cuda(), x0 = tntt.ones(rhs_tt.N).cuda(), eps = eps_solver, nswp=40, kickrank=2, verbose=True, preconditioner = 'c', local_iterations=24, resets=10).cpu()
else:
    dofs_tt = tntt.solvers.amen_solve(M_tt, rhs_tt, x0 = tntt.ones(rhs_tt.N), eps = eps_solver, nswp=40, kickrank=2, verbose=True, preconditioner = 'c', local_iterations=24, resets=10)
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


fspace = tt_iga.Function(Basis+Basis_param)
fspace.dofs = dofs_tt

fval = fspace([tn.linspace(0,1,128),tn.tensor([0.5]),tn.linspace(0,1,128),tn.tensor([-0.1]),tn.tensor([-0.1]),tn.tensor([-0.1]),tn.tensor([-0.1])])
x,y,z =  geom([tn.linspace(0,1,128),tn.tensor([0.5]),tn.linspace(0,1,128),tn.tensor([-0.1]),tn.tensor([-0.1]),tn.tensor([-0.1]),tn.tensor([-0.1])])

plt.figure()
plt.contour(x.numpy().squeeze(),z.numpy().squeeze(),fval.numpy().squeeze(), levels = 128)
plt.colorbar()

plt.figure()
plt.contourf(x.numpy().squeeze(),z.numpy().squeeze(),fval.numpy().squeeze(), levels = 128)
plt.colorbar()


fval = fspace([tn.linspace(0,1,128),tn.tensor([0.5]),tn.linspace(0,1,128),tn.tensor([0.0]),tn.tensor([0.0]),tn.tensor([0.0]),tn.tensor([0.0])])
x,y,z =  geom([tn.linspace(0,1,128),tn.tensor([0.5]),tn.linspace(0,1,128),tn.tensor([0.0]),tn.tensor([0.0]),tn.tensor([0.0]),tn.tensor([0.0])])

plt.figure()
plt.contour(x.numpy().squeeze(),z.numpy().squeeze(),fval.numpy().squeeze(), levels = 128)
plt.colorbar()

plt.figure()
plt.contourf(x.numpy().squeeze(),z.numpy().squeeze(),fval.numpy().squeeze(), levels = 128)
plt.colorbar()


fig = geom.plot_domain([tn.tensor([0.0])]*4,[(0,1),(0,1),(0.0,1)],surface_color=None, wireframe = False,frame_color='k',n = 64)
geom.plot_domain([tn.tensor([0.0])]*4,[(0,1),(0,1),(0.25,0.5)],fig=fig,surface_color=None, wireframe = False,frame_color='k',n = 64)
geom.plot_domain([tn.tensor([0.0])]*4,[(0,1),(0,1),(0.75,1)],fig=fig,surface_color=None, wireframe = False,frame_color='k',n = 64)

ax = fig.gca()
C = fval.numpy().squeeze()
norm = matplotlib.colors.Normalize(vmin=C.min(),vmax=C.max())
C = plt.cm.jet(norm(C))
C[:,:,-1] = 1
ax.plot_surface(x.numpy().squeeze(), y.numpy().squeeze(), z.numpy().squeeze(), edgecolors=None, linewidth=0, facecolors = C, antialiased=True, rcount=256, ccount=256, alpha=0.5)
fig.gca().set_xlabel(r'$x_1$', fontsize=14)
fig.gca().set_ylabel(r'$x_2$', fontsize=14)
fig.gca().set_zlabel(r'$x_3$', fontsize=14)
fig.gca().view_init(15, -60)
fig.gca().zaxis.set_rotate_label(False)
fig.gca().set_xticks([-1.5, 0, 1.5])
fig.gca().set_yticks([0,1])
fig.gca().set_zticks([-3,-1.5,0,1.5])
fig.gca().tick_params(axis='both', labelsize=14)
fig.gca().set_box_aspect(aspect = (3,1.5,4.5))
plt.savefig('./data/wg_solution.pdf')
