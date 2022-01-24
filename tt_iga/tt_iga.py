import torch as tn
import torchtt as tntt
import numpy as np
from .aux_functions import *
import matplotlib.pyplot as plt
import datetime


class Function():
      
    def __init__(self, basis):
        """
        

        Args:
            basis ([type]): [description]
        """
        self.N = [b.N for b in basis]
        self.basis = basis
        
    def interpolate(self, function, geometry = None, eps = 1e-12):
        """
        

        Args:
            function ([type]): [description]
            geometry ([type], optional): [description]. Defaults to None.
            eps ([type], optional): [description]. Defaults to 1e-12.

        Returns:
            [type]: [description]
        """
        Xg = [tn.tensor(b.interpolating_points()[0], dtype = tn.float64) for b in self.basis]
        Mg = [tn.tensor(b.interpolating_points()[1], dtype = tn.float64) for b in self.basis]
        
        corz = [tn.reshape(Mg[i].t(), [1,Mg[i].shape[0],-1,1]) for i in range(len(Mg))]
        Gm = tntt.TT(corz)
        
        if geometry == None:
            X = tntt.TT(Xg[0])**tntt.ones(self.N[1:])   
            Y = tntt.ones(self.N[:1]) ** tntt.TT(Xg[1]) ** tntt.ones(self.N[2:]) 
            Z = tntt.ones(self.N[:2]) ** tntt.TT(Xg[2]) ** tntt.ones(self.N[3:])  
        else:
            X,Y,Z = geometry(Xg)

        if len(self.basis)==3:
            evals = tntt.interpolate.function_interpolate(function, [X, Y, Z], eps)
        else:
            Np = len(self.basis[3:])
            meshgrid = tntt.meshgrid([x for x in Xg[3:]])
            meshgrid = [X,Y,Z] + [tntt.ones([n for n in self.N[:3]])**m for m in meshgrid]
            evals = tntt.interpolate.function_interpolate(function, meshgrid, eps, verbose = False)
            

        dofs = tntt.solvers.amen_solve(Gm,evals,x0 = evals,eps = eps,verbose = False)
        self.dofs = dofs
        
        return dofs
    
    def __call__(self, x, deriv = None):
        
        if deriv == None:
            deriv = [False]*len(self.N)
        
        Bs = [tn.tensor(self.basis[i](x[i].numpy(),derivative=deriv[i]), dtype = tn.float64).t() for i in range(len(self.N))]
        B_tt = tntt.TT([tn.reshape(m,[1,m.shape[0],-1,1]) for m in Bs])
        
        val = B_tt @ self.dofs
        
        return val 
    
    def L2error(self, function, geometry_map = None, level = 32):

        pts, ws = np.polynomial.legendre.leggauss(level)
        pts = (pts+1)*0.5
        ws = ws/2
        
        Xg = [tn.tensor(b.interpolating_points()[0], dtype = tn.float64) for b in self.basis]
         
        if geometry_map != None:
            X,Y,Z = geometry_map([tn.tensor(pts)]*len(self.N))
            Og_tt = geometry_map.eval_omega([tn.tensor(pts)]*3, interp=True)
        else:
            X,Y,Z = geometry_map(Xg)
        
        B_tt = tntt.TT(self.basis[0](pts),shape = [(self.basis[0].N,pts.size)]) ** tntt.TT(self.basis[1](pts),shape = [(self.basis[1].N,pts.size)]) ** tntt.TT(self.basis[2](pts),shape = [(self.basis[2].N,pts.size)]) 
        B_tt = B_tt.t()
        C_tt = tntt.eye(B_tt.M)
        
        for i in range(3,len(self.basis)):
            B_tt = B_tt ** tntt.TT(self.basis[i](pts).transpose(), shape = [(pts.size, self.basis[i].N)])
            C_tt = C_tt ** tntt.TT(self.basis[i](pts).transpose(), shape = [(pts.size, self.basis[i].N)])
        # X = C_tt @ X
        # Y = C_tt @ Y
        # Z = C_tt @ Z
        Og_tt = C_tt @ Og_tt
        d_eval = B_tt @ self.dofs
        
        if len(self.basis)==3:
            f_eval = tntt.interpolate.function_interpolate(function, [X, Y, Z], 1e-13)
        else:
            Np = len(self.N[3:])
            meshgrid = tntt.meshgrid([tn.tensor(pts)]*Np)
            meshgrid = [X,Y,Z] + [tntt.ones([pts.size]*3)**m for m in meshgrid]
            f_eval = tntt.interpolate.function_interpolate(function, meshgrid, 1e-13)
        
        diff = f_eval-d_eval
        Ws = tntt.rank1TT(len(self.basis)*[tn.tensor(ws)])
        
        integral = np.abs(tntt.dot(Ws*diff,diff*Og_tt).numpy())
        # print(integral)
        return np.sqrt(integral)
    
class Geometry():
    
    def __init__(self, basis, Xs = None):
        """
        

        Args:
            basis ([type]): [description]
            Xs ([type], optional): [description]. Defaults to None.
        """
        self.N = [b.N for b in basis]
        self.basis = basis
        self.Xs = Xs
        
    def interpolate(self, geometry_map, eps = 1e-13):
        """
        Interpolates the given geometry map.

        Args:
            geometry_map (list[function]): [description]
            eps ([type], optional): [description]. Defaults to 1e-13.
        """
        Xg = [tn.tensor(b.interpolating_points()[0], dtype = tn.float64) for b in self.basis]
        Mg = [tn.tensor(b.interpolating_points()[1], dtype = tn.float64) for b in self.basis]
        
        corz = [tn.reshape(tn.linalg.inv(Mg[i]).t(), [1,Mg[i].shape[0],-1,1]) for i in range(len(Mg))]
        Gmi = tntt.TT(corz)
        
        Xs = []
        
        for i in range(3):
            evals = tntt.interpolate.function_interpolate(geometry_map[i], tntt.meshgrid(Xg), eps = eps).round(eps)
            dofs = (Gmi @ evals).round(eps)
            Xs.append(dofs)
            
        self.Xs = Xs
        
    def __call__(self, x, deriv = None):
        
        if deriv == None:
            deriv = [False] * len(self.N)
            
        Bs = [tn.tensor(self.basis[i](x[i].numpy(),derivative=deriv[i]), dtype = tn.float64).t() for i in range(len(self.N))]

        B_tt = tntt.TT([tn.reshape(m,[1,m.shape[0],-1,1]) for m in Bs])
        
        ret = []
        
        for X in self.Xs:
            ret.append(B_tt @ X)
        
        return ret

    def eval_omega(self, x, eps = 1e-12, interp = True):
        
        #evgaluate univariate bsplines and their derivative 
        B1 = tntt.TT(tn.tensor(self.basis[0](x[0]).transpose()), [(x[0].shape[0],self.N[0])])
        dB1 = tntt.TT(tn.tensor(self.basis[0](x[0],derivative=True).transpose()), [(x[0].shape[0],self.N[0])])
        B2 = tntt.TT(tn.tensor(self.basis[1](x[1]).transpose()), [(x[1].shape[0],self.N[1])])
        dB2 = tntt.TT(tn.tensor(self.basis[1](x[1],derivative=True).transpose()), [(x[1].shape[0],self.N[1])])
        B3 = tntt.TT(tn.tensor(self.basis[2](x[2]).transpose()), [(x[2].shape[0],self.N[2])])
        dB3 = tntt.TT(tn.tensor(self.basis[2](x[2],derivative=True).transpose()), [(x[2].shape[0],self.N[2])])
       
        if interp:
            lst =  [tn.reshape(tn.eye(self.N[k], dtype = tn.float64), [1,-1,self.N[k],1]) for k in range(3,len(self.N))]
            if len(lst)>0:
                Bp = tntt.TT(lst)       
            else:
                Bp = None 
        else:
            lst =  [tn.tensor(self.basis[k](x[k]).transpose().reshape([1,-1,self.basis[k].N,1])) for k in range(3,len(self.N))]
            if len(lst)>0:
                Bp = tntt.TT(lst)       
            else:
                Bp = None
            
        Btt = dB1 ** B2 ** B3
        det1 = (Btt ** Bp) @ self.Xs[0] 
        Btt = B1 ** dB2 ** B3
        det1 *= (Btt ** Bp) @ self.Xs[1] 
        Btt = B1 ** B2 ** dB3
        det1 *= (Btt ** Bp)@ self.Xs[2] 
        
        Btt = B1 ** dB2 ** B3
        det2 = (Btt ** Bp) @ self.Xs[0] 
        Btt = B1 ** B2 ** dB3
        det2 *= (Btt ** Bp) @ self.Xs[1] 
        Btt = dB1 ** B2 ** B3
        det2 *= (Btt ** Bp)@ self.Xs[2] 
        
        Btt = B1 ** B2 ** dB3
        det3 = (Btt ** Bp) @ self.Xs[0] 
        Btt = dB1 ** B2 ** B3
        det3 *= (Btt ** Bp) @ self.Xs[1] 
        Btt = B1 ** dB2 ** B3
        det3 *= (Btt ** Bp)@ self.Xs[2] 
        
        Btt = B1 ** B2 ** dB3
        det4 = (Btt ** Bp) @ self.Xs[0] 
        Btt = B1 ** dB2 ** B3
        det4 *= (Btt ** Bp) @ self.Xs[1] 
        Btt = dB1 ** B2 ** B3
        det4 *= (Btt ** Bp)@ self.Xs[2] 
        
        Btt = dB1 ** B2 ** B3
        det5 = (Btt ** Bp) @ self.Xs[0] 
        Btt = B1 ** B2 ** dB3
        det5 *= (Btt ** Bp) @ self.Xs[1] 
        Btt = B1 ** dB2 ** B3
        det5 *= (Btt ** Bp)@ self.Xs[2] 
        
        Btt = B1 ** dB2 ** B3
        det6 = (Btt ** Bp) @ self.Xs[0] 
        Btt = dB1 ** B2 ** B3
        det6 *= (Btt ** Bp) @ self.Xs[1] 
        Btt = B1 ** B2 ** dB3
        det6 *= (Btt ** Bp)@ self.Xs[2] 
       
        res = (det1 + det2 + det3 - det4 - det5 - det6).round(eps)
        return res 

    def mass_interp(self, eps = 1e-12):
        """
        

        Args:
            eps ([type], optional): [description]. Defaults to 1e-12.

        Returns:
            [type]: [description]
        """
        p1, w1 = points_basis(self.basis[0])
        p2, w2 = points_basis(self.basis[1])
        p3, w3 = points_basis(self.basis[2])
    
        
        cores = self.eval_omega([tn.tensor(p1),tn.tensor(p2),tn.tensor(p3)], eps).cores
                
        
        cores[0] = tn.einsum('rjs,j,mj,nj->rmns',cores[0],tn.tensor(w1),tn.tensor(self.basis[0](p1)),tn.tensor(self.basis[0](p1)))
        cores[1] = tn.einsum('rjs,j,mj,nj->rmns',cores[1],tn.tensor(w2),tn.tensor(self.basis[1](p2)),tn.tensor(self.basis[1](p2)))
        cores[2] = tn.einsum('rjs,j,mj,nj->rmns',cores[2],tn.tensor(w3),tn.tensor(self.basis[2](p3)),tn.tensor(self.basis[2](p3)))

        for i in range(3,len(cores)):
            cores[i] = tn.einsum('rjs,mj,nj->rmns',cores[i],tn.eye(cores[i].shape[1],dtype = tn.float64),tn.eye(cores[i].shape[1], dtype = tn.float64))
            
        return tntt.TT(cores)

    def stiffness_interp(self, eps = 1e-10, func = None, func_reference = None, rankinv = 1024, device = None, verb = False, qtt = False):
        
        
        p1, w1 = points_basis(self.basis[0],mult=2)
        p2, w2 = points_basis(self.basis[1],mult=2)
        p3, w3 = points_basis(self.basis[2],mult=2)
        ps = [tn.tensor(p1),tn.tensor(p2),tn.tensor(p3)]
        ws = [tn.tensor(w1), tn.tensor(w2), tn.tensor(w3)]
        
        params = [tn.tensor(b.interpolating_points()[0]) for b in self.basis[3:]]
        
        tme = datetime.datetime.now()
        Og_tt = self.eval_omega([p1,p2,p3], eps)
        tme = datetime.datetime.now() - tme
        if verb: print('time omega' , tme,flush=True)

        if qtt: 
            Nqtt = qtt_shape(Og_tt, list(range(len(Og_tt.N))))
            if verb:
                print('QTT enabled:')
                print(list(Og_tt.N))
                print('  || ')
                print('  \/  ')
                print(Nqtt)
            No =list(Og_tt.N)
            Og_tt = tntt.reshape(Og_tt,Nqtt)     


        if not qtt:
            tme = datetime.datetime.now()
            # Ogi_tt = 1/Og_tt
            Ogi_tt = tntt.elementwise_divide(tntt.ones(Og_tt.N, dtype = tn.float64, device = device), Og_tt, eps = eps, starting_tensor = None, nswp = 50, kick = 8)
            tme = datetime.datetime.now() -tme
            if verb: print('time omega inv' , tme,' rank ',Ogi_tt.R,flush=True)
            #if verb: print('invert error ',(Ogi_tt*Og_tt-tt.ones(Og_tt.n)).norm()/tt.ones(Og_tt.n).norm())
            Ogi_tt = Ogi_tt.round(eps)
        else:
            pass
            # Ogi_tt = tntt.elementwise_divide(tntt.ones(Og_tt.N, dtype = tn.float64, device = device), Og_tt, eps = eps, starting_tensor = None, nswp = 50, kick = 8)
        

        if func != None or func_reference != None:
            tmp = tntt.meshgrid(params)
            if func_reference == None:
                X, Y, Z = self(ps, (False,False,False))
                F_tt = tntt.interpolate.function_interpolate(func, [X, Y, Z]+[tntt.ones(X.N[:3]) ** t for t in tmp], eps = eps , verbose=False).round(eps)
                
                if verb: print('rank of Frtt is ',F_tt.r)
            else:
                F_tt = tntt.interpolate.function_interpolate(func_reference,tntt.meshgrid(ps+params),eps = eps,verbose = False).round(eps)
                if verb: print('rank of Ftt is ',F_tt.R)
        else:
            F_tt = tntt.ones(Og_tt.N)

        if qtt:
            F_tt = tntt.reshape(F_tt,Nqtt) 
        else: 
            Ogi_tt = (Ogi_tt * F_tt).round(eps)
        
        tme = datetime.datetime.now()
        g11, g21, g31 = GTT(ps, self.basis, self.Xs, (True,False,False))
        g12, g22, g32 = GTT(ps, self.basis, self.Xs, (False,True,False))
        g13, g23, g33 = GTT(ps, self.basis, self.Xs, (False,False,True))
        if verb:
            print(g11.R) 
            print(g12.R) 
            print(g13.R) 
            print(g21.R) 
            print(g22.R) 
            print(g23.R) 
            print(g31.R) 
            print(g32.R) 
            print(g33.R) 
        tme = datetime.datetime.now() -tme
        if verb:  print('G computed in ' , tme)
        
        # adjugate
        tme = datetime.datetime.now()
        h11,h12,h13 = (g22*g33-g23*g32, g13*g32-g12*g33, g12*g23-g13*g22)
        h21,h22,h23 = (g23*g31-g21*g33, g11*g33-g13*g31, g13*g21-g11*g23)
        h31,h32,h33 = (g21*g32-g22*g31, g12*g31-g11*g32, g11*g22-g12*g21)
        
        # tme = datetime.datetime.now()
        H = [[h11.round(eps),h12.round(eps),h13.round(eps)],[h21.round(eps),h22.round(eps),h23.round(eps)],[h31.round(eps),h32.round(eps),h33.round(eps)]]

        tme = datetime.datetime.now() -tme
        if verb: print('H computed in' , tme)
        
        Bs = [tn.tensor(self.basis[i](ps[i]).transpose()) for i in range(3)]
        dBs = [tn.tensor(self.basis[i](ps[i],derivative = True).transpose()) for i in range(3)]
                
        N = self.Xs[0].N
        S = None
        SS = None
        Hs = dict()
        
        # the size of the bands
        band_size = [b.deg for b in self.basis[:3]]+[1]*len(N[3:])

        if qtt: 
            for i in range(3):
                for j in range(3):
                    H[i][j] = tntt.reshape(H[i][j], Nqtt, eps)
                    # print(H[i][j].r)

        for alpha in range(3):
            for beta in range(3):
                if verb: print('alpha, beta = ',alpha,beta)
                tme = datetime.datetime.now()

                tmp = H[alpha][0]*H[beta][0]+H[alpha][1]*H[beta][1]+H[alpha][2]*H[beta][2]
                tmp = tmp.round(eps,rankinv)
                tme = datetime.datetime.now() -tme
                if verb: print('\ttime 1 ' , tme)


                tme = datetime.datetime.now()
                if not qtt:
                    tmp = tmp*Ogi_tt
                    tmp = tmp.round(eps,rankinv)
                else:
                    tmp = tntt.elementwise_divide(tmp,Og_tt, starting_tensor = tmp, eps=eps,kick=8, nswp = 50)*F_tt 
                    # tmp = tmp*Ogi_tt*F_tt

            #  print('Rank of product',tmp.r)
                
                tme = datetime.datetime.now() -tme
                if verb: print('\ttime 2 ' , tme,' rank ',tmp.R)
                
                # print('ERR ',(tmp-tmp2).norm()/tmp.norm())

                if qtt: tmp = tntt.reshape(tmp,No)
                
                # tmp = H[alpha][0]*Hi[beta][0]+H[alpha][1]*Hi[beta][1]+H[alpha][2]*Hi[beta][2]
                # tmp = tmp.round(eps,rankinv)
                
                tme = datetime.datetime.now()
                cores = tmp.cores
                
                tme = datetime.datetime.now()
                # cores[0] = np.einsum('rjs,j,jm,jn->rmns',cores[0],w1,dB1 if alpha==0 else B1,dB1 if beta==0 else B1)
                # print(cores[0].shape,w1.shape)
                for i in range(3):
                    cores[i] = tn.einsum('rjs,j->rjs',cores[i],ws[i])
                    tmp = tn.einsum('jm,jn->jmn',dBs[i] if alpha==i else Bs[i], dBs[i] if beta==i else Bs[i])
                    cores[i] = tn.einsum('rjs,jmn->rmns',cores[i],tmp)
                    
                for i in range(3,len(cores)):
                    cores[i] = tn.einsum('rjs,mj,nj->rmns',cores[i],tn.eye(cores[i].shape[1],dtype=tn.float64),tn.eye(cores[i].shape[1],dtype=tn.float64))
                tme = datetime.datetime.now() -tme
                if verb: print('\t\ttime ' , tme)
                
                
                tme = datetime.datetime.now()
                ss = tntt.TT([tn.tensor(bandcore2ttcore(cores[i].numpy(),band_size[i])) for i in range(len(cores))])

                
                SS = ss if SS==None else SS+ss 
                
                
                tme = datetime.datetime.now() -tme
                if verb: print('\ttime 4 ' , tme)
            tme = datetime.datetime.now()

            SS = SS.round(eps)
            
            tme = datetime.datetime.now() -tme
            if verb: print('\ttime ROUND ' , tme)
        
        cores = SS.cores
        SS = tntt.TT([tn.tensor(ttcore2bandcore(cores[i].numpy(),N[i],band_size[i])) for i in range(len(N))])

        return SS

    
    def plot_domain(self, params = None, bounds = None, fig = None, wireframe = True, frame_color = 'r', n = 12, surface_color = 'blue',alpha = 0.4):
        
        if fig == None:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
        else:
            ax = fig.gca()
            
        if wireframe:
            plot_func = ax.plot_wireframe
        else:
            plot_func = ax.plot_surface
        
        if bounds == None:
            bounds = [b.interval for b in self.basis[:3]]
            
        if surface_color != None:
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][0],1, dtype = tn.float64)]+([] if params ==None else params))
            plot_func(x.full()[:,:,0].numpy().squeeze(), y.full()[:,:,0].numpy().squeeze(), z.full()[:,:,0].numpy().squeeze(), color = surface_color,alpha = alpha)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][1],bounds[2][1],1, dtype = tn.float64)]+([] if params ==None else params))
            plot_func(x.full()[:,:,0].numpy().squeeze(), y.full()[:,:,0].numpy().squeeze(), z.full()[:,:,0].numpy().squeeze(), color = surface_color,alpha = alpha)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][0],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            plot_func(x.full()[:,0,:].numpy().squeeze(), y.full()[:,0,:].numpy().squeeze(), z.full()[:,0,:].numpy().squeeze(), color = surface_color,alpha = alpha)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][1],bounds[1][1],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            plot_func(x.full()[:,0,:].numpy().squeeze(), y.full()[:,0,:].numpy().squeeze(), z.full()[:,0,:].numpy().squeeze(), color = surface_color,alpha = alpha)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][0],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            plot_func(x.full()[0,:,:].numpy().squeeze(), y.full()[0,:,:].numpy().squeeze(), z.full()[0,:,:].numpy().squeeze(), color = surface_color,alpha = alpha)
            
            x,y,z = self([tn.linspace(bounds[0][1],bounds[0][1],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            plot_func(x.full()[0,:,:].numpy().squeeze(), y.full()[0,:,:].numpy().squeeze(), z.full()[0,:,:].numpy().squeeze(), color = surface_color,alpha = alpha)
        
        if frame_color != None:
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][0],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][0],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][0],1, dtype = tn.float64),tn.linspace(bounds[2][1],bounds[2][1],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][1],bounds[1][1],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][0],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][1],n, dtype = tn.float64),tn.linspace(bounds[1][1],bounds[1][1],1, dtype = tn.float64),tn.linspace(bounds[2][1],bounds[2][1],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][0],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][0],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][0],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][1],bounds[2][1],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color)
            
            x,y,z = self([tn.linspace(bounds[0][1],bounds[0][1],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][0],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color)
            
            x,y,z = self([tn.linspace(bounds[0][1],bounds[0][1],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][1],n, dtype = tn.float64),tn.linspace(bounds[2][1],bounds[2][1],1, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][0],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][0],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color)
            
            x,y,z = self([tn.linspace(bounds[0][0],bounds[0][0],1, dtype = tn.float64),tn.linspace(bounds[1][1],bounds[1][1],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color)
            
            x,y,z = self([tn.linspace(bounds[0][1],bounds[0][1],1, dtype = tn.float64),tn.linspace(bounds[1][0],bounds[1][0],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color)
            
            x,y,z = self([tn.linspace(bounds[0][1],bounds[0][1],1, dtype = tn.float64),tn.linspace(bounds[1][1],bounds[1][1],1, dtype = tn.float64),tn.linspace(bounds[2][0],bounds[2][1],n, dtype = tn.float64)]+([] if params ==None else params))
            ax.plot(x.full().numpy().flatten(), y.full().numpy().flatten(), z.full().numpy().flatten(), frame_color)
        
        return fig

    
    
