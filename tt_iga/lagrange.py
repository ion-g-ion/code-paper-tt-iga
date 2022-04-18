import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import scipy.interpolate
import numpy as np

class Lagrange:
    
    def __init__(self,deg,interval = [-1,1],points=None):
        
        self.N=deg

        self.interval = interval
        self.basis = []
        self.dbasis = []
        
        if points is None:
            self.points,_ = np.polynomial.legendre.leggauss(self.N)
            self.points = np.sort(0.5*(self.points+1)*(interval[1]-interval[0])+interval[0])
        else:
            self.points = points 
            
        for i in range(self.N):
            c=np.zeros(self.N)
            c[i]=1
            self.basis.append(scipy.interpolate.lagrange(self.points, c))
            self.dbasis.append(scipy.interpolate.lagrange(self.points, c).deriv())
        
        
            
        int_bsp_bsp = np.zeros((self.N,self.N))
        int_bsp = np.zeros((self.N,1))
        # int_bsp_v = np.zeros((self.Nz,1))
        
        
        for i in range(self.N):
            tmp = self.basis[i].integ()
            int_bsp[i] = tmp(self.interval[1]) - tmp(self.interval[0])
                    
            for j in range(i,self.N):
               tmp = (self.basis[i]*self.basis[j]).integ()
               int_bsp_bsp[i,j] = tmp(self.interval[1]) - tmp(self.interval[0])
               int_bsp_bsp[i,j] = int_bsp_bsp[j,i]
        
        self.int_bsp_bsp = int_bsp_bsp
        # self.int_bspp_bspp = int_bspp
        self.int_bsp = int_bsp
        
    
    def __call__(self,x,i=None,derivative=False):
        if i==None:
            if derivative:
                ret = np.array([self.dbasis[i](x) for i in range(self.N)])
                return ret
            else:
                ret = np.array([self.basis[i](x) for i in range(self.N)])
                return ret
        else:
            if derivative:
                 return self.dbasis[i](x)
            else:
                return self.basis[i](x)
            
    def points_weights_matrix(self):
        pts = self.points
        ws = []
        for p in self.basis:
            i = p.integ()
            ws.append(i(self.interval[1])-i(self.interval[0]))
        np.array(ws)
        mat = np.eye(self.N)
        return pts, ws, mat
    
    def plot(self,derivative = False):
        x=np.linspace(self.interval[0],self.interval[1],1000)
        for i in range(self.N):
            plt.plot(x,self.__call__(x,i,derivative))
            
    def interpolating_points(self):
        pts = np.array([np.sum(self.knots[i+1:i+self.deg+1]) for i in range(self.N)])/(self.deg)
        Mat = self.__call__(pts)
        return pts, Mat
        
    def abscissae(self):
        return self.greville()
    def collocation_points(self,mult = 1):
        pts = []
        ws = []
        Pts, Ws = np.polynomial.legendre.leggauss(mult*(self.deg+1))
        for k in range(self.knots.size-1):
            if self.knots[k+1]>self.knots[k]:
                pts += list(self.knots[k]+(Pts+1)*0.5*(self.knots[k+1]-self.knots[k]))
                ws += list(Ws*(self.knots[k+1]-self.knots[k])/2)
        pts = np.array(pts)
        ws = np.array(ws)
        
        return pts, ws
    
class LagrangeLeg:
    
    def __init__(self,deg,interval = [-1,1]):
        """
        Creates a Lagrange polynomials basis of degree `deg` corresponding to the Gauss-Legendre nodes (scaled to a given interval).

        Args:
            deg (int): the dimension of the basis.
            interval (list, optional): the interval $I$ where the polynomials are defined. Defaults to [-1,1].
        """
        self.N=deg

        self.interval = interval
        self.basis = []
        self.dbasis = []
        
        
        self.points,self.ws = np.polynomial.legendre.leggauss(self.N)
        self.points = (0.5*(self.points+1)*(interval[1]-interval[0])+interval[0])
        self.ws = 0.5*(interval[1]-interval[0])*self.ws
            
        for i in range(self.N):
            c=np.zeros(self.N)
            c[i]=1
            self.basis.append(scipy.interpolate.lagrange(self.points, c))
            self.dbasis.append(scipy.interpolate.lagrange(self.points, c).deriv())
        
        
            
        int_bsp_bsp = np.zeros((self.N,self.N))
        int_bsp = np.zeros((self.N,1))
        # int_bsp_v = np.zeros((self.Nz,1))
        
        
        for i in range(self.N):
            tmp = self.basis[i].integ()
            int_bsp[i] = tmp(self.interval[1]) - tmp(self.interval[0])
                    
            for j in range(i,self.N):
               tmp = (self.basis[i]*self.basis[j]).integ()
               int_bsp_bsp[i,j] = tmp(self.interval[1]) - tmp(self.interval[0])
               int_bsp_bsp[i,j] = int_bsp_bsp[j,i]
        
        self.int_bsp_bsp = int_bsp_bsp
        # self.int_bspp_bspp = int_bspp
        self.int_bsp = int_bsp
        
    
    def __call__(self,x,i=None,derivative=False):
        """
        

        Args:
            x (_type_): _description_
            i (_type_, optional): _description_. Defaults to None.
            derivative (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if i==None:
            if derivative:
                ret = np.array([self.dbasis[i](x) for i in range(self.N)])
                return ret
            else:
                ret = np.array([self.basis[i](x) for i in range(self.N)])
                return ret
        else:
            if derivative:
                 return self.dbasis[i](x)
            else:
                return self.basis[i](x)
            
    def points_weights_matrix(self):
        pts = self.points
        ws = []
        for p in self.basis:
            i = p.integ()
            ws.append(i(self.interval[1])-i(self.interval[0]))
        np.array(ws)
        mat = np.eye(self.N)
        return pts, ws, mat
    
    def plot(self,derivative = False):
        x=np.linspace(self.interval[0],self.interval[1],1000)
        for i in range(self.N):
            plt.plot(x,self.__call__(x,i,derivative))
    
    def interpolating_points(self):
        pts = self.points.copy()
        Mat = np.eye(len(pts))
        return pts, Mat
        
    

    def collocation_points(self,mult = 1):
        pts = []
        ws = []
        Pts, Ws = np.polynomial.legendre.leggauss(mult*(self.N+1))
        
        pts = self.interval[0]+(Pts+1)*0.5*(self.interval[1]-self.interval[0])
        ws = Ws*0.5*(self.interval[1]-self.interval[0])
 
        return pts, ws