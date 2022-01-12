
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import scipy.interpolate
import numpy as np

class BSplineBasis:
    
    def __init__(self,knots,deg,ends = (True,True)):
        
        self.N=knots.size+deg-1
        self.deg=deg
        self.knots=np.hstack( ( np.ones(deg)*knots[0] , knots , np.ones(deg)*knots[-1] ) )
        self.spl = []
        self.dspl = []
        self.interval = (np.min(knots),np.max(knots))
        for i in range(self.N):
            c=np.zeros(self.N)
            c[i]=1
            self.spl.append(BSpline(self.knots,c,self.deg))
            self.dspl.append(scipy.interpolate.splder( BSpline(self.knots,c,self.deg) ))
        
        self.compact_support_bsp = np.zeros((self.N,2))
        for i in range(self.N):
            self.compact_support_bsp[i,0] = self.knots[i]
            self.compact_support_bsp[i,1] = self.knots[i+self.deg+1]
            
        int_bsp_bsp = np.zeros((self.N,self.N))
        int_bsp = np.zeros((self.N,1))
        # int_bsp_v = np.zeros((self.Nz,1))
        
        Pts, Ws =np.polynomial.legendre.leggauss(20)
        for i in range(self.N):
            a=self.compact_support_bsp[i,0]
            b=self.compact_support_bsp[i,1]

            for k in range(self.knots.size-1):
                if self.knots[k]>=a and self.knots[k+1]<=b:
                    pts = self.knots[k]+(Pts+1)*0.5*(self.knots[k+1]-self.knots[k])
                    ws = Ws*(self.knots[k+1]-self.knots[k])/2
                    int_bsp[i,0] += np.sum( self.__call__(pts,i) * ws )
                    
            for j in range(i,self.N):
                a=self.compact_support_bsp[j,0]
                b=self.compact_support_bsp[i,1]
                if b>a:
                    for k in range(self.knots.size-1):
                        if self.knots[k]>=a and self.knots[k+1]<=b:
                            pts = self.knots[k]+(Pts+1)*0.5*(self.knots[k+1]-self.knots[k])
                            ws = Ws*(self.knots[k+1]-self.knots[k])/2
                            int_bsp_bsp[i,j] += np.sum(  self.__call__(pts,i) *self.__call__(pts,j) * ws )
                            # int_bspp[i,j] += np.sum( self.bspp(pts)[i,:]* self.bspp(pts)[j,:]*ws )
                    if i!=j:
                        int_bsp_bsp[j,i] = int_bsp_bsp[i,j]
                        # int_bspp[j,i] = int_bspp[i,j]
                    
        
        self.int_bsp_bsp = int_bsp_bsp
        # self.int_bspp_bspp = int_bspp
        self.int_bsp = int_bsp
        
    
    def __call__(self,x,i=None,derivative=False):
        if i==None:
            if derivative:
                ret = np.array([self.dspl[i](x) for i in range(self.N)])
                return ret
            else:
                ret = np.array([self.spl[i](x) for i in range(self.N)])
                return ret
        else:
            if derivative:
                 return self.dspl[i](x)
            else:
                return self.spl[i](x)
            
    def greville(self):
        return np.array([np.sum(self.knots[i+1:i+self.deg+1]) for i in range(self.N)])/(self.deg)
        # return np.array([np.sum(self.knots[i+2:i+self.deg+2]) for i in range(self.N)])/(self.deg-1)
        
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
            
    def eval_all(self,c,x):
        c=np.hstack((c,np.zeros(self.deg-2)))
        return BSpline(self.knots,c,self.deg)(x)
    
    def plot(self,derivative = False):
        
        for i in range(self.N):
            x=np.linspace(self.compact_support_bsp[i,0],self.compact_support_bsp[i,1],500)
            plt.plot(x,self.__call__(x,i,derivative))
    def derivative(self):
        bd = scipy.interpolate.splder(BSpline(self.knots,np.zeros(self.N+self.deg-1)+1,self.deg))
        return BSplineBasis(np.unique(bd.t), bd.k)
    def integrate(self):
        BI = np.zeros((self.N+1,self.N+1))
        BII = np.zeros((self.N+1,self.N+1))
        a = self.knots[0]
        b = self.knots[-1]
        pts, ws =np.polynomial.legendre.leggauss(128)
        pts = a+(pts+1)*0.5*(b-a)
        for i in range(self.N+1):
            for j in range(i,self.N+1):
                BI[i,j] = np.sum( self.eval_single(i,pts)*self.eval_single(j,pts)*ws*(b-a)/2 )
                BI[j,i] = BI[i,j]
        return BII,BI

