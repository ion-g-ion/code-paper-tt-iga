import torch as tn
import numpy as np
import torchtt as tntt



def points_basis(basis,mult = 2):
    
    Pts, Ws =np.polynomial.legendre.leggauss(basis.deg*mult)
    p1 = []
    w1 = []
    for i in range((basis.knots).size-1):
        a = basis.knots[i]
        b = basis.knots[i+1]
        if b>a :
            pts = a+(Pts+1)*0.5*(b-a)
            ws = Ws*(b-a)/2
            p1 += list(pts)
            w1 += list(ws)
    p1 = np.array(p1)
    w1 = np.array(w1)
    
    return p1, w1

def qtt_shape(tens,dims = [0,1,2]):
    N = []
    for i in range(len(tens.N)):
        N+=prime_decomposition(tens.N[i]) if i in dims else [tens.N[i]]
    return N

def prime_decomposition(n):

    i = 2
    factors = []
    no = n
    while i <=no and n>1:
        if n%i==0:
            n = n//i
            factors.append(i)
        else:
            i+=1
    return factors

def bandcore2ttcore(core,nb):
    
    lst = [np.diagonal(core,k,1,2).transpose([0,2,1]) for k in range(-nb,nb+1)]
    
    return np.concatenate(tuple(lst),1)


def ttcore2bandcore(core,n,nb):
    core_new = np.zeros((core.shape[0],n,n,core.shape[-1]))
    idx = 0
    for k in range(-nb,nb+1):
        l = n - np.abs(k)
        s1 = list(range(max(0,-k),min(n,n-k)))
        s2 = list(range(max(0,k),min(n,n+k)))
       
        core_new[:,s1,s2,:] = core[:,(idx):(idx+l),:]
        idx += l
    return core_new


def GTT(x,Basis,Xs,deriv=(False,False,False)):
    
    
    B1 = tn.tensor(Basis[0](x[0],derivative=deriv[0]).transpose())
    B2 = tn.tensor(Basis[1](x[1],derivative=deriv[1]).transpose())
    B3 = tn.tensor(Basis[2](x[2],derivative=deriv[2]).transpose())
    
    res1 = Xs[0].mprod([B1,B2,B3],[0,1,2])
    res2 = Xs[1].mprod([B1,B2,B3],[0,1,2])
    res3 = Xs[2].mprod([B1,B2,B3],[0,1,2])
    
    return res1,res2,res3

