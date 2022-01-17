"""
Created on Fri Jun 25 14:37:37 2021

@author: yonnss
"""
import numpy as np
import scipy.sparse
import datetime
import torch as tn
import numba
import opt_einsum as oe

class timer():
    
    def __init__(self):
        self.time = datetime.datetime.now()
    
    def tic(self):
        self.time = datetime.datetime.now()
        
    
    def toc(self,msg = 'Elapsed time: '):
        print(msg,  datetime.datetime.now()-self.time)
        return (datetime.datetime.now()-self.time).total_seconds()
    
timer = timer()  
 
def eval_omega_3d(Ps,Basis,xs,to_tf=False):
    
    B1 = Basis[0](xs[0]).transpose()
    dB1 = Basis[0](xs[0],derivative=True).transpose()
    B2 = Basis[1](xs[1]).transpose()
    dB2 = Basis[1](xs[1],derivative=True).transpose()
    B3 = Basis[2](xs[2]).transpose()
    dB3 = Basis[2](xs[2],derivative=True).transpose()
     
    B1 =  tn.tensor(B1)
    B2 =  tn.tensor(B2)
    B3 =  tn.tensor(B3)
    dB1 = tn.tensor(dB1)
    dB2 = tn.tensor(dB2)
    dB3 = tn.tensor(dB3)
    Ps0 = tn.tensor(Ps[0])
    Ps1 = tn.tensor(Ps[1])
    Ps2 = tn.tensor(Ps[2])
  
    det1 =  oe.contract('ij,kl,mn,jln->ikm',dB1,B2,B3,Ps0)
    det1 *= oe.contract('ij,kl,mn,jln->ikm',B1,dB2,B3,Ps1)
    det1 *= oe.contract('ij,kl,mn,jln->ikm',B1,B2,dB3,Ps2)
    det2 =  oe.contract('ij,kl,mn,jln->ikm',B1,dB2,B3,Ps0)
    det2 *= oe.contract('ij,kl,mn,jln->ikm',B1,B2,dB3,Ps1)
    det2 *= oe.contract('ij,kl,mn,jln->ikm',dB1,B2,B3,Ps2)
    det3 =  oe.contract('ij,kl,mn,jln->ikm',B1,B2,dB3,Ps0)
    det3 *= oe.contract('ij,kl,mn,jln->ikm',dB1,B2,B3,Ps1)
    det3 *= oe.contract('ij,kl,mn,jln->ikm',B1,dB2,B3,Ps2)
    det4 =  oe.contract('ij,kl,mn,jln->ikm',B1,B2,dB3,Ps0)
    det4 *= oe.contract('ij,kl,mn,jln->ikm',B1,dB2,B3,Ps1)
    det4 *= oe.contract('ij,kl,mn,jln->ikm',dB1,B2,B3,Ps2)
    det5 =  oe.contract('ij,kl,mn,jln->ikm',dB1,B2,B3,Ps0)
    det5 *= oe.contract('ij,kl,mn,jln->ikm',B1,B2,dB3,Ps1)
    det5 *= oe.contract('ij,kl,mn,jln->ikm',B1,dB2,B3,Ps2)
    det6 =  oe.contract('ij,kl,mn,jln->ikm',B1,dB2,B3,Ps0)
    det6 *= oe.contract('ij,kl,mn,jln->ikm',dB1,B2,B3,Ps1)
    det6 *= oe.contract('ij,kl,mn,jln->ikm',B1,B2,dB3,Ps2)
   
    res = (det1 + det2 + det3 - det4 - det5 - det6)
    if not to_tf:
        res =res.numpy()
    return res 

def get_inegration_points(nl,a,b):
     
    Pts, Ws =np.polynomial.legendre.leggauss(nl)
        
    pts = a+(Pts+1)*0.5*(b-a)
    ws = Ws*(b-a)/2
      
    return pts, ws

@numba.jit(nopython=True)
def mass_aux(i1,i2,j1,j2,k1,k2,B1,B2,B3,Bm,Bn,Bo,omega_eval,ws1,ws2,ws3):
    n = (i2-i1)*(j2-j1)*(k2-k1)
    vals = np.zeros((n))
    idxs = np.zeros((n,3))
    idx = 0
    for i in range(i1,i2):
        for j in range(j1,j2):
            for k in range(k1,k2):
                Bi = B1[i,:]
                Bj = B2[j,:]
                Bk = B3[k,:]
                            
                integral = np.dot(omega_eval,Bo*Bk*ws3)
                integral = np.dot(integral,Bn*Bj*ws2)
                integral = np.dot(integral,Bm*Bi*ws1)
                            
                vals[idx] = integral
                idxs[idx,0] = i
                idxs[idx,1] = j
                idxs[idx,2] = k
                idx += 1
    
    return idxs, vals

                           
def construct_mass_sparse2(basis,Ps,normal_ordering=True, kappa = None):
    
    N = [basis[0].N, basis[1].N, basis[2].N]
    
    rows = np.zeros((np.prod(N)*(2*basis[0].deg+1)*(2*basis[1].deg+1)*(2*basis[2].deg+1),3),dtype=int)
    cols = np.zeros((np.prod(N)*(2*basis[0].deg+1)*(2*basis[1].deg+1)*(2*basis[2].deg+1),3),dtype=int)
    vals = np.zeros((np.prod(N)*(2*basis[0].deg+1)*(2*basis[1].deg+1)*(2*basis[2].deg+1)))
    idx = 0
    
    nw = basis[0].deg*3
    
    Pts, Ws =np.polynomial.legendre.leggauss(nw)
    
    knots = np.unique(basis[0].knots)
    pts1 = []
    ws1 = []
    for i in range(knots.size-1):
        # pts, ws = get_inegration_points(basis[0].deg*3,knots[i],knots[i+1])
        pts = knots[i]+(Pts+1)*0.5*(knots[i+1]-knots[i])
        ws = Ws*(knots[i+1]-knots[i])/2
        
        pts1 += list(pts)
        ws1 += list(ws)
        
    pts1 = np.array(pts1)
    ws1 = np.array(ws1)
    
    knots = np.unique(basis[1].knots)
    pts2 = []
    ws2 = []
    for i in range(knots.size-1):
        # pts, ws = get_inegration_points(basis[1].deg*3,knots[i],knots[i+1])
        pts = knots[i]+(Pts+1)*0.5*(knots[i+1]-knots[i])
        ws = Ws*(knots[i+1]-knots[i])/2
        
        pts2 += list(pts)
        ws2 += list(ws)
    pts2 = np.array(pts2)
    ws2 = np.array(ws2)
    
    knots = np.unique(basis[2].knots)
    pts3 = []
    ws3 = []
    for i in range(knots.size-1):
        # pts, ws = get_inegration_points(basis[2].deg*3,knots[i],knots[i+1])
        pts = knots[i]+(Pts+1)*0.5*(knots[i+1]-knots[i])
        ws = Ws*(knots[i+1]-knots[i])/2
        
        pts3 += list(pts)
        ws3 += list(ws)
    pts3 = np.array(pts3)
    ws3 = np.array(ws3)
       

    B1 = basis[0](pts1)
    B2 = basis[1](pts2)
    B3 = basis[2](pts3)      

    # omega_all = eval_omega_3d(Ps, basis, [pts1, pts2, pts3])

    for m in range(N[0]):
        i1 = max([m-basis[0].deg,0])
        i2 = min([m+basis[0].deg+1,N[0]])
        p1 = pts1[i1*nw:i2*nw]
        w1 = ws1[i1*nw:i2*nw]
        omega_all = eval_omega_3d(Ps, basis, [p1, pts2, pts3])
        for n in range(N[1]):
            for o in range(N[2]):
                # print(m,n,o)
                # timer.tic()
                
                
                j1 = max([n-basis[1].deg,0])
                j2 = min([n+basis[1].deg+1,N[1]])
                k1 = max([o-basis[2].deg,0])
                k2 = min([o+basis[2].deg+1,N[2]])
                
                p2 = pts2[j1*nw:j2*nw]
                p3 = pts3[k1*nw:k2*nw]
                
                w2 = ws2[j1*nw:j2*nw]
                w3 = ws3[k1*nw:k2*nw]
                
                Bm = B1[m,i1*nw:i2*nw]
                Bn = B2[n,j1*nw:j2*nw]
                Bo = B3[o,k1*nw:k2*nw]
                # timer.toc()
                # timer.tic()
                # omega_eval = eval_omega_3d(Ps, basis, [p1, p2, p3],to_tf=True)
                omega_eval = omega_all[:,j1*nw:j2*nw,k1*nw:k2*nw]
                # timer.toc()
                # timer.tic()
                
                # timer.tic()
                Bi = B1[i1:i2,i1*nw:i2*nw]
                Bj = B2[j1:j2,j1*nw:j2*nw]
                Bk = B3[k1:k2,k1*nw:k2*nw]
                
                ints = oe.einsum('ijk,i,j,k,ai,bj,ck->abc',omega_eval,Bm*w1,Bn*w2,Bo*w3,Bi,Bj,Bk)
                toadd = (i2-i1)*(j2-j1)*(k2-k1)
  
                vals[idx:idx+toadd] = ints.numpy().flatten()            
                # timer.toc()
                
                # timer.tic()
                
                
                rows_tmp = [[m,n,o]]*(i2-i1)*(j2-j1)*(k2-k1)
                cols_tmp = [[i,j,k] for i in range(i1,i2) for j in range(j1,j2) for k in range(k1,k2)]
                     
                rows[idx:idx+toadd] = np.array(rows_tmp)
                cols[idx:idx+toadd] = np.array(cols_tmp)
                
                idx += toadd
                            
                # timer.toc('classic ')
                # timer.tic()
                # I,V = mass_aux(max([m-basis[0].deg,0]),min([m+basis[0].deg+1,N[0]]),max([n-basis[1].deg,0]),min([n+basis[1].deg+1,N[1]]),max([o-basis[2].deg,0]),min([o+basis[2].deg+1,N[2]]),B1,B2,B3,Bm,Bn,Bo,omega_eval,ws1,ws2,ws3)
                # timer.toc('compiled ')
    
    vals = vals[:idx]
    rows = rows[:idx,:]
    cols = cols[:idx,:]
    
    if normal_ordering:
        rows = [r[0]*N[1]*N[2]+r[1]*N[2]+r[2] for r in rows] 
        cols = [c[0]*N[1]*N[2]+c[1]*N[2]+c[2] for c in cols] 
    else:    
        rows = [r[2]*N[1]*N[0]+r[1]*N[0]+r[0] for r in rows] 
        cols = [c[2]*N[1]*N[0]+c[1]*N[0]+c[0] for c in cols] 
               
    # return rows, cols, vals      
    return scipy.sparse.coo_matrix((vals, (rows, cols)))  
    
def construct_mass_sparse(basis,Ps,normal_ordering=True):
    
    N = [basis[0].N, basis[1].N, basis[2].N]
    
    rows = []
    cols = []
    vals = []
    
    Pts, Ws =np.polynomial.legendre.leggauss(basis[0].deg*3)
    
    
    for m in range(N[0]):
        for n in range(N[1]):
            for o in range(N[2]):
                # print(m,n,o)
                # timer.tic()
                knots = np.unique(basis[0].knots)
                pts1 = []
                ws1 = []
                for i in range(knots.size-1):
                    if knots[i]>=basis[0].compact_support_bsp[m,0] and knots[i+1]<=basis[0].compact_support_bsp[m,1]:
                        
                        # pts, ws = get_inegration_points(basis[0].deg*3,knots[i],knots[i+1])
                        pts = knots[i]+(Pts+1)*0.5*(knots[i+1]-knots[i])
                        ws = Ws*(knots[i+1]-knots[i])/2
                        
                        pts1 += list(pts)
                        ws1 += list(ws)
                pts1 = np.array(pts1)
                ws1 = np.array(ws1)
                
                knots = np.unique(basis[1].knots)
                pts2 = []
                ws2 = []
                for i in range(knots.size-1):
                    if knots[i]>=basis[1].compact_support_bsp[n,0] and knots[i+1]<=basis[1].compact_support_bsp[n,1]:
                        
                        # pts, ws = get_inegration_points(basis[1].deg*3,knots[i],knots[i+1])
                        pts = knots[i]+(Pts+1)*0.5*(knots[i+1]-knots[i])
                        ws = Ws*(knots[i+1]-knots[i])/2
                        
                        pts2 += list(pts)
                        ws2 += list(ws)
                pts2 = np.array(pts2)
                ws2 = np.array(ws2)
                
                knots = np.unique(basis[2].knots)
                pts3 = []
                ws3 = []
                for i in range(knots.size-1):
                    if knots[i]>=basis[2].compact_support_bsp[o,0] and knots[i+1]<=basis[2].compact_support_bsp[o,1]:
                        
                        # pts, ws = get_inegration_points(basis[2].deg*3,knots[i],knots[i+1])
                        pts = knots[i]+(Pts+1)*0.5*(knots[i+1]-knots[i])
                        ws = Ws*(knots[i+1]-knots[i])/2
                        
                        pts3 += list(pts)
                        ws3 += list(ws)
                pts3 = np.array(pts3)
                ws3 = np.array(ws3)
                # timer.toc()
                
                # timer.tic()
                omega_eval = (eval_omega_3d(Ps, basis, [pts1, pts2, pts3]))
                # timer.toc()
                
                B1 = basis[0](pts1)
                B2 = basis[1](pts2)
                B3 = basis[2](pts3)
                
    
                Bm = B1[m,:]
                Bn = B2[n,:]
                Bo = B3[o,:]
                
                # timer.tic()
                # Bi = B1[max([m-basis[0].deg,0]):min([m+basis[0].deg+1,N[0]]),:]
                # Bj = B2[max([n-basis[1].deg,0]):min([n+basis[1].deg+1,N[1]]),:]
                # Bk = B3[max([o-basis[2].deg,0]):min([o+basis[2].deg+1,N[2]]),:]
                
                # ints = tf.einsum('ijk,i,j,k,ai,bj,ck->abc',omega_eval,Bm*ws1,Bn*ws2,Bo*ws3,Bi,Bj,Bk)
                # vals+= list(ints.numpy().flatten())
                
                # timer.toc()
                
                # timer.tic()
                for i in range(max([m-basis[0].deg,0]),min([m+basis[0].deg+1,N[0]])):
                    for j in range(max([n-basis[1].deg,0]),min([n+basis[1].deg+1,N[1]])):
                        for k in range(max([o-basis[2].deg,0]),min([o+basis[2].deg+1,N[2]])):
                            Bi = B1[i,:]
                            Bj = B2[j,:]
                            Bk = B3[k,:]
                            
                            # # Bmno = np.einsum('m,n,o->mno',Bm,Bn,Bo)
                            # # Bijk = np.einsum('m,n,o->mno',Bi,Bj,Bk)

                            # # integral = np.sum(Bmno*Bijk*omega_eval*np.einsum('a,b,c->abc',ws1,ws2,ws3))
                            
                            # # integral = np.einsum('a,b,c,abc->',Bm*Bi*ws1,Bn*Bj*ws2,Bo*Bk*ws3,omega_eval)
                            # # integral = np.sum(np.outer(np.outer(Bm*Bi*ws1,Bn*Bj*ws2),Bo*Bk*ws3)*omega_eval)
                            # # integral = tf.einsum('abc,a,b,c->',omega_eval,Bm*Bi*ws1,Bn*Bj*ws2,Bo*Bk*ws3).numpy()
                            integral = np.dot(omega_eval,Bo*Bk*ws3)
                            integral = np.dot(integral,Bn*Bj*ws2)
                            integral = np.dot(integral,Bm*Bi*ws1)
                            
                            rows.append((m,n,o))
                            cols.append((i,j,k))
                            vals.append(integral)
                            
                # timer.toc('classic ')
                # timer.tic()
                # I,V = mass_aux(max([m-basis[0].deg,0]),min([m+basis[0].deg+1,N[0]]),max([n-basis[1].deg,0]),min([n+basis[1].deg+1,N[1]]),max([o-basis[2].deg,0]),min([o+basis[2].deg+1,N[2]]),B1,B2,B3,Bm,Bn,Bo,omega_eval,ws1,ws2,ws3)
                # timer.toc('compiled ')
    if normal_ordering:
        rows = [r[0]*N[1]*N[2]+r[1]*N[2]+r[2] for r in rows] 
        cols = [c[0]*N[1]*N[2]+c[1]*N[2]+c[2] for c in cols] 
    else:    
        rows = [r[2]*N[1]*N[0]+r[1]*N[0]+r[0] for r in rows] 
        cols = [c[2]*N[1]*N[0]+c[1]*N[0]+c[0] for c in cols] 
               
    # return rows, cols, vals      
    return scipy.sparse.coo_matrix((vals, (rows, cols)))                       
                    
        
def eval_G(x,Basis,Xs,deriv=(False,False,False)):
    
    
    B1 = tn.tensor(Basis[0](x[0],derivative=deriv[0]).transpose())
    B2 = tn.tensor(Basis[1](x[1],derivative=deriv[1]).transpose())
    B3 = tn.tensor(Basis[2](x[2],derivative=deriv[2]).transpose())
    
    res1 = oe.contract('ij,kl,mn,jln->ikm',B1,B2,B3,tn.tensor(Xs[0]))
    res2 = oe.contract('ij,kl,mn,jln->ikm',B1,B2,B3,tn.tensor(Xs[1]))
    res3 = oe.contract('ij,kl,mn,jln->ikm',B1,B2,B3,tn.tensor(Xs[2]))
    
    
    return res1,res2,res3


def construct_stiff_sparse(basis,Ps,normal_ordering=True, kappa_ref = None):
    
    N = [basis[0].N, basis[1].N, basis[2].N]
    
    rows = np.zeros((np.prod(N)*(2*basis[0].deg+1)*(2*basis[1].deg+1)*(2*basis[2].deg+1),3),dtype=int)
    cols = np.zeros((np.prod(N)*(2*basis[0].deg+1)*(2*basis[1].deg+1)*(2*basis[2].deg+1),3),dtype=int)
    vals = np.zeros((np.prod(N)*(2*basis[0].deg+1)*(2*basis[1].deg+1)*(2*basis[2].deg+1)))
    idx = 0
    
    
    nw = basis[0].deg*2
    
    Pts, Ws =np.polynomial.legendre.leggauss(nw)
    
    knots =  basis[0].knots[basis[0].deg:-basis[0].deg] # np.unique(basis[0].knots)

    pts1 = []
    ws1 = []
    for i in range(knots.size-1):
        # pts, ws = get_inegration_points(basis[0].deg*3,knots[i],knots[i+1])
        pts = knots[i]+(Pts+1)*0.5*(knots[i+1]-knots[i])
        ws = Ws*(knots[i+1]-knots[i])/2
        
        pts1 += list(pts)
        ws1 += list(ws)
        
    pts1 = np.array(pts1)
    ws1 = np.array(ws1)
    
    knots =  basis[1].knots[basis[1].deg:-basis[1].deg] #np.unique(basis[1].knots)
    pts2 = []
    ws2 = []
    for i in range(knots.size-1):
        # pts, ws = get_inegration_points(basis[1].deg*3,knots[i],knots[i+1])
        pts = knots[i]+(Pts+1)*0.5*(knots[i+1]-knots[i])
        ws = Ws*(knots[i+1]-knots[i])/2
        
        pts2 += list(pts)
        ws2 += list(ws)
    pts2 = np.array(pts2)
    ws2 = np.array(ws2)
    
    knots = basis[2].knots[basis[2].deg:-basis[2].deg] # np.unique(basis[2].knots)
    pts3 = []
    ws3 = []
    for i in range(knots.size-1):
        # pts, ws = get_inegration_points(basis[2].deg*3,knots[i],knots[i+1])
        pts = knots[i]+(Pts+1)*0.5*(knots[i+1]-knots[i])
        ws = Ws*(knots[i+1]-knots[i])/2
        
        pts3 += list(pts)
        ws3 += list(ws)
    pts3 = np.array(pts3)
    ws3 = np.array(ws3)
       

    B1 = basis[0](pts1)
    B2 = basis[1](pts2)
    B3 = basis[2](pts3)      
    dB1 = basis[0](pts1,derivative=True)
    dB2 = basis[1](pts2,derivative=True)
    dB3 = basis[2](pts3,derivative=True)
                
    for m in range(N[0]):
        i1 = max([m-basis[0].deg,0])
        i2 = min([m+basis[0].deg+1,N[0]])
        p1 = pts1[i1*nw:i2*nw]
        w1 = ws1[i1*nw:i2*nw]
        omega_inv_all = 1/eval_omega_3d(Ps, basis, [p1, pts2, pts3],to_tf=True)
        if kappa_ref != None:
            P1 = oe.einsum('i,j,k->ijk',p1,pts2*0+1,pts3*0+1)
            P2 = oe.einsum('i,j,k->ijk',p1*0+1,pts2,pts3*0+1)
            P3 = oe.einsum('i,j,k->ijk',p1*0+1,pts2*0+1,pts3)
            kappa_val = tn.tensor(kappa_ref(P1,P2,P3))
            omega_inv_all = omega_inv_all * kappa_val
        G11,G21,G31 = eval_G([p1, pts2, pts3],basis,Ps,(True,False,False))
        G12,G22,G32 = eval_G([p1, pts2, pts3],basis,Ps,(False,True,False))
        G13,G23,G33 = eval_G([p1, pts2, pts3],basis,Ps,(False,False,True))
        for n in range(N[1]):
            for o in range(N[2]):
                # print(m,n,o)
                              
                tme = datetime.datetime.now()
                j1 = max([n-basis[1].deg,0])
                j2 = min([n+basis[1].deg+1,N[1]])
                k1 = max([o-basis[2].deg,0])
                k2 = min([o+basis[2].deg+1,N[2]])
                        
                p2 = pts2[j1*nw:j2*nw]
                p3 = pts3[k1*nw:k2*nw]
                
                
                w2 = ws2[j1*nw:j2*nw]
                w3 = ws3[k1*nw:k2*nw]
                
                # timer.tic()
                
                omega_inv = omega_inv_all[:,j1*nw:j2*nw,k1*nw:k2*nw]
                
                g11 = G11[:,j1*nw:j2*nw,k1*nw:k2*nw]
                g12 = G12[:,j1*nw:j2*nw,k1*nw:k2*nw]
                g13 = G13[:,j1*nw:j2*nw,k1*nw:k2*nw]
                g21 = G21[:,j1*nw:j2*nw,k1*nw:k2*nw]
                g22 = G22[:,j1*nw:j2*nw,k1*nw:k2*nw]
                g23 = G23[:,j1*nw:j2*nw,k1*nw:k2*nw]
                g31 = G31[:,j1*nw:j2*nw,k1*nw:k2*nw]
                g32 = G32[:,j1*nw:j2*nw,k1*nw:k2*nw]
                g33 = G33[:,j1*nw:j2*nw,k1*nw:k2*nw]
               
                h11,h12,h13 = (g22*g33-g23*g32,g13*g32-g12*g33,g12*g23-g13*g22)
                h21,h22,h23 = (g23*g31-g21*g33,g11*g33-g13*g31,g13*g21-g11*g23)
                h31,h32,h33 = (g21*g32-g22*g31,g12*g31-g11*g32,g11*g22-g12*g21)
    
                H = [[h11,h12,h13],[h21,h22,h23],[h31,h32,h33]]


                # timer.toc()
                
                # timer.tic()
                
                toadd = (i2-i1)*(j2-j1)*(k2-k1)
                rows_tmp = [[m,n,o]]*(i2-i1)*(j2-j1)*(k2-k1)
                cols_tmp = [[i,j,k] for i in range(i1,i2) for j in range(j1,j2) for k in range(k1,k2)]
                rows[idx:idx+toadd] = np.array(rows_tmp)
                cols[idx:idx+toadd] = np.array(cols_tmp)
                
                for alpha in range(3):
                    for beta in range(3):
                        tmp = H[alpha][0]*H[beta][0]+H[alpha][1]*H[beta][1]+H[alpha][2]*H[beta][2]
                        integrand = tmp * omega_inv
                        
                        
                        
                        Bm = dB1[m,i1*nw:i2*nw] if alpha==0 else B1[m,i1*nw:i2*nw]
                        Bn = dB2[n,j1*nw:j2*nw] if alpha==1 else B2[n,j1*nw:j2*nw]
                        Bo = dB3[o,k1*nw:k2*nw] if alpha==2 else B3[o,k1*nw:k2*nw]
                        Bi = dB1[i1:i2,i1*nw:i2*nw] if beta==0 else B1[i1:i2,i1*nw:i2*nw]
                        Bj = dB2[j1:j2,j1*nw:j2*nw] if beta==1 else B2[j1:j2,j1*nw:j2*nw]
                        Bk = dB3[k1:k2,k1*nw:k2*nw] if beta==2 else B3[k1:k2,k1*nw:k2*nw]



                        ints = oe.contract('ijk,i,j,k,ai,bj,ck->abc',integrand,Bm*w1,Bn*w2,Bo*w3,Bi,Bj,Bk)
                        
                        # print(m,n,o,idx,toadd,vals.shape)
                        vals[idx:idx+toadd] += ints.numpy().flatten()            
                        # timer.toc()
                        
                        # timer.tic()
                        
                        
                tme = datetime.datetime.now() -tme 
                        
                        
                idx += toadd
                            

                            
                # timer.toc()
    # rows = [r[2]*N[1]*N[0]+r[1]*N[0]+r[0] for r in rows] 
    # cols = [c[2]*N[1]*N[0]+c[1]*N[0]+c[0] for c in cols]  
    vals = vals[:idx]
    rows = rows[:idx,:]
    cols = cols[:idx,:]
    
    if normal_ordering:
        rows = [r[0]*N[1]*N[2]+r[1]*N[2]+r[2] for r in rows] 
        cols = [c[0]*N[1]*N[2]+c[1]*N[2]+c[2] for c in cols] 
    else:    
        rows = [r[2]*N[1]*N[0]+r[1]*N[0]+r[0] for r in rows] 
        cols = [c[2]*N[1]*N[0]+c[1]*N[0]+c[0] for c in cols]        
        
    # return rows, cols, vals      
    return scipy.sparse.coo_matrix((vals, (rows, cols)))   

def construct_sparse_from_tt(basis,Stt,additional_indices):

    N = [b.N for b in basis[:3]]
    
    rows = np.zeros((np.prod(N)*(2*basis[0].deg+1)*(2*basis[1].deg+1)*(2*basis[2].deg+1),3),dtype=int)
    cols = np.zeros((np.prod(N)*(2*basis[0].deg+1)*(2*basis[1].deg+1)*(2*basis[2].deg+1),3),dtype=int)
    vals = np.zeros((np.prod(N)*(2*basis[0].deg+1)*(2*basis[1].deg+1)*(2*basis[2].deg+1)))
    idx = 0
    
   
    cores = [c.numpy() for c in Stt.cores]
    for i in range(2+len(additional_indices),2,-1):
        cores[i-1] = oe.contract('ijkl,lo->ijko',cores[i-1],cores[i][:,additional_indices[i-3],additional_indices[i-3],:])

    for m in range(N[0]):
        for n in range(N[1]):
            for o in range(N[2]):
                i1 = max([m-basis[0].deg,0])
                i2 = min([m+basis[0].deg+1,N[0]])
                j1 = max([n-basis[1].deg,0])
                j2 = min([n+basis[1].deg+1,N[1]])
                k1 = max([o-basis[2].deg,0])
                k2 = min([o+basis[2].deg+1,N[2]])
                toadd = (i2-i1)*(j2-j1)*(k2-k1)
                rows_tmp = [[m,n,o]]*(i2-i1)*(j2-j1)*(k2-k1)
                cols_tmp = [[i,j,k] for i in range(i1,i2) for j in range(j1,j2) for k in range(k1,k2)]
                # cols_tmp = [[i,j,k] for k in range(k1,k2) for j in range(j1,j2) for i in range(i1,i2)]
                rows[idx:idx+toadd] = np.array(rows_tmp)
                cols[idx:idx+toadd] = np.array(cols_tmp)
                vals[idx:idx+toadd] = oe.contract('jk,klm,mn->jln',cores[0][0,m,i1:i2,:],cores[1][:,n,j1:j2,:],cores[2][:,o,k1:k2,0]).flatten()            
                idx += toadd

    normal_ordering = True 
    if normal_ordering:
        rows = [r[0]*N[1]*N[2]+r[1]*N[2]+r[2] for r in rows] 
        cols = [c[0]*N[1]*N[2]+c[1]*N[2]+c[2] for c in cols] 
    else:    
        rows = [r[2]*N[1]*N[0]+r[1]*N[0]+r[0] for r in rows] 
        cols = [c[2]*N[1]*N[0]+c[1]*N[0]+c[0] for c in cols]        
        
    # return rows, cols, vals      
    return scipy.sparse.coo_matrix((vals, (rows, cols)))   
                


def interpolate_iga(f, Basis, Xs):

    N = [b.N for b in Basis]    
    Xg = [b.greville() for b in Basis]
    
    corz = [np.linalg.inv(b(b.greville())).transpose() for b in Basis]
      
    
    X, Y, Z = eval_G(Xg,Basis,Xs)
    X = X.numpy()
    Y = Y.numpy()
    Z = Z.numpy()
    
    evals = f(np.hstack((X.reshape([-1,1]),Y.reshape([-1,1]),Z.reshape([-1,1])))).reshape(N)
    
    dofs = oe.contract('ijk,mi,nj,ok->mno',evals,corz[0],corz[1],corz[2]).numpy().reshape([-1,1])
    
    return dofs

def boundary_matrices(basis,opened = None, normal_ordering=True):
    
    N = [basis[0].N, basis[1].N, basis[2].N]
    
    rows = []
    cols = []
    vals = []
    if opened == None:
        opened = [[0,0],[0,0],[0,0]]
        
        
    for m in range(0 if opened[0][0] else 1,N[0] if opened[0][1] else (N[0]-1)):
        for n in range(0 if opened[1][0] else 1,N[1] if opened[1][1] else (N[1]-1)):
            for o in range(0 if opened[2][0] else 1,N[2] if opened[2][1] else (N[2]-1)):
                rows.append((m,n,o))
                cols.append((m,n,o))
                vals.append(1.0)
                
    if normal_ordering:
        rows = [r[0]*N[1]*N[2]+r[1]*N[2]+r[2] for r in rows] 
        cols = [c[0]*N[1]*N[2]+c[1]*N[2]+c[2] for c in cols] 
    else:    
        rows = [r[2]*N[1]*N[0]+r[1]*N[0]+r[0] for r in rows] 
        cols = [c[2]*N[1]*N[0]+c[1]*N[0]+c[0] for c in cols]        
        
    # return rows, cols, vals      
    Pin = scipy.sparse.coo_matrix((vals, (rows, cols)),shape=(np.prod(N),np.prod(N)))   
    Pbd = scipy.sparse.coo_matrix(([1.0]*np.prod(N), (list(range(np.prod(N))),list(range(np.prod(N)))))) - Pin

    return Pin, Pbd
