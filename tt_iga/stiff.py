import torchtt as tntt
import torch as tn
import numpy as np
from tt_iga.aux_functions import *
import datetime


def stiffnessTT(Basis, Xs, eps=1e-12, rankinv=999, func=None, params = None, verb = False, qtt = True, func_reference = None):
    
    p1, w1 = points_basis(Basis[0],mult=2)
    p2, w2 = points_basis(Basis[1],mult=2)
    p3, w3 = points_basis(Basis[2],mult=2)
    ps = [p1,p2,p3]
    
    tme = datetime.datetime.now()
    Og_tt = omega_eval(Basis, [p1,p2,p3], Xs, eps)
    tme = datetime.datetime.now() - tme
    if verb: print('time omega' , tme,flush=True)

    if qtt: 
        Nqtt = qtt_shape(Og_tt, list(range(len(Og_tt.n))))
        if verb:
            print('QTT enabled:')
            print(list(Og_tt.n))
            print('  || ')
            print('  \/  ')
            print(Nqtt)
        No =list(Og_tt.n)
        Og_tt = reshape_tt(Og_tt,Nqtt)
    
    tme = datetime.datetime.now()
    # Ogi_tt = tt.tensor(1/Og_tt.full(),eps)
    Ogi_tt = invert_tt2(Og_tt,eps)
    tme = datetime.datetime.now() -tme
    if verb: print('time omega inv' , tme,' rank ',Ogi_tt.r,flush=True)
    if verb: print('invert error ',(Ogi_tt*Og_tt-tt.ones(Og_tt.n)).norm()/tt.ones(Og_tt.n).norm())
    Ogi_tt = Ogi_tt.round(eps)
    # print(Ogi_tt.r)
    # print(Ogi_tt)
    

    if func != None or func_reference != None:
        tmp = tt_meshgrid(params)
        if func_reference == None:
            X, Y, Z = GTT(ps, Basis, Xs, (False,False,False), eps)
            F_tt = tt.multifuncrs2([X, Y, Z]+[tt.kron(tt.ones(X.n[:3]),t) for t in tmp], func, eps,verb=0).round(eps)
            
            if verb: print('rank of Frtt is ',F_tt.r)
        else:
            
            F_tt = tt.multifuncrs2(tt_meshgrid(ps+params), func_reference, eps, verb=0).round(eps)
            if verb: print('rank of Ftt is ',F_tt.r)
        
   
    else:
        F_tt = tt.ones(Ogi_tt.n)

    if qtt:
        F_tt = reshape_tt(F_tt,Nqtt)  
    Ogi_tt = (Ogi_tt * F_tt).round(eps)
    
    tme = datetime.datetime.now()
    g11, g21, g31 = GTT(ps, Basis, Xs, (True,False,False), eps)
    g12, g22, g32 = GTT(ps, Basis, Xs, (False,True,False), eps)
    g13, g23, g33 = GTT(ps, Basis, Xs, (False,False,True), eps)
    if verb:
        print(g11.r) 
        print(g12.r) 
        print(g13.r) 
        print(g21.r) 
        print(g22.r) 
        print(g23.r) 
        print(g31.r) 
        print(g32.r) 
        print(g33.r) 
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
    
    B1 = Basis[0](p1).transpose()
    dB1 = Basis[0](p1,derivative=True).transpose()
    B2 = Basis[1](p2).transpose()
    dB2 = Basis[1](p2,derivative=True).transpose()
    B3 = Basis[2](p3).transpose()
    dB3 = Basis[2](p3,derivative=True).transpose()
    
    N = Xs[0].n
    S = None
    SS = None
    Hs = dict()
    
    # the size of the bands
    band_size = [b.deg for b in Basis]+[1]*len(N[3:])
    
    # tme = datetime.datetime.now()
    # Hi = [[(H[0][0]*Ogi_tt).round(eps),(H[0][1]*Ogi_tt).round(eps),(H[0][2]*Ogi_tt).round(eps)],[(H[1][0]*Ogi_tt).round(eps),(H[1][1]*Ogi_tt).round(eps),(H[1][2]*Ogi_tt).round(eps)],[(H[2][0]*Ogi_tt).round(eps),(H[2][1]*Ogi_tt).round(eps),(H[2][2]*Ogi_tt).round(eps)]]
    # tme = datetime.datetime.now() -tme
    # if verb: print('\ttime Hi ' , tme)

    if qtt: 
        for i in range(3):
            for j in range(3):
                H[i][j] = reshape_tt(H[i][j], Nqtt, eps)
                # print(H[i][j].r)

    for alpha in range(3):
        for beta in range(3):
            if verb: print('alpha, beta = ',alpha,beta)
            tme = datetime.datetime.now()

            tmp = H[alpha][0]*H[beta][0]+H[alpha][1]*H[beta][1]+H[alpha][2]*H[beta][2]
            tmp = tmp.round(eps,rankinv)
            tme = datetime.datetime.now() -tme
            if verb: print('\ttime 1 ' , tme)

          #   print('rank',tmp.r,' size',tmp.n)
            # tme = datetime.datetime.now()
            # cores_list = [tt_core_eye(c) for c in Ogi_tt.to_list(Ogi_tt)] 
            # tmp2 = tt.amen.amen_mv(tt.matrix().from_list(cores_list),tmp,eps)
            # tme = datetime.datetime.now() -tme
            # print('time alternative ',tme)
            tme = datetime.datetime.now()
            # tmp = tmp*Ogi_tt
            tmp = inverse_mult_pointwise(Og_tt,tmp)*F_tt

           #  print('Rank of product',tmp.r)
            tmp = tmp.round(eps,rankinv)
            tme = datetime.datetime.now() -tme
            if verb: print('\ttime 2 ' , tme,' rank ',tmp.r)
            
            # print('ERR ',(tmp-tmp2).norm()/tmp.norm())

            if qtt: tmp = reshape_tt(tmp,No)
            
            # tmp = H[alpha][0]*Hi[beta][0]+H[alpha][1]*Hi[beta][1]+H[alpha][2]*Hi[beta][2]
            # tmp = tmp.round(eps,rankinv)
            
            tme = datetime.datetime.now()
            cores = tmp.to_list(tmp)
            
            tme = datetime.datetime.now()
            # cores[0] = np.einsum('rjs,j,jm,jn->rmns',cores[0],w1,dB1 if alpha==0 else B1,dB1 if beta==0 else B1)
            # print(cores[0].shape,w1.shape)
            cores[0] = tf.einsum('rjs,j->rjs',cores[0],w1)
            tmp = tf.einsum('jm,jn->jmn',dB1 if alpha==0 else B1,dB1 if beta==0 else B1)
            cores[0] = tf.einsum('rjs,jmn->rmns',cores[0],tmp)
            tme = datetime.datetime.now() -tme
            if verb: print('\t\ttime ' , tme)
            tme = datetime.datetime.now()
            # cores[1] = np.einsum('rjs,j,jm,jn->rmns',cores[1],w2,dB2 if alpha==1 else B2,dB2 if beta==1 else B2)
            cores[1] = tf.einsum('rjs,j->rjs',cores[1],w2)
            tmp = tf.einsum('jm,jn->jmn',dB2 if alpha==1 else B2,dB2 if beta==1 else B2)
            cores[1] = tf.einsum('rjs,jmn->rmns',cores[1],tmp)
            tme = datetime.datetime.now() -tme
            if verb: print('\t\ttime ' , tme)
            tme = datetime.datetime.now()
            # cores[2] = np.einsum('rjs,j,jm,jn->rmns',cores[2],w3,dB3 if alpha==2 else B3,dB3 if beta==2 else B3)
            cores[2] = tf.einsum('rjs,j->rjs',cores[2],w3)
            tmp = tf.einsum('jm,jn->jmn',dB3 if alpha==2 else B3,dB3 if beta==2 else B3)
            cores[2] = tf.einsum('rjs,jmn->rmns',cores[2],tmp)
            tme = datetime.datetime.now() -tme
            if verb: print('\t\ttime ' , tme)
            tme = datetime.datetime.now()
            for i in range(3,len(cores)):
                cores[i] = tf.einsum('rjs,mj,nj->rmns',cores[i],np.eye(cores[i].shape[1]),np.eye(cores[i].shape[1]))
            tme = datetime.datetime.now() -tme
            if verb: print('\t\ttime ' , tme)
            tme = datetime.datetime.now() -tme
            if verb: print('\ttime 3 ' , tme)
            
            tme = datetime.datetime.now()
            ss = tt.tensor().from_list([bandcore2ttcore(cores[i].numpy(),band_size[i]) for i in range(len(cores)) ])
            # s = tt.matrix().from_list(cores).round(eps)
            # if verb: print('S',S)
            # if verb: print('s',s)
            # S = s if S==None else S+s 
            SS = ss if SS==None else SS+ss 
            
            
            tme = datetime.datetime.now() -tme
            if verb: print('\ttime 4 ' , tme)
        tme = datetime.datetime.now()
        # S = S.round(eps)
        SS = SS.round(eps)
        
        tme = datetime.datetime.now() -tme
        if verb: print('\ttime ROUND ' , tme)
    cores = SS.to_list(SS)
    # print([c.shape for c in cores],N,band_size)
    SS = tt.matrix().from_list([ttcore2bandcore(cores[i],N[i],band_size[i]) for i in range(len(N))])
    # print('ERRRRR ',(S-SS).norm()/S.norm())
    return SS