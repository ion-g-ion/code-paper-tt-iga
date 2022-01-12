import torchtt as tntt
import torch as tn

def get_projectors(N,opened=None):

    cores = []
    for i in range(len(N)):
        tmp = tn.eye(N[i], dtype = tn.float64)
        if opened!=None:
            tmp[0,0] = 1 if opened[i][0]  else 0
            tmp[-1,-1] = 1 if opened[i][1]  else 0
            
        else:
            tmp[0,0] = 0
            tmp[-1,-1] = 0
        cores.append(tn.reshape(tmp,[1,N[i],N[i],1]))
    Pin_tt = tntt.TT(cores)
    Pbd_tt = tntt.eye(N) - Pin_tt
    
    return Pin_tt,Pbd_tt
