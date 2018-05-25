from numpy.random import randn
import numpy as np
import scipy as sp

def Ginibre(N, COMPLEX=False):
    if COMPLEX==True:
        out=(randn(N,N) + 1j*randn(N,N) )/ sp.sqrt(2*N)
    else:
        out=randn(N,N)/ sp.sqrt(N)
    return np.matrix(out)


def info_plus_noise(N,param_mat,variance=1,COMPLEX=False):
    X = Ginibre(N,COMPLEX)
    P = param_mat + variance*X
    P = np.matrix(P)
    if COMPLEX:
        Y = P.H.dot(P)
    else:
        Y = P.T.dot(P)

    return Y
