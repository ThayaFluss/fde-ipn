from numpy.random import randn
import numpy as np
import scipy as sp
from matrix_util import *


def Ginibre(M, N, COMPLEX=False):
    if COMPLEX==True:
        out=(randn(M,N) + 1j*randn(M,N) )/ sp.sqrt(2*N)
    else:
        out=randn(M,N)/ sp.sqrt(N)
    return np.matrix(out)

def haar_unitary(M, COMPLEX=False):
    G = Ginibre(M,M,COMPLEX)
    U,D,V = np.linalg.svd(G)
    return U

def random_from_diag(M,N,zero_dim=0, min_singular=0,COMPLEX=False):
    array  = np.random.uniform( min_singular, 1, N)
    for i in range(zero_dim):
        array[i] = 0
    D = rectangular_diag(array, M,N)
    U = haar_unitary(M, COMPLEX)
    V = haar_unitary(N, COMPLEX)
    return U @ D @ V

def info_plus_noise_symm(p_dim , dim, param_mat, sigma=1, COMPLEX=False):
    out = np.zeros([2*p_dim, 2*p_dim])
    C = Ginibre(p_dim , dim , COMPLEX)
    for i in range(p_dim):
        for j in range(dim):
            out[p_dim + i][j] = C[i,j]
            out[j][p_dim +i] = np.conj(C[i,j])

    return sigma*out



def info_plus_noise(param_mat,sigma=1,COMPLEX=False):
    p_dim = param_mat.shape[0]
    dim = param_mat.shape[1]
    assert np.allclose(param_mat.shape, [p_dim,dim])
    X = Ginibre(p_dim, dim,COMPLEX)
    P = param_mat + sigma*X
    P = np.matrix(P)
    if COMPLEX:
        Y = P.H.dot(P)
    else:
        Y = P.T.dot(P)

    return Y
