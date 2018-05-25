import numpy as np
import scipy as sp


from scipy import stats
from matrix_util import *
from random_matrices import *
import matplotlib.pyplot as plt
from timer import Timer

from itertools import chain #for ESD

import time
import logging



E = np.zeros([2,2,2,2])
for i in range(2):
    for j in range(2):
        E[i][j][i][j] = 1
matrix_units = np.asarray( E, np.complex128)
eye2 = np.eye(2,dtype=np.complex128)
d00eta =  matrix_units[1][1]
d01eta =  np.zeros([2,2], np.complex128)
d10eta =  np.zeros([2,2], np.complex128)
d11eta =  matrix_units[0][0]

J_eta = np.asarray([[d00eta, d01eta], [d10eta, d11eta]])

i_TEST_MODE= False

#@jitclass(spec)
class SemiCircular(object):
    """Matrix valued SemiCircular."""
    def __init__(self,dim=1,p_dim=-1, scale=1e-1):
        super(SemiCircular, self).__init__()
        self.diag_A = np.asarray([0])
        self.sigma = 0
        self.scale= scale
        self.test_grads = False
        self.dim = dim
        ### rectangular
        ### p_dim \times dim
        if p_dim > 0:
            self.p_dim = p_dim
        else:
            self.p_dim = dim
        self.G= np.eye(2*self.dim)*(1-1j)
        self.grads = np.zeros( (self.dim+1, 2*self.dim, 2*self.dim), dtype=np.complex128)

        ### for subordination
        self.des = Descrete(self.diag_A)
        self.G2 = np.ones(2)*(-1j)
        self.grads2 = np.zeros((self.dim+1, 2,2),dtype = np.complex128)
        self.omega = np.ones(2)*1j
        self.omega_sc = np.ones(2)*1j

    def set_params(self, a,sigma):
        assert self.dim == a.shape[0]
        assert self.dim == a.size

        self.diag_A = a
        self.des = Descrete(self.diag_A, p_dim=self.p_dim)
        self.sigma = sigma

    def update_params(self, a,sigma):
        self.diag_A = a
        self.des.__init__(a,p_dim=self.p_dim)
        self.sigma = sigma

    def eta_array(self, in_mat):
        M = in_mat.shape[0]
        #assert  M % 2 == 0 and  M == in_mat.shape[1]
        half_M = int(M/2)
        t2 = np.trace(in_mat[half_M:,half_M:])/half_M
        t1 = np.trace(in_mat[:half_M,:])/half_M
        #assert t2 + t1 == np.trace(in_mat)/(half_M)
        out = np.empty(M, dtype=np.complex128)
        for i in range(half_M):
            out[i]= t2
        for i in range(half_M, M):
            out[i]= t1

        return out


    ###  G^{-1} = b - \eta(G)
    ### -jbW + \eta(W)W = 1
    ###  VW + \eta(W)W = 1
    #@jit
    def fixed_point(self, init_mat, var_mat , max_iter=100, thres=1e-7):
        W = init_mat
        size = W.shape[0]
        sub = thres + 1
        #timer = Timer()
        #timer.tic()
        flag = False
        for it in range(max_iter):
            sub = np.linalg.inv( self.eta(W)+ var_mat) - W
            sub*= 0.5
            if it > 1 and np.linalg.norm(sub) < thres*np.linalg.norm(W):
                flag = True
            W += sub
            if flag:
                break
        #timer.toc()
        #logging.info("cauchy time={}/ {}-iter".format(timer.total_time, it))
        return W
    #@jit
    def cauchy(self, init_G, var_mat,sigma):
        #assert init_G.shape == var_mat.shape
        #assert sigma > 0 or sigma ==0
        if abs(sigma) == 0:
            print(sigma)
            G = np.linalg.inv(var_mat)
        else:
            init_W = 1j*init_G*sigma
            var_mat *= -1j/sigma
            W = self.fixed_point(init_W, var_mat)
            G = -1j*W/sigma
        return G


    def ESD(self, num_shot, dim_cauchy_vec=0,COMPLEX=False):
        evs_list = []
        param_mat = rectangular_diag(self.diag_A, self.p_dim, self.dim)
        for n in range(num_shot):
            W = info_plus_noise(param_mat, self.sigma, COMPLEX)
            evs =  np.linalg.eigh(W)[0]

            c_noise =  sp.stats.cauchy.rvs(loc=0, scale=self.scale, size=dim_cauchy_vec)
            if dim_cauchy_vec >0:
                for k in range(dim_cauchy_vec):
                    evs_list.append( (evs - c_noise[k]).tolist())
            else:
                evs_list.append(evs.tolist())
        out = list(chain.from_iterable(evs_list))

        return out



    def ESD_symm(self, num_shot, dim_cauchy_vec=0,COMPLEX=False):
        evs_list = []
        param_mat = rectangular_diag(self.diag_A, self.p_dim, self.p_dim)
        for n in range(num_shot):
            W = info_plus_noise_symm(self.p_dim, self.dim, param_mat, self.sigma, COMPLEX)
            evs =  np.linalg.eigh(W)[0]

            c_noise =  sp.stats.cauchy.rvs(loc=0, scale=self.scale, size=dim_cauchy_vec)
            if dim_cauchy_vec >0:
                for k in range(dim_cauchy_vec):
                    evs_list.append( (evs - c_noise[k]).tolist())
            else:
                evs_list.append(evs.tolist())
        out = list(chain.from_iterable(evs_list))

        return out



    ##########################
    ###### Subordinatioin ####
    ##########################
    def cauchy_subordination(self, B, \
    init_omega,init_G_sc, max_iter=1000,thres=1e-7, TEST_MODE=i_TEST_MODE):
        des = self.des
        omega = init_omega
        flag = False;
        sc_g = init_G_sc
        for n in range(max_iter):
            assert omega.imag[0] > 0
            assert omega.imag[1] > 0

            sc_g = self.cauchy_2by2(omega, sc_g)
            sc_h = 1/sc_g - omega
            omega_transform = des.h_transform(sc_h + B) + B
            sub = omega_transform - omega
            if np.linalg.norm(sub) < thres:
                flag = True
            omega += sub
            if flag :
                break
        out = self.cauchy_2by2(omega, sc_g)
        omega_sc = 1/out - omega + B
        if TEST_MODE:
            G1 = out
            G2 = des.cauchy_transform(omega_sc)
            G3 = 1/(omega + omega_sc - B)
            assert ( np.allclose(G1, G2))
            assert ( np.allclose(G1, G3))
            assert ( np.allclose(G2, G3))

        return out, omega, omega_sc


    def rho(self, x, G, omega):
        z = x+1j*self.scale
        L = sp.sqrt(z)*np.ones(2)
        G,omega, omgega_sc = self.cauchy_subordination(B=L, init_omega=omega, init_G_sc=G)
        self.G2 = G
        G_out = G[0]/ sp.sqrt(z)
        rho =  - G_out.imag/sp.pi
        return rho, G, omega


    def rho_symm(self, x, G, omega):
        z = x+1j*self.scale
        L = z*np.ones(2)
        G,omega, omgega_sc = self.cauchy_subordination(B=L, init_omega=omega, init_G_sc=G)
        rho =- ntrace(G).imag/sp.pi
        return rho, G, omega



    def density_subordinaiton(self, x_array):
        num = len(x_array)
        omega = 1j*np.ones(2)
        G = -1j*np.ones(2)
        rho_list = []
        for i in range(num):
            rho, G, omega = self.rho(x_array[i], G, omega)
            if rho < 0:
                print(rho)
            #assert rho > 0
            rho_list.append(rho)

        return np.array(rho_list)


    def density_subordinaiton_symm(self, x_array):
        num = len(x_array)
        omega = 1j*np.eye(2,dtype=np.complex128)
        G = -1j*np.eye(2,dtype=np.complex128)
        rho_list = []
        for i in range(num):
            rho, G, omega = self.rho_symm(x_array[i], G, omega)
            if rho < 0:
                print(rho)
            #assert rho > 0
            rho_list.append(rho)

        return np.array(rho_list)

    def cauchy_2by2(self,Z,  G_init, max_iter=1000, thres=1e-7):
        G = G_init
        sigma = self.sigma
        flag = False
        for d in range(max_iter):
            eta = np.copy(G[::-1])
            eta[0] *=float(self.p_dim)/self.dim ### for recutangular matrix
            sub = 1/(Z - sigma**2*eta) -G
            sub *= 0.5
            if np.linalg.norm(sub) < thres:
                flag = True
            G += sub
            if flag:
                return G
        #logging.info("cauchy_2by2: sub = {} @ iter= {}".format(np.linalg.norm(sub),d))
        loggin.info("cauchy_2by2: reahed max_iter")
        return G_init


    ######## Derivations of SemiCircular
    ### transpose of tangent
    ### 2 x 2
    ### i  k
    ### \part f_k / \part x_i
    def eta_2by2(self,G):
        eta = np.copy(G[::-1])
        eta[0]*=float(self.p_dim)/self.dim ### for recutangular matrix
        return eta



class Descrete(object):
    """docstring for Descrete."""
    def __init__(self, a, p_dim=-1):
        super(Descrete, self).__init__()
        self.a = a
        self.dim = a.shape[0]
        if p_dim > 0:
            assert p_dim >= self.dim
            self.p_dim = p_dim
        else:
            self.p_dim = self.dim
        self.G = 0
        self.f = 0
        self.h = 0

    def cauchy_transform(self,W):
        #assert np.allclose(W.shape, [2,2])
        a = self.a

        sum_inv_det = np.sum( 1/(W[1]*W[0] - a*a) )
        G = [ (1/self.dim)*W[1]*sum_inv_det,\
         (1./self.p_dim)*(W[0]*sum_inv_det + (self.p_dim -self.dim)/W[1] ) ]

        """
        T = [ [W[1][1]*np.ones(self.dim), a - W[0][1]],\
            [a - W[1][0], W[0][0]*np.ones(self.dim)] ] \
              / (W[1][1]*W[0][0] - (W[0][1]-a)*(W[1][0]-a) )
        G = np.mean(T, axis=2)
        """
        return np.asarray(G)

    def f_transfrom(self, W):
        return 1/(self.cauchy_transform(W))

    def h_transform(self,W):
        return self.f_transfrom(W) - W
