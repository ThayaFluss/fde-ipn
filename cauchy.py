import numpy as np
import scipy as sp

from matrix_util import *
from random_matrices import *
import matplotlib.pyplot as plt
from timer import Timer


class SemiCircular(object):
    """Matrix valued SemiCircular."""
    def __init__(self):
        super(SemiCircular, self).__init__()
        self.arg = 0

    def eta(self, in_mat):
        M = in_mat.shape[0]
        #assert  M % 2 == 0 and  M == in_mat.shape[1]
        t2 = ntrace(in_mat[M/2:,M/2:])
        t1 = ntrace(in_mat[:M/2,:M/2])
        #assert t2 + t1 == np.trace(in_mat)/(M/2)
        out = np.zeros(M, dtype=np.complex)
        for i in range(M/2):
            out[i]= t2
        for i in range(M/2, M):
            out[i]= t1
        out = np.diag(out)

        return out


    def fixed_point(self, init_mat, var_mat , max_iter=1000, thres=1e-9):
        W = init_mat
        sub = thres + 1
        for it in range(max_iter):
            con = np.linalg.inv( self.eta(W)+ var_mat)
            W_new = 0.5*W + 0.5*con
            if it % 5 == 0:
                sub = np.linalg.norm(W_new-W)
                #print("iter=",it, ":", "sub=", sub,"rho_by_(0,0)=", W_new[0,0].real/sp.pi, "rho_by_trace:", ntrace(W_new).real/sp.pi)
                if sub < thres:
                    W = W_new
                    break
            W = W_new
        return W

    ### -jZG + \eta(G)G = 1
    ###  VW + \eta(W)W = 1
    def Cauchy(self, init_G, var_mat, variance=1):
        assert init_G.shape == var_mat.shape
        assert variance > 0 or variance ==0
        if variance == 0:
            G = np.linalg.inv(var_mat)
        else:
            init_W = 1j*init_G*variance
            var_mat *= -1j/variance
            W = self.fixed_point(init_W, var_mat)
            G = -1j*W/variance
        return G


    def square_density(self, x_array, param_mat,variance=1, eps=1e-9):
        size = param_mat.shape[1]
        assert param_mat.shape[0] == size
        param_mat = np.matrix(param_mat)
        e_param_mat = np.zeros(4*size**2, dtype=np.complex).reshape([2*size, 2*size])
        for k in range(size):
            for l in range(size):
                e_param_mat[k][size+l] = param_mat.H[k,l]
                e_param_mat[size+k][l] = param_mat[k,l]
        e_param_mat = np.matrix(e_param_mat)
        G = np.identity(2*size)*(1-1j)
        G = np.matrix(G)
        num = len(x_array)
        rho_list = []
        for i in range(num):
            x = x_array[i]
            z = sp.sqrt(x+1j*eps)
            L = z*np.identity(2*size)
            #L = Lambda(z,  2*size, -1)
            L = np.matrix(L)
            var_mat = L - e_param_mat
            G = self.Cauchy(G, var_mat, variance)
            G_2 = G / z   ### zG_2(z^2) = G(z)
            rho =  -ntrace(G_2).imag/sp.pi
            #print "(density_info_plus_noise)rho(", x, ")= " ,rho
            rho_list.append(rho)

        return np.array(rho_list)



    def Lambda(self, z,  size, eps=-1, test=0):
        assert z.imag > 0
        if eps < 0:
            #If not using Linearizaion Trick
            out = z*np.identity(size)
        elif test==1:
            #For Linearizaion TricK
            out = np.zeros((size,size), dtype=np.complex)
            for i in range(size/2):
                out[i][i]=z
            for i in range(size/2, size):
                out[i][i] = eps*1j
        else:
            out = np.zeros((size,size), dtype=np.complex)
            out[0][0] = z
            for i in range(1, size):
                out[i][i] = eps*1j

        return out




    def plot_density_info_plus_noise(self, param_mat,variance=1, eps=1e-9,  min_x = 0.01, max_x = 500,\
    resolution=0.2, num_sample = 100):
        size = param_mat.shape[1]
        assert param_mat.shape[0] == size
        param_mat = np.matrix(param_mat)

        evs_list =[]
        for i  in range(num_sample):
            evs= np.linalg.eigh(info_plus_noise(size, param_mat,variance, COMPLEX=True))[0]
            evs_list += evs.tolist()
        plt.figure()
        plt.hist(evs_list, bins=100, normed=True, label="empirical eigenvalues")

        max_x = min(max_x, max(evs_list) )
        min_x = max(min_x, min(evs_list))
        resolution = min(resolution,(max_x - min_x) /100)
        max_x += resolution*10
        Timer0 = Timer()
        Timer0.tic()

        e_param_mat = np.zeros(4*size**2, dtype=np.complex).reshape([2*size, 2*size])
        for k in range(size):
            for l in range(size):
                e_param_mat[k][size+l] = param_mat.H[k,l]
                e_param_mat[size+k][l] = param_mat[k,l]
        e_param_mat = np.matrix(e_param_mat)

        G = np.identity(2*size)*(1-1j)
        G = np.matrix(G)
        x = min_x
        x_list = []
        rho_list = []
        count =0
        while(x < max_x):
            print "(plot_density_info_plus_noise)x=", x
            x_list.append(x)
            z = sp.sqrt(x+1j*eps)
            L = self.Lambda(z,  2*size, -1)
            L = np.matrix(L)
            var_mat = L - e_param_mat
            G = self.Cauchy(G, var_mat, variance)
            G_2 = G / z   ### zG_2(z^2) = G(z)
            rho =  -ntrace(G_2).imag/sp.pi
            print "(plot_density_info_plus_noise)rho=", rho
            rho_list.append(rho)
            if x < 0.2:
                temp = 0.05
            else:
                temp = 1
            x += temp*resolution
            count += 1

        Timer0.toc()
        time = Timer0.total_time
        print("(plot_density_info_plus_noise)Total {} points, Took {} sec, {} sec/point".format(count, time, time/count ) )
        plt.plot(x_list,rho_list, label="probability density",color="red", lw = 2)
        plt.legend(loc="upper right")
        plt.show()


        return x_list, rho_list
