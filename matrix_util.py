import scipy as sp
#from scipy.stats import wishart, chi2
from scipy import linalg
#discrete cosine transform
#from scipy.fftpack import dct
import numpy as np
import math
import random
import matplotlib.pyplot as plt
#from spec import *
import sys
import os
#this_dir = os.getcwd()
#sys.path.insert(0, this_dir)
from timer import Timer

def ntrace(matrix):
    return np.trace(matrix)/ matrix.shape[1]

def rectangular_diag(array, p_dim, dim):
    ell = array.size
    assert ell <= min(p_dim , dim)
    out = np.zeros([p_dim, dim])
    for i in range(ell):
        out[i][i] = array[i]
    return out


def nsubtrace(matrix, main_dim, sub_dim):
    assert np.allclose(matrix.shape , [main_dim*sub_dim, main_dim*sub_dim])
    A = np.array(matrix).reshape([main_dim, sub_dim, main_dim, sub_dim]).transpose([0,2,1,3])
    out = np.zeros([main_dim, main_dim], np.complex)
    for m in range(main_dim):
        for n in range(main_dim):
            out[m][n] = ntrace(A[m][n])
    return out


def get_moments(matrix, max_index=5):
    ERROR_DEBUG("(get_moments)start...")
    moments = np.array([1], dtype=float)
    M = matrix
    n = M.shape[0]
    M_power = M
    for i in range(1,max_index):
        moments = np.append(moments, np.trace(M_power) / n)
        M_power = np.dot(M_power, M)
    return moments




def get_sum(sequence):
    sum = sequence[0] + 2*np.sum(sequence[1:])
    print("(get_sum)sum=",sum )
    return sum


def L2_distance(M,N):
    if not M.shape == N.shape:
        print ("(L2_distance)Error, shape is different.")
        return -1
    dis_mat = np.dot(M-N, (M-N).transpose())
    n = dis_mat.shape[0]
    dis = np.matrix.trace(dis_mat)/float(n)
    return dis




    #sample covariacne matrix
def singular_values(X, normalized=False, COMPLEX=False):
    if COMPLEX:
        Z = np.dot(X, np.matrix(X).H)
    else:
        Z = np.dot(X,X.transpose())
    if normalized:
        n = len(Z[0])
        evs=linalg.eig(Z/float(n))[0]
    else:
        evs=linalg.eig(Z)[0]
    #evs=np.sort(evs, axis=None)
    if COMPLEX:
        evs = evs.real
    evs=list(evs)
    print ("-->calculate singular values done.")
    return evs

def plot_evs(X, job_name, activation_name="", small_threshold=-1, large_threshold=-1, TYPE = "Singular", COMPLEX = False):
    evs_list = []

    print("Get evs....")
    print("--shape=", X.shape)
    timer = Timer()
    timer.tic()
    if TYPE == "Singular" or "singular":
        evs=singular_values(X, COMPLEX = COMPLEX)
    elif TYPE == "Hermitian" or "Symmetric" or "symmetric":
        evs=np.linalg.eigh(X)
    else:
        evs = np.linalg.eig(X)
    timer.toc()
    print("...got evs. ")
    if large_threshold > small_threshold and large_threshold > 0:
        print("extract small and large...")
        evs_cut=list()
        for x in evs:
            #positve matrix does not have negative eigen value.
            if small_threshold < x and x < large_threshold:
                evs_cut.append(x)
        plt.hist(evs_cut, bins=100, normed=True)
    else:
        plt.hist(evs, bins=100, normed=True)

    #plt.show()
    evs_list_array = np.array(evs_list)
    #print evs_list_array
    #temp = np.ones((1,iteration))
    #print temp

    name = ("took {:.4f}s").format(timer.total_time)
    plt.title("{} {}".format(job_name, activation_name))
    plt.xlabel("Eigen values (of X X^T)")
    this_dir = os.getcwd()
    log_dir ="{}/../log".format(this_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filename="{}/{}x{}_{}{}.png".format(log_dir, X.shape[0],X.shape[1], job_name, activation_name)
    print("-->{}".format(filename))
    plt.savefig(filename)
    plt.clf()








def get_moments_by_fourier(sequence,max_index, division=10):

    def real_fourier_trans(t, index):
        sum = sequence[0]
        num = len(sequence)
        for n in  range(1,num):
            sum +=  2 * sequence[n] * sp.cos(n*t)
        return ( math.pow(sum, index) / (2*sp.pi) )

    print("(get_moments_by_fourier)sequence.size=", sequence.size)
    #print ("(get_moments)value at 0=", 2*sp.pi * fourier(0))
    moments = []
    print ("(get_moments_by_fourier)division=", division)
    timer = Timer()
    timer.tic()
    for index in range(max_index):
        integrate = 0
        for r in range(division):
            integrate += sp.integrate.quad(real_fourier_trans, 2.*sp.pi*r/division, 2.*sp.pi*(r+1)/division, index)[0]
        moments.append(integrate)
    timer.toc()
    print ("(get_moments_by_fourier)integrate took {:.3f}s]".format(timer.total_time))

    return np.array(moments)




def compare_moments(M, max_index=6):
    if max_index < 0:
        max_index = 0
    MTM = np.dot(M, M.transpose())

    """
    error = 0

    for i in range(MTM.shape[0]):
        for j in range(MTM.shape[1]):
            ran=  range(min(MTM.shape[0]-i, MTM.shape[1]-j))
            for h in ran:
                error += abs(MTM[i, j] - MTM[i+h, j+h])

    print("Toeplitz_error=",error)
    """

    sequence = MTM[0]
    #print ("sequence=", sequence)
    moments = get_moments_by_fourier(sequence, max_index)
    moments_direct = get_moments(MTM, max_index)
    #get_moments(M, 100000)
    #get_moments(M_edge)

    """
    evs_list = singular_values(MA)
    moments_evs =[]
    for i in range(max_index):
        moment = 0
        for ev in evs_list:
            moment += math.pow(ev, i)
        moments_evs.append(moment / len(evs_list))
        """
    print ("(compare_moments)moments_direct=\n", moments_direct)
    #print ("(compare_moments)moments_by_evs=\n", moments_evs)
    print ("(compare_moments)moments_by_fourier=\n", moments)
    for i in range(max_index):
        print( "(compare_moments)direct/fourier of ",i,"-th moment=", moments[i]/moments_direct[i])




def random_Toeplitz(size):
    out = np.zeros((size, size))
    sead = np.random.randn(2*size-1)

    for i in range(size):
        for j in range(size):
            out[i][j] = sead[i+j]

    return out


def generate_random_Toeplitz(row, column):
    return random_Toeplitz(row)
