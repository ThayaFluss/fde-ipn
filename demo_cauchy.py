import numpy as np
import scipy as sp

from scipy import stats
from matrix_util import *
from random_matrices import *
import matplotlib.pyplot as plt
from timer import Timer


import time
import logging

#from cauchy import SemiCircular as SC
from fde_sc_c2 import SemiCircular as SC ###for rectanglar



def plot_density(A, sigma, scale, min_x=0.01, max_x=100, num_pt=200,\
 dim_cauchy_vec=100, num_shot=0,\
 jobname="sample_and_DE"):
    ### Compute singular_values of parameter
    p_dim = A.shape[0]
    dim = A.shape[1]
    _,diag_A,_ = np.linalg.svd(A)
    true_sc = SC(dim=dim, p_dim = p_dim, scale=scale)
    true_sc.set_params(diag_A,sigma)

    x_array = np.linspace(min_x, max_x, num_pt)
    true_density = true_sc.density_subordinaiton(x_array)


    plt.figure()
    plt.rc("text", usetex=True)
    if num_shot>0:
        sample = true_sc.ESD(num_shot=num_shot,dim_cauchy_vec=dim_cauchy_vec)
        plt.hist(sample, range=(min_x, max_x), bins=100, normed=True, label="sampling from true model \n perturbed by cauchy($0,\gamma$)",color="pink")

    plt.plot(x_array, true_density, linestyle="--", label="theoretical value", color="red")

    plt.legend(loc="upper left")
    dirname = "images/plot_density"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = "{}/{}.png".format(dirname, jobname)
    print ("output: {}".format(filename))
    plt.savefig(filename)
    plt.clf()
    plt.close()
