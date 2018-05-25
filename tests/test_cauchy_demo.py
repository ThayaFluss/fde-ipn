import unittest
import numpy as np
from demo_cauchy import *



class TestCauchyDemo(unittest.TestCase):
    def test_plot_density(self):
        dim = 40
        p_dim = 240
        scale = 0.2
        sigma = 0.1
        min_x = -10
        max_x = 50
        diag_A = np.arange(dim)/sp.sqrt(dim)
        A = rectangular_diag(diag_A,p_dim, dim)

        plot_density(A, sigma,scale, min_x, max_x,\
        jobname="only_density")

        dim_cauchy_vec = 100
        num_shot = 1
        plot_density(A, sigma,scale, min_x, max_x, \
        dim_cauchy_vec=dim_cauchy_vec,  num_shot=num_shot,jobname="single_shot")

        num_shot = 100
        plot_density(A, sigma,scale, min_x, max_x, \
        dim_cauchy_vec=dim_cauchy_vec,  num_shot=num_shot,jobname="multi_shot")
