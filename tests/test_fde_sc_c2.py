import unittest
import numpy as np
from fde_sc_c2 import *
from random_matrices import *
from matrix_util import *

class TestSmiCircular(unittest.TestCase):
    def test_set_params(self):
        d = 20
        p = 100
        sc = SemiCircular(dim=d, p_dim=p)
        diag_A = np.arange(d)/d
        sigma = 0.1
        sc.set_params(diag_A, sigma)

    def test_density_subordination(self):
        d = 50
        p = 100
        sc = SemiCircular(dim=d, p_dim=p)
        a = np.arange(d)/d
        sigma = 0.1
        sc.set_params(a, sigma)
        x_array = np.linspace(0.01, 10, 200)
        sc.density_subordinaiton(x_array)
