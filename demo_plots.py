import numpy as np
from matrix_util import *
from demo_cauchy import *

num_shot = 200

dim = 40
p_dim = 240
sigma = 0.1
min_x = -5
max_x = 20
diag_A = np.arange(dim)/10
A = rectangular_diag(diag_A,p_dim, dim)

scale = 1e-1
plot_density(A, sigma,scale, min_x, max_x, \
num_shot=num_shot,jobname="240by40")


dim = 64
p_dim = dim
scale = 1e-3
sigma = 0.1
min_x = -0.1
max_x = 0.1
diag_A = np.zeros(dim)
A = rectangular_diag(diag_A,p_dim, dim)
num_shot = 200
plot_density(A, sigma,scale, min_x, max_x, \
num_shot=num_shot,jobname="MP")



dim = 64
p_dim = dim
scale = 1e-3
sigma = 0.1
min_x = 0
max_x = 2
diag_A = np.ones(dim)
A = rectangular_diag(diag_A,p_dim, dim)
num_shot = 200
plot_density(A, sigma,scale, min_x, max_x, \
num_shot=num_shot,jobname="identity")



dim = 64
p_dim = dim
scale = 1e-3
sigma = 0.1
min_x = -0.1
max_x = 0.5
a = 5*np.ones(dim)
half = int(dim/2)
for i in range(half):
    a[i] =  2
diag_A = a/10
A = rectangular_diag(diag_A,p_dim, dim)
num_shot = 200
plot_density(A, sigma,scale, min_x, max_x, \
num_shot=num_shot,jobname="2-5")



dim = 64
p_dim = dim
scale = 1e-3
sigma = 0.1
min_x = -0.25
max_x = 1
a = np.arange(64) + 1
diag_A = a/80
A = rectangular_diag(diag_A,p_dim, dim)
num_shot = 200
plot_density(A, sigma,scale, min_x, max_x, \
num_shot=num_shot,jobname="64")
