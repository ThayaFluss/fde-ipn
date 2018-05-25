# FDE-IPN

### Free Deterministic Equivalents of the information plus noise random matrices
This is a simple tool to compute the probability densities of eigenvalue distributions of the information plus noise random matrices.

## Description
We consider an information plus noise random matrices as follows:
for a given p x d matrix A and a real number \sigma > 0, define a random matrix

####    W = (A + \sigma Z)^T (A +  \sigma Z),

where Z is a p x d (real or complex) Ginibre random matrix (i.e. whose entries are i.i.d. and distributed with N(0, sqrt(1/d)) ).
If the size N is large enough, eigenvalue distribution of W can be approximated by  deterministic probability distribution on positive real line.
This tool computes the probability density function of the deterministic eigenvalue distribution.


Our argorithm is based on the papers [Operator-valued Semicircular Elements: Solving A Quadratic
Matrix Equation with Positivity Constraints (J. William Helton, Reza Rashidi Far and Roland Speicher, 2007, IMRN)](http://www.math.ucsd.edu/~helton/BILLSPAPERSscanned/HRS07.pdf) and
[Free Probability Theory: Deterministic Equivalents and Combinatorics(Carlos Vargas Obieta, 2015)](http://d-nb.info/1070819107/34)


## DEMO
* p=240,d=40, A = rectangular_diag(0,0.1,0.2, ..., 3.9), v = 0.1:
![MP](../images/plot_density/240by40.png)


* p=d=64, A = 0, v = 0.1 (Marchenko-Pastur type):
![MP](https://github.com/ThayaFluss/fde-ipn/blob/master/images/plot_density/MP.png)

* p=d=64, A = 1, v = 0.1:
![identity](https://github.com/ThayaFluss/fde-ipn/blob/master/images/plot_density/identity.png)

* p=d=64, A = diag(0.2,0.2..., 0.5, 0.5,...) (32 of 0.2 and 32 0.5), v = 0.1:
![2-5](https://github.com/ThayaFluss/fde-ipn/blob/master/images/plot_density/2-5.png)

* p=d=64, A = diag(1,2,3, ..., 64)/80, v = 0.1:
![64](https://github.com/ThayaFluss/fde-ipn/blob/master/images/plot_density/64.png)

## Requirement
python 3, numpy, scipy, matplotlib. We recommend [Anaconda](https://www.continuum.io/downloads).

## Installation

```bash
$ git clone https://github.com/ThayaFluss/fde-ipn.git
```
## Unit test
```bash
$ cd fde-ipn
$ python -m unittest tests/*.py
```


## Usage
```bash
$ cd fde-ipn
$ ipython
```
```python
import numpy as np
from matrix_util import *
from demo_cauchy import *
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
num_shot = 200 ### The number of sample matrices. If we set num_shot=0, only density is plotted.
plot_density(A, sigma,scale, min_x, max_x, \
num_shot=num_shot,jobname="2-5")  
```


To plot all figures in this README, try
```bash
$ cd fde-ipn
$ python demo_plots.py
```


## License

  [MIT](https://github.com/ThayaFluss/fde-ipn/blob/master/LICENSE)
