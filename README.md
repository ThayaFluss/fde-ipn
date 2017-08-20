# FDE-IPN

### Free Deterministic Equivalents of the information plus noise random matrices
This is a simple tool to compute the probability densities of eigenvalue distributions of the information plus noise random matrices.

## Description
We consider an information plus noise random matrices as follows:
for a given N x N matrix A and a real number v > 0, define a random matrix

####    Y = A + vZ,

where Z is a (real or complex) Ginibre random matrix (i.e. whose entries are i.i.d. and distributed with N(0, sqrt(1/N)) ).
If the size N is large enough, eigenvalue distribution of Y^* Y can be approximated by  deterministic probability distribution on positive real line.
This tool computes the probability density function of the deterministic eigenvalue distribution.


Our argorithm is based on the papers [Operator-valued Semicircular Elements: Solving A Quadratic
Matrix Equation with Positivity Constraints (J. William Helton, Reza Rashidi Far and Roland Speicher, 2007, IMRN)](http://www.math.ucsd.edu/~helton/BILLSPAPERSscanned/HRS07.pdf) and
[Free Probability Theory: Deterministic Equivalents and Combinatorics(Carlos Vargas Obieta, 2015)](http://d-nb.info/1070819107/34)


## DEMO
* N=64, A = 0, v = 1 (Marchenko-Pastur type):
![MP](https://github.com/ThayaFluss/fde-ipn/blob/master/images/MP.png)

* N=64, A = 1, v = 0.1:
![identity](https://github.com/ThayaFluss/fde-ipn/blob/master/images/identity.png)

* N=64, A = diag(2,2..., 5, 5,...) (32 of 2 and 32 5), v = 0.1:
![2-5](https://github.com/ThayaFluss/fde-ipn/blob/master/images/2-5.png)

* N=64, A = diag(1,2,3, ..., 64)/8, v = 0.1:
![64](https://github.com/ThayaFluss/fde-ipn/blob/master/images/64.png)

## Requirement
python 2 or 3, numpy, scipy, matplotlib. We recommend [Anaconda](https://www.continuum.io/downloads).

## Installation

```bash
$ git clone https://github.com/ThayaFluss/fde-ipn.git
```

## Usage
```bash
$ cd fde-ipn
$ ipython
 import numpy as np
 from cauchy import SemiCircular
 sc = SemiCircular()
```

* A useful function to comparing  the probability density and the emprical eigenvalue distirbution :
```python
 A = np.identity(64) ## Modify here 
 variance = 0.1      ## as you want
 sc.plot_density_info_plus_noise(A, variance)
```
Then we get a figure such as demo.

* If you need a value of the density at x:
```python
 x = 1.1             ## Modify 
 A = np.identity(64) ## here 
 variance = 0.1      ## as you want
 sc.square_density(x,A, variance)
```
We get the result such as 
```python
out: array([ 2.0288012])
```
* To get the probability density on an interval:
```python
 x = np.linspace(0.01, 2, 201) ## Modify
 A = np.identity(64)           ## here 
 variance = 0.1                ## as you want
 y =sc.square_density(x,A, variance)
```
To plot the result y:
```python
import matplotlib.pylab as plt
plt.plot(x,y)
plt.show()
```

## License

  [MIT](https://github.com/ThayaFluss/fde-ipn/blob/master/LICENSE)
 
