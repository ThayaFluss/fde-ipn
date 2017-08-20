# FDE-IPN


## Free Deterministic Equivalents of the information plus noise random matrices
This is a simple tool to compute the probability densities of eigenvalue distributions of the information plus noise random matrices.

## Description
We consider a information plus noise random matrices as follows:
for a given N x N matrix A and a real number v > 0, define a random matrix

####    Y = A + vZ,

where Z is a (real or complex) Ginibre random matrix (i.e. whose entries are i.i.d. and distributed with N(0, sqrt(1/N)) ).
If the size N is large enough, eigenvalue distribution of YY^* can be approximated by  deterministic probability distribution on positive real line.
This tool computes the probability density function of the deterministic eigenvalue distribution.


The argorithm are based on the papers [Operator-valued Semicircular Elements: Solving A Quadratic
Matrix Equation with Positivity Constraints (J. William Helton, Reza Rashidi Far and Roland Speicher, 2007, IMRN)](http://www.math.ucsd.edu/~helton/BILLSPAPERSscanned/HRS07.pdf) and
[Free Probability Theory: Deterministic Equivalents and Combinatorics(Carlos Vargas Obieta, 2015)](http://d-nb.info/1070819107/34)


##　DEMO
