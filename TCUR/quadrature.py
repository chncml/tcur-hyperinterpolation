import numpy as np

def gauss_legendre(n):
    """n-point Gauss-Legendre quadrature nodes and weights on [-1,1]."""
    return np.polynomial.legendre.leggauss(n)
