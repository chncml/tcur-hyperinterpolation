import numpy as np
try:
    from scipy.special import eval_legendre
except ImportError:
    def eval_legendre(n, x):
        return np.polynomial.legendre.Legendre.basis(n)(x)

class LegendreBasis:
    """Orthonormal Legendre polynomials on [-1,1]."""
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, x, n):
        if n > self.degree:
            raise ValueError(f"n={n} exceeds degree={self.degree}")
        return eval_legendre(n, x) * np.sqrt((2*n + 1) / 2.0)
