import numpy as np
from numpy.linalg import svd, norm

def min_singular_ratio(matrix):
    if matrix.size == 0:
        return 0.0
    s = svd(matrix, compute_uv=False)
    frob = norm(matrix, 'fro')
    return s[-1] / frob if frob > 0 else 0.0

def basis_degrees(basis_list):
    return [b.degree for b in basis_list]

def make_empty_index_sets(n_modes):
    return [[] for _ in range(n_modes)]

def extend_index_sets(I_sets, block_sizes, k, upper_bounds):
    for n, block_size in enumerate(block_sizes):
        start = (k - 1) * block_size + 1
        end = min(k * block_size, upper_bounds[n])
        if start <= end:
            I_sets[n].extend(range(start - 1, end))
