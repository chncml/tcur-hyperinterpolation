import numpy as np
from .basis import LegendreBasis
from .quadrature import gauss_legendre
from .tensor_ops import mode_n_unfold
from .utils import basis_degrees
from itertools import product

# ----- inner product counter (for performance measurements) -----
_inner_counter = 0

def reset_inner_counter():
    global _inner_counter
    _inner_counter = 0

def get_inner_counter():
    return _inner_counter

def _increment_counter(n=1):
    global _inner_counter
    _inner_counter += n

# ----- basic building blocks -----
def weighted_basis_matrix(basis, nodes, weights, indices):
    B = np.empty((len(nodes), len(indices)))
    for col, idx in enumerate(indices):
        B[:, col] = basis(nodes, idx) * weights
    return B

def sample_on_quadrature(f, quad_nodes):
    grids = np.meshgrid(*quad_nodes, indexing='ij')
    return f(*grids)

def coefficient_tensor_from_values(f_vals, basis_list, quad_nodes, quad_weights, index_sets):
    """Contract f_vals with weighted basis matrices mode by mode."""
    result = f_vals
    for mode, indices in enumerate(index_sets):
        B = weighted_basis_matrix(basis_list[mode], quad_nodes[mode],
                                  quad_weights[mode], indices)
        result = np.tensordot(result, B, axes=([0], [0]))
    return result

def discrete_inner_product(f, indices, basis_list, quad_nodes, quad_weights):
    """Compute <f, Phi_{indices}>_M with quadrature."""
    _increment_counter()
    N = len(indices)
    grids = np.meshgrid(*quad_nodes, indexing='ij')
    f_vals = f(*grids)
    prod = 1.0
    for n in range(N):
        basis_vals = basis_list[n](quad_nodes[n], indices[n])
        if n == 0:
            prod = basis_vals * quad_weights[n]
        else:
            prod = np.multiply.outer(prod, basis_vals * quad_weights[n])
    return np.sum(f_vals * prod)

# ----- full coefficient tensor -----
def compute_full_coefficient_tensor(f, basis_list, quad_nodes, quad_weights):
    degrees = basis_degrees(basis_list)
    shape = tuple(d+1 for d in degrees)
    index_sets = [list(range(s)) for s in shape]
    _increment_counter(int(np.prod(shape)))
    f_vals = sample_on_quadrature(f, quad_nodes)
    return coefficient_tensor_from_values(f_vals, basis_list, quad_nodes,
                                          quad_weights, index_sets)

# ----- subtensor extraction (for given index sets) -----
def compute_subtensor(f, basis_list, quad_nodes, quad_weights, index_sets):
    shape = [len(s) for s in index_sets]
    if any(length == 0 for length in shape):
        return np.zeros(shape)
    _increment_counter(int(np.prod(shape)))
    f_vals = sample_on_quadrature(f, quad_nodes)
    return coefficient_tensor_from_values(f_vals, basis_list, quad_nodes,
                                          quad_weights, index_sets)

def compute_mode_completion_matrix(f, basis_list, quad_nodes, quad_weights,
                                   degrees, I_sets, mode):
    """Build C_n matrix for mode n completion."""
    index_sets = [
        list(range(degree+1)) if m == mode else I_sets[m]
        for m, degree in enumerate(degrees)
    ]
    C_tensor = compute_subtensor(f, basis_list, quad_nodes, quad_weights,
                                 index_sets)
    return mode_n_unfold(C_tensor, mode)

def compute_factor_matrices(f, basis_list, quad_nodes, quad_weights, I_sets):
    """Compute factors C_n U_n^† and norms of pinv(U_n)."""
    from numpy.linalg import pinv, norm
    degrees = basis_degrees(basis_list)
    factors = []
    U_inv_norms = []
    for n in range(len(I_sets)):
        C_matrix = compute_mode_completion_matrix(f, basis_list, quad_nodes,
                                                  quad_weights, degrees,
                                                  I_sets, n)
        U = C_matrix[I_sets[n], :]
        U_pinv = pinv(U)
        factors.append(C_matrix @ U_pinv)
        U_inv_norms.append(norm(U_pinv, 2))
    return factors, U_inv_norms

def compute_fiber_matrix(f, basis_list, quad_nodes, quad_weights, degrees,
                         mode, other_index_sets):
    """Build fiber matrix C' for Algorithm 5.4."""
    total_cols = np.prod([len(s) for s in other_index_sets]) if other_index_sets else 0
    C_prime = np.zeros((degrees[mode]+1, total_cols))
    col = 0
    for comb in product(*other_index_sets):
        for in_idx in range(degrees[mode]+1):
            indices = []
            pos = 0
            for m in range(len(degrees)):
                if m == mode:
                    indices.append(in_idx)
                else:
                    indices.append(comb[pos])
                    pos += 1
            C_prime[in_idx, col] = discrete_inner_product(
                f, tuple(indices), basis_list, quad_nodes, quad_weights
            )
        col += 1
    return C_prime
