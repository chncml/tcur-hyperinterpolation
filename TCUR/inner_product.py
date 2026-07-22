import numpy as np
from .quadrature import sample_on_quadrature

_inner_product_count = 0

def reset_inner_counter():
    global _inner_product_count
    _inner_product_count = 0

def increment_inner_counter():
    global _inner_product_count
    _inner_product_count += 1

def get_inner_counter():
    global _inner_product_count
    return _inner_product_count

def discrete_inner_product(f, indices, basis_list, quad_nodes, quad_weights):
    global _inner_product_count
    _inner_product_count += 1
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

def weighted_basis_matrix(basis, nodes, weights, indices):
    B = np.empty((len(nodes), len(indices)))
    for col, idx in enumerate(indices):
        B[:, col] = basis(nodes, idx) * weights
    return B

def coefficient_tensor_from_values(f_vals, basis_list, quad_nodes, quad_weights, index_sets):
    result = f_vals
    for mode, indices in enumerate(index_sets):
        B = weighted_basis_matrix(basis_list[mode], quad_nodes[mode], quad_weights[mode], indices)
        result = np.tensordot(result, B, axes=([0], [0]))
    return result

def compute_full_coefficient_tensor(f, basis_list, quad_nodes, quad_weights):
    """Full coefficient tensor for all indices (0..degree)."""
    degrees = [b.degree for b in basis_list]
    shape = tuple(d + 1 for d in degrees)
    index_sets = [list(range(s)) for s in shape]
    f_vals = sample_on_quadrature(f, quad_nodes)
    return coefficient_tensor_from_values(f_vals, basis_list, quad_nodes, quad_weights, index_sets)
