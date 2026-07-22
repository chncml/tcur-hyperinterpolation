import numpy as np
from numpy.linalg import pinv, norm, qr
from .coefficient import (compute_subtensor, compute_mode_completion_matrix,
                          compute_factor_matrices, compute_fiber_matrix,
                          discrete_inner_product)
from .tensor_ops import mode_n_unfold, st_hosvd, reconstruct_tucker
from .utils import (min_singular_ratio, basis_degrees, make_empty_index_sets,
                    extend_index_sets)
from itertools import product

# ----- Algorithm 5.1 (factor‑based) -----
def algorithm_5_1(f, basis_list, quad_nodes, quad_weights, b, tau,
                  max_iters=30, min_sizes=None):
    N = len(b)
    degrees = basis_degrees(basis_list)
    I_sets = make_empty_index_sets(N)
    if min_sizes is None:
        min_sizes = [1]*N
    K = 0
    for k in range(1, max_iters+1):
        extend_index_sets(I_sets, b, k, [d+1 for d in degrees])
        ratios = []
        for n in range(N):
            C = compute_mode_completion_matrix(f, basis_list, quad_nodes,
                                               quad_weights, degrees, I_sets, n)
            ratios.append(min_singular_ratio(C))
        enough = all(len(I_sets[n]) >= min_sizes[n] for n in range(N))
        if enough and max(ratios) < tau:
            K = k
            break
    if K == 0:
        K = max_iters
    factors, U_inv_norms = compute_factor_matrices(f, basis_list, quad_nodes,
                                                   quad_weights, I_sets)
    G = compute_subtensor(f, basis_list, quad_nodes, quad_weights, I_sets)
    cond_factors = [norm(F, 2) for F in factors]
    return G, factors, I_sets, K, cond_factors, U_inv_norms

# ----- Algorithm 5.2 (core‑based) -----
def algorithm_5_2(f, basis_list, quad_nodes, quad_weights, b, tau,
                  max_iters=30, min_sizes=None):
    N = len(b)
    degrees = basis_degrees(basis_list)
    I_sets = make_empty_index_sets(N)
    if min_sizes is None:
        min_sizes = [1]*N
    K = 0
    for k in range(1, max_iters+1):
        extend_index_sets(I_sets, b, k, [d+1 for d in degrees])
        G = compute_subtensor(f, basis_list, quad_nodes, quad_weights, I_sets)
        ratios = [min_singular_ratio(mode_n_unfold(G, n)) for n in range(N)]
        enough = all(len(I_sets[n]) >= min_sizes[n] for n in range(N))
        if enough and max(ratios) < tau:
            K = k
            break
    if K == 0:
        K = max_iters
    G_final = compute_subtensor(f, basis_list, quad_nodes, quad_weights, I_sets)
    factors, U_inv_norms = compute_factor_matrices(f, basis_list, quad_nodes,
                                                   quad_weights, I_sets)
    cond_factors = [norm(F, 2) for F in factors]
    return G_final, factors, I_sets, K, cond_factors, U_inv_norms

# ----- Algorithm 5.4 (fiber‑based) -----
def algorithm_5_4(f, basis_list, quad_nodes, quad_weights, b, b_prime,
                  tau1, tau2, max_iters=30, min_sizes=None):
    # Step 1: run Algorithm 5.2 to get core and I_sets
    G_core, _, I_sets, K_G, _, _ = algorithm_5_2(
        f, basis_list, quad_nodes, quad_weights, b, tau1,
        max_iters=max_iters, min_sizes=min_sizes
    )
    N = len(b)
    degrees = basis_degrees(basis_list)
    J_n_m = [[[] for _ in range(N)] for _ in range(N)]
    K_C = 0
    for k in range(1, max_iters+1):
        for n in range(N):
            for m in range(N):
                if m == n:
                    continue
                start = (k-1)*b_prime[m] + 1
                end = min(k*b_prime[m], degrees[m]+1)
                if start <= end:
                    J_n_m[n][m].extend(range(start-1, end))
        ratios = []
        for n in range(N):
            other_lists = [J_n_m[n][m] for m in range(N) if m != n]
            if any(len(lst)==0 for lst in other_lists):
                ratios.append(1.0)
                continue
            C_prime = compute_fiber_matrix(f, basis_list, quad_nodes, quad_weights,
                                           degrees, n, other_lists)
            ratios.append(min_singular_ratio(C_prime))
        if max(ratios) < tau2:
            K_C = k
            break
    if K_C == 0:
        K_C = max_iters
    # Build final factors
    factors = []
    U_inv_norms = []
    for n in range(N):
        other_lists = [J_n_m[n][m] for m in range(N) if m != n]
        C_prime = compute_fiber_matrix(f, basis_list, quad_nodes, quad_weights,
                                       degrees, n, other_lists)
        U_prime = C_prime[I_sets[n], :]
        U_pinv = pinv(U_prime)
        factors.append(C_prime @ U_pinv)
        U_inv_norms.append(norm(U_pinv, 2))
    cond_factors = [norm(F, 2) for F in factors]
    return G_core, factors, I_sets, K_G, K_C, cond_factors, U_inv_norms

# ----- TCUR to Tucker recompression (Algorithm A.1) -----
def tcur_to_tucker(G, factors, target_ranks):
    N = G.ndim
    Q_list = []
    R_list = []
    for n in range(N):
        Q, R = qr(factors[n], mode='reduced')
        Q_list.append(Q)
        R_list.append(R)
    # Build intermediate core
    hat_G = reconstruct_tucker(G, R_list)
    S_core, W = st_hosvd(hat_G, target_ranks)
    V_final = [Q_list[n] @ W[n] for n in range(N)]
    return S_core, V_final
