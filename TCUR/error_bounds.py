import numpy as np
from numpy.linalg import svd, pinv, norm
from .tensor_ops import mode_n_unfold, st_hosvd, reconstruct_tucker, column_indices_for_mode
from .coefficient import compute_mode_completion_matrix
from .utils import basis_degrees

def compute_tcur_error_bound(f, full_tensor, G, factors, basis_list,
                             quad_nodes, quad_weights, I_sets, R_rank):
    """
    Returns (E_exact, E_th, E_tight) for Theorem 4.4 and Remark 4.9.
    """
    N = full_tensor.ndim
    shape = full_tensor.shape
    approx_tcur = reconstruct_tucker(G, factors)
    E_exact = norm(full_tensor - approx_tcur)

    # Mode-n SVDs
    U_list, s_list, Vt_list = [], [], []
    for n in range(N):
        A_n = mode_n_unfold(full_tensor, n)
        U, s, Vt = svd(A_n, full_matrices=False)
        U_list.append(U); s_list.append(s); Vt_list.append(Vt)

    J_sets = [column_indices_for_mode(shape, I_sets, n) for n in range(N)]

    # Delta_Rn
    Delta = []
    for n in range(N):
        s = s_list[n]
        R = R_rank[n]
        if R < len(s):
            delta_sq = np.sum(s[R:]**2)
        else:
            delta_sq = 0.0
        Delta.append(np.sqrt(delta_sq))

    # alpha, beta
    alpha, beta = [], []
    for n in range(N):
        U = U_list[n]; V = Vt_list[n].T
        R = R_rank[n]
        U_R = U[:, :R]; V_R = V[:, :R]
        I_n = I_sets[n]; J_n = J_sets[n]
        if len(I_n)==0 or len(J_n)==0:
            alpha.append(np.inf); beta.append(np.inf); continue
        U_R_I = U_R[I_n, :]; V_R_J = V_R[J_n, :]
        norm_U_inv = norm(pinv(U_R_I), 2) if U_R_I.shape[0] >= U_R_I.shape[1] else np.inf
        norm_V_inv = norm(pinv(V_R_J), 2) if V_R_J.shape[0] >= V_R_J.shape[1] else np.inf
        norm_U_inv = min(norm_U_inv, 1e6); norm_V_inv = min(norm_V_inv, 1e6)
        a = norm_U_inv + norm_V_inv + 3*norm_U_inv*norm_V_inv + 1
        b = norm_U_inv + norm_V_inv + norm_U_inv*norm_V_inv + 1
        alpha.append(a); beta.append(b)

    # Bound sum
    prod_cond = 1.0
    total_bound = 0.0
    for n in range(N):
        if n > 0:
            prod_cond *= norm(factors[n-1], 2)
        degrees = basis_degrees(basis_list)
        C_matrix = compute_mode_completion_matrix(f, basis_list, quad_nodes,
                                                  quad_weights, degrees, I_sets, n)
        U_n = C_matrix[I_sets[n], :]
        norm_U_inv = norm(pinv(U_n), 2)
        term = alpha[n] * Delta[n] + beta[n] * norm_U_inv * (Delta[n]**2)
        total_bound += prod_cond * term

    # Tight bound (Remark 4.9)
    core_tucker, factors_tucker = st_hosvd(full_tensor, R_rank)
    tucker_approx = reconstruct_tucker(core_tucker, factors_tucker)
    err_full_tucker = norm(full_tensor - tucker_approx)
    err_tucker_tcur = norm(tucker_approx - approx_tcur)
    E_tight = err_full_tucker + err_tucker_tcur

    return E_exact, total_bound, E_tight

# ----- ε-Tucker rank search (used in validation_tucker_rank) -----
def find_epsilon_tucker_rank(tensor, eps, max_rank=None, verbose=False):
    """Greedy search for minimal Tucker rank with max-norm error ≤ eps."""
    N = tensor.ndim
    shape = tensor.shape
    if max_rank is None:
        max_rank = shape
    ranks = [1]*N
    core, factors = st_hosvd(tensor, ranks)
    approx = reconstruct_tucker(core, factors)
    err_max = np.max(np.abs(tensor - approx))
    while err_max > eps:
        improved = False
        for n in range(N):
            if ranks[n] < max_rank[n]:
                new_ranks = ranks.copy()
                new_ranks[n] += 1
                new_core, new_factors = st_hosvd(tensor, new_ranks)
                new_approx = reconstruct_tucker(new_core, new_factors)
                new_err = np.max(np.abs(tensor - new_approx))
                if new_err < err_max:
                    ranks = new_ranks; approx = new_approx; err_max = new_err
                    improved = True
                    if verbose:
                        print(f"Ranks {ranks}: max error {err_max:.3e}")
                    break
        if not improved:
            new_ranks = [min(r+1, max_rank[i]) for i, r in enumerate(ranks)]
            if all(nr==r for nr,r in zip(new_ranks, ranks)):
                break
            ranks = new_ranks
            core, factors = st_hosvd(tensor, ranks)
            approx = reconstruct_tucker(core, factors)
            err_max = np.max(np.abs(tensor - approx))
            if verbose:
                print(f"Ranks {ranks}: max error {err_max:.3e}")
    return ranks, approx, core, factors

def theoretical_rank_bounds(shape, eps):
    """Theoretical upper bounds from Theorem 4.1."""
    N = len(shape)
    R = [0]*N
    for n in range(N):
        product_prev = 1 if n==0 else np.prod(R[:n])
        product_future = 1 if n==N-1 else np.prod(shape[n+1:])
        term = shape[n] + product_prev * product_future + 1
        R[n] = int(np.ceil(72 * np.log(term) / (eps**2)))
    return R
