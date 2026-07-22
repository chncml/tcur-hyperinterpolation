import numpy as np
import time

import sys
from pathlib import Path

# Make the sibling src package importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TCUR import (
    LegendreBasis, gauss_legendre,
    algorithm_5_1, algorithm_5_2, algorithm_5_4,
    compute_full_coefficient_tensor, reconstruct_tucker,
    reset_inner_counter, get_inner_counter
)

def f2(x, y, z):
    return 1.0 / (np.cosh(3*(x+y+z)))**2

def validate_greedy_tcur():
    N = 3
    I = 30
    M = 2*I + 1
    basis_list = [LegendreBasis(I) for _ in range(N)]
    nodes, weights = gauss_legendre(M)
    quad_nodes = [nodes for _ in range(N)]
    quad_weights = [weights for _ in range(N)]

    print("Computing full tensor...")
    reset_inner_counter()
    full_tensor = compute_full_coefficient_tensor(f2, basis_list, quad_nodes, quad_weights)
    full_count = get_inner_counter()
    print(f"Full tensor shape: {full_tensor.shape}, inner products: {full_count}")

    block_sizes = [2,3,4,5]
    taus = [0.05,0.02,0.01]

    # Algorithm 5.1
    print("\n" + "="*100)
    print("Algorithm 5.1 (factor-based)")
    print("="*100)
    print(f"{'b':<4} {'tau':<6} {'K':<4} {'S':<6} {'Inner prods':<14} {'||error||_F':<12} {'max ||C_n U_n^†||_2':<22} {'max ||U_n^†||_2':<15}")
    for b in block_sizes:
        for tau in taus:
            reset_inner_counter()
            G, factors, I_sets, K, cond_factors, U_inv_norms = algorithm_5_1(
                f2, basis_list, quad_nodes, quad_weights, [b]*N, tau, max_iters=20, min_sizes=None
            )
            inner_count = get_inner_counter()
            S = len(I_sets[0])
            approx = reconstruct_tucker(G, factors)
            err_F = np.linalg.norm(full_tensor - approx)
            max_cond = max(cond_factors)
            max_Uinv = max(U_inv_norms)
            print(f"{b:<4} {tau:.2f}  {K:<4} {S:<6} {inner_count:<14} {err_F:.3e}  {max_cond:.2f}  {max_Uinv:.2f}")

    # Algorithm 5.2
    print("\n" + "="*100)
    print("Algorithm 5.2 (Core-based)")
    print("="*100)
    print(f"{'b':<4} {'tau':<6} {'K':<4} {'S':<6} {'Inner prods':<14} {'||error||_F':<12} {'max ||C_n U_n^†||_2':<22} {'max ||U_n^†||_2':<15}")
    for b in block_sizes:
        for tau in taus:
            reset_inner_counter()
            G, factors, I_sets, K, cond_factors, U_inv_norms = algorithm_5_2(
                f2, basis_list, quad_nodes, quad_weights, [b]*N, tau, max_iters=20, min_sizes=None
            )
            inner_count = get_inner_counter()
            S = len(I_sets[0])
            approx = reconstruct_tucker(G, factors)
            err_F = np.linalg.norm(full_tensor - approx)
            max_cond = max(cond_factors)
            max_Uinv = max(U_inv_norms)
            print(f"{b:<4} {tau:.2f}  {K:<4} {S:<6} {inner_count:<14} {err_F:.3e}  {max_cond:.2f}  {max_Uinv:.2f}")

    # Algorithm 5.4
    print("\n" + "="*100)
    print("Algorithm 5.4 (fiber‑based)")
    print("="*100)
    print(f"{'b':<4} {'tau':<6} {'K':<4} {'S':<6} {'Inner prods':<14} {'||error||_F':<12} {'max ||C_n U_n^†||_2':<22} {'max ||U_n^†||_2':<15}")
    for b in block_sizes:
        for tau in taus:
            reset_inner_counter()
            G, factors, I_sets, K_G, K_C, cond_factors, U_inv_norms = algorithm_5_4(
                f2, basis_list, quad_nodes, quad_weights, [b]*N, [b]*N, tau, tau, max_iters=20
            )
            inner_count = get_inner_counter()
            S = len(I_sets[0])
            approx = reconstruct_tucker(G, factors)
            err_F = np.linalg.norm(full_tensor - approx)
            max_cond = max(cond_factors)
            max_Uinv = max(U_inv_norms)
            print(f"{b:<4} {tau:.2f}  {K_G:<5} {K_C:<5} {S:<6} {inner_count:<14} {err_F:.3e}   {max_cond:.2f}              {max_Uinv:.2f}")
            # results_54.append((b, tau, K_G, K_C, S, inner_count, err_F, max_cond, max_Uinv))
if __name__ == "__main__":
    validate_greedy_tcur()
