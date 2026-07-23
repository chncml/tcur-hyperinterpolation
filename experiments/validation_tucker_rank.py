import numpy as np
import sys
from pathlib import Path

# Ensure the src package is importable (adjust path if needed)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ===== Imports from the src package =====
from TCUR.basis import LegendreBasis
from TCUR.quadrature import gauss_legendre
from TCUR.tensor_ops import mode_n_unfold, st_hosvd, reconstruct_tucker
from TCUR.coefficient import compute_full_coefficient_tensor
from TCUR.error_bounds import find_epsilon_tucker_rank, theoretical_rank_bounds
from TCUR.test_functions import f1, f2, f3

# ===== Additional test function for higher dimensions =====
def f3_nd(*args):
    """Additive version of f3 for arbitrary dimension."""
    return sum(np.cos(2 * np.pi * arg) ** 2 for arg in args)


# ===== Validation experiments =====
def validate_rank_scaling():

    N = 3
    eps = 1e-2
    degrees = [10, 15, 20, 25, 30, 35, 40]

    print("=" * 70)
    print(f"Validation of rank scaling with polynomial degree (eps = {eps:.0e})")
    print("=" * 70)
    print(f"{'I':<6} {'Observed R':<15} {'Theoretical R':<15} {'Max error':<12}")
    print("-" * 55)

    for I in degrees:
        basis_list = [LegendreBasis(I) for _ in range(N)]
        nodes, weights = gauss_legendre(2 * I + 1)
        quad_nodes = [nodes for _ in range(N)]
        quad_weights = [weights for _ in range(N)]

        full = compute_full_coefficient_tensor(f3, basis_list, quad_nodes, quad_weights)
        ranks, approx, _, _ = find_epsilon_tucker_rank(full, eps, verbose=False)
        err_max = np.max(np.abs(full - approx))

        shape = full.shape
        R_theo = theoretical_rank_bounds(shape, eps)
        print(f"{I:<6} {str(ranks):<15} {str(R_theo):<15} {err_max:.3e}")


def validate_rank_vs_eps():
    
    N = 3
    I = 20
    eps_list = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]

    basis_list = [LegendreBasis(I) for _ in range(N)]
    nodes, weights = gauss_legendre(2 * I + 1)
    quad_nodes = [nodes for _ in range(N)]
    quad_weights = [weights for _ in range(N)]

    full = compute_full_coefficient_tensor(f3, basis_list, quad_nodes, quad_weights)
    shape = full.shape

    print("\n" + "=" * 70)
    print("Validation of rank vs tolerance eps (I=20, f3)")
    print("=" * 70)
    print(f"{'eps':<10} {'Observed R':<15} {'Theoretical R':<15} {'Max error':<12}")
    print("-" * 60)

    for eps in eps_list:
        ranks, approx, _, _ = find_epsilon_tucker_rank(full, eps, verbose=False)
        err_max = np.max(np.abs(full - approx))
        R_theo = theoretical_rank_bounds(shape, eps)
        print(f"{eps:.0e}   {str(ranks):<15} {str(R_theo):<15} {err_max:.3e}")


def validate_dimension_independence():
    
    eps = 1e-2
    I = 15
    N_list = [3, 4, 5]

    print("\n" + "=" * 70)
    print(f"Dimension independence test (eps={eps:.0e}, I={I})")
    print("=" * 70)
    print(f"{'N':<5} {'Observed R':<20} {'Theoretical R':<20} {'Max error':<12}")
    print("-" * 65)

    for N in N_list:
        basis_list = [LegendreBasis(I) for _ in range(N)]
        M = 2 * I + 1
        nodes, weights = gauss_legendre(M)
        quad_nodes = [nodes for _ in range(N)]
        quad_weights = [weights for _ in range(N)]

        # Define f_func for this N using f3_nd
        def f_func(*args):
            return f3_nd(*args)

        full = compute_full_coefficient_tensor(f_func, basis_list, quad_nodes, quad_weights)
        ranks, approx, _, _ = find_epsilon_tucker_rank(full, eps, verbose=False)
        err_max = np.max(np.abs(full - approx))
        shape = full.shape
        R_theo = theoretical_rank_bounds(shape, eps)
        print(f"{N:<5} {str(ranks):<20} {str(R_theo):<20} {err_max:.3e}")


def validate_function_dependence():
    
    N = 3
    I = 25
    eps_list = [1e-1, 5e-2, 1e-2]
    functions = [f1, f2, f3]
    func_names = ['f1 (peak)', 'f2 (smooth)', 'f3 (oscillatory)']

    print("\n" + "=" * 70)
    print("Function dependence of Tucker ranks (I=25)")
    print("=" * 70)
    print(f"{'Function':<15} {'eps':<10} {'Observed R':<15} {'Theoretical R':<15} {'Max error':<12}")
    print("-" * 75)

    for func, name in zip(functions, func_names):
        basis_list = [LegendreBasis(I) for _ in range(N)]
        nodes, weights = gauss_legendre(2 * I + 1)
        quad_nodes = [nodes for _ in range(N)]
        quad_weights = [weights for _ in range(N)]

        full = compute_full_coefficient_tensor(func, basis_list, quad_nodes, quad_weights)
        shape = full.shape

        for eps in eps_list:
            ranks, approx, _, _ = find_epsilon_tucker_rank(full, eps, verbose=False)
            err_max = np.max(np.abs(full - approx))
            R_theo = theoretical_rank_bounds(shape, eps)
            print(f"{name:<15} {eps:.0e}   {str(ranks):<15} {str(R_theo):<15} {err_max:.3e}")
        print("-" * 75)


# ===== Main execution =====
if __name__ == "__main__":
    validate_rank_scaling()
    validate_rank_vs_eps()
    validate_dimension_independence()
    validate_function_dependence()
