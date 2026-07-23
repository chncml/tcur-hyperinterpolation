import numpy as np
import time
from pathlib import Path
import sys

# Make sure the src package is importable (adjust path if needed)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TCUR import (
    LegendreBasis, gauss_legendre,
    algorithm_5_1, algorithm_5_2, algorithm_5_4,
    compute_full_coefficient_tensor, reconstruct_tucker,
    reset_inner_counter, get_inner_counter,
    eval_full_hyperinterpolation, eval_tcur_approximation,
    eval_tucker_approximation, tcur_to_tucker,
    compute_tcur_error_bound, theoretical_rank_bounds,
    basis_degrees
)

# ----------------------------------------------------------------------
# Test functions (same as in the paper)
# ----------------------------------------------------------------------
def f1(x, y, z):
    r2 = x**2 + y**2 + z**2
    return np.exp(-1.0 / (r2 + 1e-15))

def f2(x, y, z):
    return 1.0 / (np.cosh(3*(x+y+z)))**2

def f3(x, y, z):
    return np.cos(2*np.pi*x)**2 + np.cos(2*np.pi*y)**2 + np.cos(2*np.pi*z)**2

# ----------------------------------------------------------------------
# Helper: L2 error on a set of points (Monte Carlo, volume 8)
# ----------------------------------------------------------------------
def compute_l2_error(points, true_vals, approx_vals):
    volume = 8.0
    mse = np.mean((true_vals - approx_vals)**2)
    return np.sqrt(volume * mse)

# ----------------------------------------------------------------------
# End‑to‑end error‑chain validation (including composite bound)
# ----------------------------------------------------------------------
def run_end_to_end_validation():
    """Reproduces Table 6.10 style output for f1, f2, f3."""
    funcs = [f1, f2, f3]
    func_names = ['f1 (peak)', 'f2 (smooth)', 'f3 (oscillatory)']

    N = 3
    I = 30
    M = 2*I + 1
    basis_list = [LegendreBasis(I) for _ in range(N)]
    nodes, weights = gauss_legendre(M)
    quad_nodes = [nodes for _ in range(N)]
    quad_weights = [weights for _ in range(N)]

    # Fixed parameters (chosen to give a reasonable approximation)
    b = [4, 4, 4]
    tau = 0.02
    target_ranks = (10, 10, 10)

    # Monte Carlo test points
    num_test = 5000
    np.random.seed(0)
    test_points = np.random.uniform(-1, 1, (num_test, N))

    print("\n" + "="*120)
    print("End‑to‑End Error Chain Validation (cf. Table 6.10)")
    print("="*120)
    header = (
        f"{'Function':<15} {'||L f - f||_L2':<18} "
        f"{'||L̃ f - L f||_L2':<22} {'||L̃ f - f||_L2':<18} "
        f"{'Bound (Remark A.2)':<20} {'Ratio':<10}"
    )
    print(header)
    print("-"*120)

    results = []
    for func, name in zip(funcs, func_names):
        print(f"\nProcessing {name} ...")
        # Full tensor
        full_tensor = compute_full_coefficient_tensor(func, basis_list, quad_nodes, quad_weights)
        # True values
        true_vals = func(test_points[:,0], test_points[:,1], test_points[:,2])

        # 1. Discretisation error: full hyperinterpolation
        approx_full = eval_full_hyperinterpolation(test_points, full_tensor, basis_list)
        err_disc = compute_l2_error(test_points, true_vals, approx_full)
        print(f"  Discretisation error: {err_disc:.3e}")

        # 2. TCUR approximation (using Algorithm 5.1)
        reset_inner_counter()
        G, factors, I_sets, K, cond_factors, U_inv_norms, _ = algorithm_5_1(
            func, basis_list, quad_nodes, quad_weights, b, tau,
            max_iters=20, min_sizes=target_ranks
        )
        S = len(I_sets[0])
        print(f"  TCUR core size: {G.shape}, S={S}")

        # TCUR evaluation
        approx_tcur = eval_tcur_approximation(test_points, G, factors, basis_list)
        err_tcur_vs_full = compute_l2_error(test_points, approx_full, approx_tcur)
        err_total_tcur = compute_l2_error(test_points, true_vals, approx_tcur)

        # 3. Recompression to Tucker (Algorithm A.1)
        S_core, V_final = tcur_to_tucker(G, factors, target_ranks)
        approx_tucker = eval_tucker_approximation(test_points, S_core, V_final, basis_list)
        err_recomp_vs_full = compute_l2_error(test_points, approx_full, approx_tucker)
        err_total_recomp = compute_l2_error(test_points, true_vals, approx_tucker)

        print(f"  TCUR vs full L2: {err_tcur_vs_full:.3e}")
        print(f"  Recompressed vs full L2: {err_recomp_vs_full:.3e}")
        print(f"  Total L2 (recompressed vs true): {err_total_recomp:.3e}")

        # 4. Composite bound (Remark A.2)
        # Use the dedicated function from error_bounds.py
        E_exact, E_th, E_tight = compute_tcur_error_bound(
            func, full_tensor, G, factors, basis_list,
            quad_nodes, quad_weights, I_sets, target_ranks
        )
        # E_th is the Frobenius bound from Theorem 4.4; convert to L2
        # by multiplying by sqrt(prod(I_n+1)) – see Remark A.2
        volume_factor = np.sqrt(np.prod(full_tensor.shape))
        bound_tcur_l2 = E_th * volume_factor

        # Tucker truncation term from Theorem A.1 (needs to be added)
        # We have the SVD of the full tensor from inside compute_tcur_error_bound,
        # but we can recompute tail sum easily.
        tail_sq_sum = 0.0
        from src.tensor_ops import mode_n_unfold
        for n in range(N):
            A_n = mode_n_unfold(full_tensor, n)
            s = np.linalg.svd(A_n, compute_uv=False)
            R = target_ranks[n]
            if R < len(s):
                tail_sq_sum += np.sum(s[R:]**2)
        tail = np.sqrt(tail_sq_sum)
        sum_norm = sum(np.linalg.norm(F, 2) for F in factors)
        tucker_term_l2 = sum_norm * tail * volume_factor

        composite_bound = err_disc + bound_tcur_l2 + tucker_term_l2
        ratio = composite_bound / err_total_recomp if err_total_recomp > 0 else np.inf

        print(f"  Composite bound: {composite_bound:.3e}, ratio: {ratio:.2f}")
        results.append((name, err_disc, err_tcur_vs_full, err_total_recomp,
                        composite_bound, ratio))

    # Final table
    print("\n" + "="*120)
    print("Summary Table (comparable to Table 6.10)")
    print("="*120)
    print(header)
    print("-"*120)
    for name, e_disc, e_tcur, e_total, bound, ratio in results:
        print(f"{name:<15} {e_disc:.3e}        {e_tcur:.3e}            "
              f"{e_total:.3e}        {bound:.3e}          {ratio:.2f}")
    print("="*120)

    # Storage reduction
    full_storage = np.prod(full_tensor.shape)
    final_storage = np.prod(target_ranks) + sum((I+1)*R for I,R in zip([I]*N, target_ranks))
    reduction = full_storage / final_storage
    print(f"\nStorage reduction (full to Tucker with R={target_ranks}): {reduction:.2f}x")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":

    # Detailed end‑to‑end validation with error bounds
    run_end_to_end_validation()
