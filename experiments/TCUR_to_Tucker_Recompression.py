import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, svd
import sys
from pathlib import Path

# Ensure the src package is importable (adjust path if needed)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ===== Imports from the src package =====
from TCUR import LegendreBasis, gauss_legendre
from TCUR.algorithms import algorithm_5_2, tcur_to_tucker
from TCUR.coefficient import compute_full_coefficient_tensor
from TCUR.tensor_ops import mode_n_unfold, reconstruct_tucker

# ===== Helper: recompression bound (Theorem A.1) with extra info =====
def compute_recompression_bound_with_sum(full_tensor, G, factors, I_sets, target_ranks):
    
    N = full_tensor.ndim

    # TCUR approximation error
    approx_tcur = reconstruct_tucker(G, factors)
    E_tcur = np.linalg.norm(full_tensor - approx_tcur)

    # Recompress to Tucker with target ranks
    S_core, V_final = tcur_to_tucker(G, factors, target_ranks)
    approx_tucker = reconstruct_tucker(S_core, V_final)
    E_exact = np.linalg.norm(full_tensor - approx_tucker)

    # Tail energy: sum over modes of squared singular values beyond the rank
    tail_sq_sum = 0.0
    for n in range(N):
        A_n = mode_n_unfold(full_tensor, n)
        s = svd(A_n, compute_uv=False)
        R = target_ranks[n]
        if R < len(s):
            tail_sq_sum += np.sum(s[R:]**2)
    tail = np.sqrt(tail_sq_sum)

    # Sum of 2‑norms of the factor matrices
    sum_norm = sum(norm(F, 2) for F in factors)

    # Bound from Theorem A.1
    E_bound = E_tcur + sum_norm * tail

    return E_exact, E_bound, E_tcur, tail, sum_norm


# ===== Plotting function =====
def plot_recompression_errors(full_tensor, G, factors, I_sets, target_ranks_list):
    
    rank_products = [np.prod(R) for R in target_ranks_list]
    exact = []
    bound = []
    tcur = []
    tail = []
    sum_norm = []

    for R in target_ranks_list:
        e_ex, e_bd, e_tc, t, s = compute_recompression_bound_with_sum(
            full_tensor, G, factors, I_sets, R
        )
        exact.append(e_ex)
        bound.append(e_bd)
        tcur.append(e_tc)
        tail.append(t)
        sum_norm.append(s)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # 1) Exact error vs bound
    ax1.semilogy(rank_products, exact, 'o-', label='Exact error')
    ax1.semilogy(rank_products, bound, 's-', label='Bound (Thm A.1)')
    ax1.set_xlabel('Product of Tucker ranks')
    ax1.set_ylabel('Frobenius norm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Exact Error vs. Theoretical Bound')

    # 2) Additive decomposition
    truncation_term = [s * t for s, t in zip(sum_norm, tail)]
    sum_of_parts = [tc + tr for tc, tr in zip(tcur, truncation_term)]
    ax2.semilogy(rank_products, tcur, 'o-', label='TCUR residual')
    ax2.semilogy(rank_products, truncation_term, 's-', label='Truncation term')
    ax2.semilogy(rank_products, sum_of_parts, '--', label='Sum of the two parts')
    ax2.set_xlabel('Product of Tucker ranks')
    ax2.set_ylabel('Frobenius norm')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Decomposition of the Additive Bound')

    # 3) Tail energy
    ax3.semilogy(rank_products, tail, 'o-', color='green')
    ax3.set_xlabel('Product of Tucker ranks')
    ax3.set_ylabel('Tail energy')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Truncation Tail Energy')

    # 4) Tightness ratio
    ratio = [b / e if e > 0 else np.inf for b, e in zip(bound, exact)]
    ax4.plot(rank_products, ratio, 'o-', color='red')
    ax4.axhline(1, linestyle='--', color='gray', alpha=0.5)
    ax4.set_xlabel('Product of Tucker ranks')
    ax4.set_ylabel('Bound / Exact ratio')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Tightness of the Bound')

    plt.tight_layout()
    plt.savefig('tcur_to_tucker_recompression.png', dpi=300)
    plt.show()


# ===== Validation driver (with table and plots) =====
def validate_recompression_with_plot():
    """
    Runs the full validation:
      - computes full coefficient tensor for f2,
      - runs Algorithm 5.2 to get TCUR,
      - recompresses to Tucker for target ranks,
      - prints a table and generates diagnostic plots.
    """
    # Test function (same as in the paper)
    def f2(x, y, z):
        return 1.0 / (np.cosh(3 * (x + y + z))) ** 2

    N = 3
    I = 30
    M = 2 * I + 1
    basis_list = [LegendreBasis(I) for _ in range(N)]
    nodes, weights = gauss_legendre(M)
    quad_nodes = [nodes for _ in range(N)]
    quad_weights = [weights for _ in range(N)]

    print("Computing full coefficient tensor...")
    full_tensor = compute_full_coefficient_tensor(f2, basis_list, quad_nodes, quad_weights)
    print(f"Full tensor shape: {full_tensor.shape}")

    # Parameters for TCUR (Algorithm 5.2)
    b = [4, 4, 4]
    tau = 0.02
    print("Running TCUR (Algorithm 5.2)...")
    # algorithm_5_2 returns 7 items: (G, factors, I_sets, K, cond_factors, U_inv_norms, tol_history)
    # We only need the first three.
    G, factors, I_sets, *_ = algorithm_5_2(
        f2, basis_list, quad_nodes, quad_weights, b, tau,
        max_iters=20, min_sizes=None
    )
    S = len(I_sets[0])
    print(f"TCUR core size: {G.shape}, S = {S}")

    # Target ranks to test
    target_ranks_list = [(2, 2, 2), (4, 4, 4), (6, 6, 6), (8, 8, 8)]

    # Print summary table
    print("\n" + "=" * 120)
    print("TCUR-to-Tucker Recompression Performance (Validation of Theorem A.1)")
    print("=" * 120)
    print(f"{'R=(R1,R2,R3)':<16} {'E_exact':<12} {'E_bound (Thm A.1)':<18} "
          f"{'Ratio (bound/exact)':<20} {'E_TCUR':<12} {'tail':<12} {'Storage reduction':<18}")
    for R in target_ranks_list:
        E_exact, E_bound, E_tcur, tail, _ = compute_recompression_bound_with_sum(
            full_tensor, G, factors, I_sets, R
        )
        reduction = (S ** 3) / np.prod(R)
        ratio = E_bound / E_exact if E_exact > 0 else np.inf
        print(f"{str(R):<16} {E_exact:.3e}   {E_bound:.3e}          "
              f"{ratio:.2f}               {E_tcur:.3e}   {tail:.3e}   {reduction:.1f}x")

    # Generate diagnostic plots
    plot_recompression_errors(full_tensor, G, factors, I_sets, target_ranks_list)


# ===== Optional: original validate_recompression (without plots) =====
def validate_recompression():
    """Same as above but without plotting (kept for compatibility)."""
    def f2(x, y, z):
        return 1.0 / (np.cosh(3 * (x + y + z))) ** 2

    N = 3
    I = 30
    M = 2 * I + 1
    basis_list = [LegendreBasis(I) for _ in range(N)]
    nodes, weights = gauss_legendre(M)
    quad_nodes = [nodes for _ in range(N)]
    quad_weights = [weights for _ in range(N)]

    print("Computing full coefficient tensor...")
    full_tensor = compute_full_coefficient_tensor(f2, basis_list, quad_nodes, quad_weights)
    print(f"Full tensor shape: {full_tensor.shape}")

    b = [4, 4, 4]
    tau = 0.02
    print("Running TCUR (Algorithm 5.2)...")
    G, factors, I_sets, *_ = algorithm_5_2(
        f2, basis_list, quad_nodes, quad_weights, b, tau,
        max_iters=20, min_sizes=None
    )
    S = len(I_sets[0])
    print(f"TCUR core size: {G.shape}, S = {S}")

    target_ranks_list = [(2, 2, 2), (4, 4, 4), (6, 6, 6), (8, 8, 8), (10, 10, 10), (12, 12, 12)]

    print("\n" + "=" * 120)
    print("TCUR-to-Tucker Recompression Performance (Validation of Theorem A.1)")
    print("=" * 120)
    print(f"{'R=(R1,R2,R3)':<16} {'E_exact':<12} {'E_bound (Thm A.1)':<18} "
          f"{'Ratio (bound/exact)':<20} {'E_TCUR':<12} {'tail':<12} {'Storage reduction':<18}")
    for R in target_ranks_list:
        E_exact, E_bound, E_tcur, tail, _ = compute_recompression_bound_with_sum(
            full_tensor, G, factors, I_sets, R
        )
        reduction = (S ** 3) / np.prod(R)
        ratio = E_bound / E_exact if E_exact > 0 else np.inf
        print(f"{str(R):<16} {E_exact:.3e}   {E_bound:.3e}          "
              f"{ratio:.2f}               {E_tcur:.3e}   {tail:.3e}   {reduction:.1f}x")


if __name__ == "__main__":
    # Run the full validation with plots
    validate_recompression_with_plot()
    # Optionally, also run the table-only version
    validate_recompression()
