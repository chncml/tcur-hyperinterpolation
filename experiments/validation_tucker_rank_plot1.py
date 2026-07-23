import numpy as np
import matplotlib.pyplot as plt

# ===== Imports from the src package =====
from TCUR.basis import LegendreBasis
from TCUR.quadrature import gauss_legendre
from TCUR.coefficient import compute_full_coefficient_tensor
from TCUR.error_bounds import find_epsilon_tucker_rank, theoretical_rank_bounds

def f1(x, y, z):
    r2 = x**2 + y**2 + z**2
    return np.exp(-1.0 / (r2 + 1e-15))

def f2(x, y, z):
    return 1.0 / (np.cosh(3 * (x + y + z))) ** 2

def f3(x, y, z):
    return np.cos(2 * np.pi * x)**2 + np.cos(2 * np.pi * y)**2 + np.cos(2 * np.pi * z)**2

def plot_rank_scaling(eps=1e-2, degrees=[10, 15, 20, 25, 30, 35, 40]):

    N = 3
    obs_max_ranks = []
    theo_max_ranks = []
    max_errors = []

    for I in degrees:
        print(f"Processing I = {I} ...")
        # Build basis and quadrature
        basis_list = [LegendreBasis(I) for _ in range(N)]
        nodes, weights = gauss_legendre(2 * I + 1)
        quad_nodes = [nodes for _ in range(N)]
        quad_weights = [weights for _ in range(N)]

        # Compute full coefficient tensor for f2
        full = compute_full_coefficient_tensor(f2, basis_list, quad_nodes, quad_weights)

        # Find minimal ε-Tucker rank
        ranks, approx, _, _ = find_epsilon_tucker_rank(full, eps, verbose=False)
        err_max = np.max(np.abs(full - approx))

        # Theoretical bound from Theorem 4.1
        shape = full.shape
        R_theo = theoretical_rank_bounds(shape, eps)

        obs_max_ranks.append(max(ranks))
        theo_max_ranks.append(max(R_theo))
        max_errors.append(err_max)

        print(f"  Observed ranks = {ranks}, max = {max(ranks)}")
        print(f"  Theoretical ranks = {R_theo}, max = {max(R_theo)}")
        print(f"  Max error = {err_max:.3e}\n")

    # ---- Plotting ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Rank vs. I
    ax1.plot(degrees, obs_max_ranks, 'o-', label='Observed max rank',
             linewidth=2, markersize=8)
    ax1.semilogy(degrees, theo_max_ranks, 's--', label='Theoretical max rank',
                 linewidth=2, markersize=8)
    ax1.set_xlabel('Polynomial degree $I$', fontsize=12)
    ax1.set_ylabel('Tucker rank', fontsize=12)
    ax1.set_title(f'Rank scaling with $I$ ($\\varepsilon={eps:.0e}$)', fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: Error vs. I
    ax2.semilogy(degrees, max_errors, 'D-', color='red',
                 label='Max reconstruction error', linewidth=2)
    ax2.axhline(y=eps, color='k', linestyle=':', label=f'Target $\\varepsilon={eps:.0e}$')
    ax2.set_xlabel('Polynomial degree $I$', fontsize=12)
    ax2.set_ylabel('Max absolute error', fontsize=12)
    ax2.set_title('Error verification', fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('rank_scaling_f2.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Run the plot with eps=1e-2 and degrees from 10 to 40
    plot_rank_scaling(eps=1e-2, degrees=[10, 15, 20, 25, 30, 35, 40])
