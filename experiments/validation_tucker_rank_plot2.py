import numpy as np
import matplotlib.pyplot as plt

# ===== Imports from the src package =====
from TCUR.basis import LegendreBasis
from TCUR.quadrature import gauss_legendre
from TCUR.coefficient import compute_full_coefficient_tensor
from TCUR.error_bounds import find_epsilon_tucker_rank

def f1(x, y, z):
    r2 = x**2 + y**2 + z**2
    return np.exp(-1.0 / (r2 + 1e-15))

def f2(x, y, z):
    return 1.0 / (np.cosh(3 * (x + y + z))) ** 2

def f3(x, y, z):
    return np.cos(2 * np.pi * x)**2 + np.cos(2 * np.pi * y)**2 + np.cos(2 * np.pi * z)**2
  
def plot_function_comparison(I=30, eps_list=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3]):
    
    N = 3
    functions = [f1, f2, f3]
    func_names = ['$f_1$ (peak)', '$f_2$ (smooth)', '$f_3$ (oscillatory)']
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', 'D']

    # Pre‑compute basis and quadrature once (shared by all functions)
    basis_list = [LegendreBasis(I) for _ in range(N)]
    nodes, weights = gauss_legendre(2 * I + 1)
    quad_nodes = [nodes for _ in range(N)]
    quad_weights = [weights for _ in range(N)]

    # Storage for ranks and errors
    rank_results = {name: [] for name in func_names}
    error_results = {name: [] for name in func_names}

    for func, name in zip(functions, func_names):
        print(f"Processing {name} ...")
        # Compute full coefficient tensor (independent of ε)
        full = compute_full_coefficient_tensor(func, basis_list, quad_nodes, quad_weights)

        for eps in eps_list:
            ranks, approx, _, _ = find_epsilon_tucker_rank(full, eps, verbose=False)
            err_max = np.max(np.abs(full - approx))
            max_rank = max(ranks)
            rank_results[name].append(max_rank)
            error_results[name].append(err_max)
            print(f"  eps={eps:.0e}: ranks={ranks}, max={max_rank}, err={err_max:.3e}")

    # ---- Plotting ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Maximum rank vs ε
    for idx, name in enumerate(func_names):
        ax1.semilogx(eps_list, rank_results[name],
                     marker=markers[idx], color=colors[idx],
                     linewidth=2, markersize=8, label=name)
    ax1.set_xlabel('Tolerance $\\varepsilon$', fontsize=14)
    ax1.set_ylabel('Maximum Tucker rank', fontsize=14)
    ax1.set_title(f'Rank comparison for different functions ($I={I}$)', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=12)

    # Subplot 2: Reconstruction error vs ε
    for idx, name in enumerate(func_names):
        ax2.loglog(eps_list, error_results[name],
                   marker=markers[idx], color=colors[idx],
                   linewidth=2, markersize=8, label=name)
    ax2.loglog(eps_list, eps_list, 'k--', label='target $\\varepsilon$', alpha=0.5)
    ax2.set_xlabel('Tolerance $\\varepsilon$', fontsize=14)
    ax2.set_ylabel('Max reconstruction error', fontsize=14)
    ax2.set_title('Error verification', fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('rank_epsilon_f2.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Run the comparison plot with the given ε list
    plot_function_comparison(I=30, eps_list=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3])
