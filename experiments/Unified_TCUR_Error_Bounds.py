import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, norm
import sys
from pathlib import Path

# Ensure the src package is importable (adjust path if needed)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ===== Imports from the src package =====
from TCUR.basis import LegendreBasis
from TCUR.quadrature import gauss_legendre
from TCUR.tensor_ops import mode_n_unfold, reconstruct_tucker, st_hosvd
from TCUR.coefficient import compute_full_coefficient_tensor
from TCUR.utils import basis_degrees
from TCUR.algorithms import algorithm_5_2, algorithm_5_4
from TCUR.error_bounds import compute_tcur_error_bound

def f1(x, y, z):
    r2 = x**2 + y**2 + z**2
    return np.exp(-1.0 / (r2 + 1e-15))

def f2(x, y, z):
    return 1.0 / (np.cosh(3 * (x + y + z))) ** 2

def f3(x, y, z):
    return np.cos(2 * np.pi * x)**2 + np.cos(2 * np.pi * y)**2 + np.cos(2 * np.pi * z)**2

# ----------------------------------------------------------------------
# Validation drivers
# ----------------------------------------------------------------------

def validate_error_bounds():

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

    R_rank = (10, 10, 10)          # target Tucker rank for the bounds
    block_sizes = [2, 3, 4, 5]
    taus = [0.05, 0.02, 0.01]

    print("\n" + "=" * 120)
    print("Validation of TCUR Error Bounds (Theorem 4.4 and Remark 4.9)")
    print("=" * 120)
    print(f"{'b':<4} {'tau':<6} {'S':<6} {'E_exact':<12} "
          f"{'E_th (Thm 4.4)':<16} {'Ratio (E_th/E_exact)':<20} "
          f"{'E_tight (Remark 4.9)':<20} {'Ratio (E_tight/E_exact)':<20}")
    for b in block_sizes:
        for tau in taus:
            # Algorithm 5.4 returns 7 items: G, factors, I_sets, K_G, K_C, cond_factors, U_inv_norms
            G, factors, I_sets, _, _, _, _ = algorithm_5_4(
                f2, basis_list, quad_nodes, quad_weights,
                [b] * N, [b] * N, tau, tau, max_iters=20
            )
            S = len(I_sets[0])   # all modes equal
            E_exact, E_th, E_tight = compute_tcur_error_bound(
                f2, full_tensor, G, factors, basis_list,
                quad_nodes, quad_weights, I_sets, R_rank
            )
            ratio1 = E_th / E_exact if E_exact > 0 else np.inf
            ratio2 = E_tight / E_exact if E_exact > 0 else np.inf
            print(f"{b:<4} {tau:.2f}  {S:<6} {E_exact:.3e}   "
                  f"{E_th:.3e}          {ratio1:.2f}               "
                  f"{E_tight:.3e}          {ratio2:.2f}")


def validate_function_dependence():

    N = 3
    I = 30
    M = 2 * I + 1
    basis_list = [LegendreBasis(I) for _ in range(N)]
    nodes, weights = gauss_legendre(M)
    quad_nodes = [nodes for _ in range(N)]
    quad_weights = [weights for _ in range(N)]

    # Compute full tensors for all functions
    full_tensors = {}
    for func, name in zip([f1, f2, f3], ['f1', 'f2', 'f3']):
        print(f"Computing full tensor for {name}...")
        full_tensors[name] = compute_full_coefficient_tensor(func, basis_list, quad_nodes, quad_weights)
        print(f"Shape: {full_tensors[name].shape}")

    R_rank = (10, 10, 10)
    b = 4
    tau = 0.05

    print("\n" + "=" * 120)
    print("Function Dependence of TCUR Error Bounds")
    print("=" * 120)
    print(f"{'Function':<6} {'S':<6} {'E_exact':<12} "
          f"{'E_th (Thm 4.4)':<16} {'Ratio (E_th/E_exact)':<20} "
          f"{'E_tight (Remark 4.9)':<20} {'Ratio (E_tight/E_exact)':<20}")
    for func, name in zip([f1, f2, f3], ['f1', 'f2', 'f3']):
        # Algorithm 5.2 returns 7 items (G, factors, I_sets, K, cond_factors, U_inv_norms, tol_history)
        G, factors, I_sets, _, _, _, _ = algorithm_5_2(
            func, basis_list, quad_nodes, quad_weights,
            [b] * N, tau, max_iters=20, min_sizes=None
        )
        S = len(I_sets[0])
        full_tensor = full_tensors[name]
        E_exact, E_th, E_tight = compute_tcur_error_bound(
            func, full_tensor, G, factors, basis_list,
            quad_nodes, quad_weights, I_sets, R_rank
        )
        ratio1 = E_th / E_exact if E_exact > 0 else np.inf
        ratio2 = E_tight / E_exact if E_exact > 0 else np.inf
        print(f"{name:<6} {S:<6} {E_exact:.3e}   "
              f"{E_th:.3e}          {ratio1:.2f}               "
              f"{E_tight:.3e}          {ratio2:.2f}")


def plot_singular_value_spectra():

    N = 3
    I = 30
    M = 2 * I + 1
    basis_list = [LegendreBasis(I) for _ in range(N)]
    nodes, weights = gauss_legendre(M)
    quad_nodes = [nodes for _ in range(N)]
    quad_weights = [weights for _ in range(N)]

    funcs = [f1, f2, f3]
    labels = [r'$f_1$', r'$f_2$', r'$f_3$']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for mode in range(3):
        ax = axes[mode]
        for func, label in zip(funcs, labels):
            full_tensor = compute_full_coefficient_tensor(func, basis_list, quad_nodes, quad_weights)
            unfold = mode_n_unfold(full_tensor, mode)
            s = svd(unfold, compute_uv=False)
            s_norm = s / s[0]
            ax.semilogy(range(1, len(s_norm) + 1), s_norm,
                        marker='o', markersize=3, label=label)
        ax.set_title(f'Mode‑{mode+1} unfolding')
        ax.set_xlabel('Index $i$')
        if mode == 0:
            ax.set_ylabel('Normalized singular value $\\sigma_i / \\sigma_1$')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.savefig('Normalized_singular_value_spectra.png', dpi=300)
    plt.show()


# ===== Main execution =====
if __name__ == "__main__":
    validate_error_bounds()
    validate_function_dependence()
    plot_singular_value_spectra()
