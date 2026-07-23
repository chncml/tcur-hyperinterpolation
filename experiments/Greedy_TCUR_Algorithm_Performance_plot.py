import matplotlib.pyplot as plt
import numpy as np
# import time

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

# ----------------------------------------------------------------------
# 主程序：运行并绘图
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 定义测试函数 f2
    def f2(x, y, z):
        return 1.0 / (np.cosh(3*(x+y+z)))**2

    # 参数设置
    N = 3
    I = 30
    M = 2*I + 1
    basis_list = [LegendreBasis(I) for _ in range(N)]
    nodes, weights = gauss_legendre(M)
    quad_nodes = [nodes for _ in range(N)]
    quad_weights = [weights for _ in range(N)]

    # 运行 Algorithm 5.1 (b=4, tau=0.02)
    b = 2
    tau = 0.01
    max_iters = 30   # 足够大以观察收敛

    reset_inner_counter()
    G, factors, I_sets, K, cond_factors, U_inv_norms, tol_history = algorithm_5_2(
        f2, basis_list, quad_nodes, quad_weights, [b]*N, tau, max_iters
    )

    # 打印停止时的迭代次数和最终 tol
    print(f"Stopped at K = {K}, final tol = {tol_history[-1]:.4e}, threshold tau = {tau}")

    # 绘图
    iterations = np.arange(1, len(tol_history)+1)
    plt.figure(figsize=(8, 5))
    plt.semilogy(iterations, tol_history, 'bo-', linewidth=2, markersize=6)
    plt.axhline(y=tau, color='r', linestyle='--', label=f'Threshold $\\tau = {tau}$')
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel('$\\mathrm{tol}$', fontsize=14)
    plt.title('Convergence of Algorithm 5.2 ($f_2(x_1,x_2,x_3)$, $b=2$, $\\tau=0.01$)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('tol_convergence_alg52.png', dpi=300)
    plt.show()
