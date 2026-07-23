import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # 新增
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

# 定义测试函数 f2
def f2(x, y, z):
    return 1.0 / (np.cosh(3 * (x + y + z)))**2

# 参数设置（与验证代码一致）
N = 3
I = 30
M = 2 * I + 1
basis_list = [LegendreBasis(I) for _ in range(N)]
nodes, weights = gauss_legendre(M)
quad_nodes = [nodes for _ in range(N)]
quad_weights = [weights for _ in range(N)]
target_ranks = (10, 10, 10)

# 计算全系数张量
full_tensor = compute_full_coefficient_tensor(f2, basis_list, quad_nodes, quad_weights)

# 运行 TCUR 算法（Algorithm 5.1）获取中间 Tucker 表示
b = [4, 4, 4]
tau = 0.02
G1, factors1, I_sets, K, cond_factors, U_inv_norms, _ = algorithm_5_1(
    f2, basis_list, quad_nodes, quad_weights, b, tau,
    max_iters=20, min_sizes=target_ranks
)

# 对中间 Tucker 进行目标秩的重新压缩（得到最终 Tucker 表示）
S_core1, V_final1 = tcur_to_tucker(G1, factors1, target_ranks)

G2, factors2, I_sets, K, cond_factors, U_inv_norms, _ = algorithm_5_2(
    f2, basis_list, quad_nodes, quad_weights, b, tau,
    max_iters=20, min_sizes=target_ranks
)

# 对中间 Tucker 进行目标秩的重新压缩（得到最终 Tucker 表示）
S_core2, V_final2 = tcur_to_tucker(G2, factors2, target_ranks)


G3, factors3, I_sets, K_G, K_C, cond_factors, U_inv_norms= algorithm_5_4(
    f2, basis_list, quad_nodes, quad_weights, b, b, tau, tau,
    max_iters=20, min_sizes=target_ranks
)

# 对中间 Tucker 进行目标秩的重新压缩（得到最终 Tucker 表示）
S_core3, V_final3 = tcur_to_tucker(G3, factors3, target_ranks)

# 生成切片 x3 = 0 上的二维网格（100×100 点）
n_pts = 100
x1 = np.linspace(-1, 1, n_pts)
x2 = np.linspace(-1, 1, n_pts)
X1, X2 = np.meshgrid(x1, x2)
X3 = np.zeros_like(X1)  # x3 = 0

# 构造测试点 (N×3 数组)
points = np.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=1)

# 真实函数值
true_vals = f2(points[:, 0], points[:, 1], points[:, 2])

# 全超插值近似
approx_full = eval_full_hyperinterpolation(points, full_tensor, basis_list)

# Tucker 压缩近似
approx_tucker1 = eval_tucker_approximation(points, S_core1, V_final1, basis_list)
approx_tucker2 = eval_tucker_approximation(points, S_core2, V_final2, basis_list)
approx_tucker3 = eval_tucker_approximation(points, S_core3, V_final3, basis_list)

# 重塑为网格形状以便绘图
true_grid = true_vals.reshape(n_pts, n_pts)
full_grid = approx_full.reshape(n_pts, n_pts)
tucker_grid1 = approx_tucker1.reshape(n_pts, n_pts)
tucker_grid2 = approx_tucker2.reshape(n_pts, n_pts)
tucker_grid3 = approx_tucker3.reshape(n_pts, n_pts)

# 计算相对 L² 误差（在切片上）
error1 = true_grid - tucker_grid1
error2 = true_grid - tucker_grid2
error3 = true_grid - tucker_grid3
relative1_l2 = np.sqrt(np.mean(error1**2)) / np.sqrt(np.mean(true_grid**2))
relative2_l2 = np.sqrt(np.mean(error2**2)) / np.sqrt(np.mean(true_grid**2))
relative3_l2 = np.sqrt(np.mean(error3**2)) / np.sqrt(np.mean(true_grid**2))

# ---------- 绘图部分（修改点） ----------
# 计算所有网格数据的全局范围，用于共享颜色映射
all_data = [full_grid, tucker_grid1, tucker_grid2, tucker_grid3]
vmin = min(np.min(data) for data in all_data)
vmax = max(np.max(data) for data in all_data)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# 使用 constrained_layout 自动调整布局
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True,
                         constrained_layout=True)

# 左上图：全超插值
im1 = axes[0, 0].contourf(X1, X2, full_grid, levels=50, cmap='viridis', norm=norm)
axes[0, 0].set_title('Full Hyperinterpolation')
axes[0, 0].set_xlabel('$x_1$')
axes[0, 0].set_ylabel('$x_2$')

# 右上图：Tucker 压缩近似 (Algorithm 5.1)
im2 = axes[0, 1].contourf(X1, X2, tucker_grid1, levels=50, cmap='viridis', norm=norm)
axes[0, 1].set_title(f'Algorithm 5.1\n(Relative $L^2$ error = {relative1_l2:.2e})')
axes[0, 1].set_xlabel('$x_1$')
axes[0, 1].set_ylabel('$x_2$')

# 左下图：Tucker 压缩近似 (Algorithm 5.2)
im3 = axes[1, 0].contourf(X1, X2, tucker_grid2, levels=50, cmap='viridis', norm=norm)
axes[1, 0].set_title(f'Algorithm 5.2\n(Relative $L^2$ error = {relative2_l2:.2e})')
axes[1, 0].set_xlabel('$x_1$')
axes[1, 0].set_ylabel('$x_2$')

# 右下图：Tucker 压缩近似 (Algorithm 5.4)
im4 = axes[1, 1].contourf(X1, X2, tucker_grid3, levels=50, cmap='viridis', norm=norm)
axes[1, 1].set_title(f'Algorithm 5.4\n(Relative $L^2$ error = {relative3_l2:.2e})')
axes[1, 1].set_xlabel('$x_1$')
axes[1, 1].set_ylabel('$x_2$')

# 添加共享颜色条（自动放置）
cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', fraction=0.1, pad=0.02)
cbar.set_label('$f_2(x_1, x_2, 0)$')

# 保存并显示
plt.savefig('f2_slice_comparison.png', dpi=300)
plt.show()
