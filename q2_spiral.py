import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def plot_q2_spiral_analysis(ax, points, viewpoint, alpha, title):
    viewpoint = np.array(viewpoint)
    
    # 1. HPR 映射 (球面反转映射)
    dists = np.linalg.norm(points - viewpoint, axis=1)
    R = alpha * np.max(dists)
    direction = (points - viewpoint) / dists[:, np.newaxis]
    reflected_points = points + 2 * (R - dists[:, np.newaxis]) * direction
    
    # 2. 构建包含外部视点的凸包
    augmented_points = np.vstack([reflected_points, viewpoint])
    hull = ConvexHull(augmented_points)
    
    # 判定可见性
    visible_indices = [idx for idx in hull.vertices if idx < len(points)]
    is_visible = np.zeros(len(points), dtype=bool)
    is_visible[visible_indices] = True

    # 3. 绘图：模仿作业 PDF 的“原物体+映射物体”效果
    # --- 绘制原始螺旋线 ---
    # 可见点用蓝色大点，隐藏点用蓝色小点
    ax.scatter(points[is_visible, 0], points[is_visible, 1], 
                c='blue', s=25, zorder=5, label='Visible $p_k$')
    ax.scatter(points[~is_visible, 0], points[~is_visible, 1], 
                c='blue', s=2, alpha=0.3, zorder=4)
    
    # --- 绘制映射后的螺旋线 ---
    # 可见映射点用红色大点，隐藏映射点用红色小点
    ax.scatter(reflected_points[is_visible, 0], reflected_points[is_visible, 1], 
                c='red', s=25, zorder=3, label='Reflected $\hat{p}_k$')
    ax.scatter(reflected_points[~is_visible, 0], reflected_points[~is_visible, 1], 
                c='red', s=2, alpha=0.2, zorder=2)
    
    # --- 绘制外部视点 ---
    ax.scatter(viewpoint[0], viewpoint[1], marker='x', s=100, 
                c='black', linewidth=1.5, zorder=6, label='Viewpoint $c$')
    
    # --- 绘制黄色包络切线 (核心视觉效果) ---
    view_idx = len(augmented_points) - 1
    for simplex in hull.simplices:
        if view_idx in simplex:
            target_idx = simplex[simplex != view_idx][0]
            target_point = augmented_points[target_idx]
            ax.plot([viewpoint[0], target_point[0]], [viewpoint[1], target_point[1]], 
                     color='yellow', lw=1, alpha=0.6, zorder=1)

    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.4)

# --- 生成螺旋线数据 ---
m = 300
t = np.linspace(0, 2.5 * np.pi, m)
# 调整位置，确保视点 [0,0] 在物体外部
spiral_pts = np.column_stack([(t/2 + 3) * np.cos(t), (t/2 + 3) * np.sin(t)])

# 设定 alpha 参数
alphas = [1.1, 10, 100, 1000]
view_c = [0, 20]

# 创建 2x2 画布
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, a in enumerate(alphas):
    plot_q2_spiral_analysis(axes[i], spiral_pts, view_c, a, f"Q2: Spiral (alpha={a})")

plt.tight_layout()
plt.savefig('q2_spiral_mapping.png')
plt.show()