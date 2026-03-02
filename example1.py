import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def plot_hpr_subplot(ax, points, viewpoint, alpha, title):
    viewpoint = np.array(viewpoint)
    
    # 1. HPR 映射逻辑 (球面反转映射)
    dists = np.linalg.norm(points - viewpoint, axis=1)
    R = alpha * np.max(dists)
    direction = (points - viewpoint) / dists[:, np.newaxis]
    reflected_points = points + 2 * (R - dists[:, np.newaxis]) * direction
    
    # 2. 构建包含视点的增广点集并计算凸包
    augmented_points = np.vstack([reflected_points, viewpoint])
    hull = ConvexHull(augmented_points)
    
    # 判定可见性
    visible_indices = [idx for idx in hull.vertices if idx < len(points)]
    is_visible = np.zeros(len(points), dtype=bool)
    is_visible[visible_indices] = True

    # 3. 增强绘图效果
    # --- 绘制原始点云 ---
    ax.scatter(points[is_visible, 0], points[is_visible, 1], 
                c='blue', s=30, label='Visible $p_k$', edgecolors='white', linewidth=0.3, zorder=5)
    ax.scatter(points[~is_visible, 0], points[~is_visible, 1], 
                c='lightcoral', s=15, alpha=0.6, label='Hidden $p_k$', zorder=4)
    
    # --- 绘制映射点云 ---
    ax.scatter(reflected_points[is_visible, 0], reflected_points[is_visible, 1], 
                c='navy', s=40, marker='o', edgecolors='cyan', label='Visible $\hat{p}_k$', zorder=3)
    ax.scatter(reflected_points[~is_visible, 0], reflected_points[~is_visible, 1], 
                c='lightgray', s=10, alpha=0.2, label='Hidden $\hat{p}_k$', zorder=2)
    
    # --- 绘制视点 ---
    ax.scatter(viewpoint[0], viewpoint[1], marker='*', s=200, 
                c='gold', edgecolors='black', label='Viewpoint $c$', zorder=6)
    
    # --- 绘制投影包络切线 ---
    view_idx = len(augmented_points) - 1
    for simplex in hull.simplices:
        if view_idx in simplex:
            target_idx = simplex[simplex != view_idx][0]
            target_point = augmented_points[target_idx]
            ax.plot([viewpoint[0], target_point[0]], [viewpoint[1], target_point[1]], 
                     color='khaki', lw=1, alpha=0.7, zorder=1)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')

# 生成数据
m = 80
theta = np.linspace(0, 2*np.pi, m, endpoint=False)
circle_points = np.column_stack([4 + np.cos(theta), np.sin(theta)])

# 设定三个示例的参数
examples = [
    {"c": [0, 0], "alpha": 2.0, "title": "Example 1: Far View\n($c=[0,0], \\alpha=2.0$)"},
    {"c": [2.5, 0.5], "alpha": 2.0, "title": "Example 2: Close View\n($c=[2.5,0.5], \\alpha=2.0$)"},
    {"c": [4, 3], "alpha": 1.1, "title": "Example 3: Top View\n($c=[4,3], \\alpha=1.1$)"}
]

# 创建 1x3 的并排画布
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for i, ex in enumerate(examples):
    plot_hpr_subplot(axes[i], circle_points, ex["c"], ex["alpha"], ex["title"])

# 在底部添加统一图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.02))

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('q1_combined_enhanced.png') # 保存为合并后的图片
plt.show()