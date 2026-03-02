import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def visualize_hpr_enhanced(points, viewpoint, alpha, title):
    viewpoint = np.array(viewpoint)
    
    # 1. HPR 映射逻辑
    dists = np.linalg.norm(points - viewpoint, axis=1)
    R = alpha * np.max(dists)
    direction = (points - viewpoint) / dists[:, np.newaxis]
    reflected_points = points + 2 * (R - dists[:, np.newaxis]) * direction
    
    # 2. 计算凸包判定可见性
    augmented_points = np.vstack([reflected_points, viewpoint])
    hull = ConvexHull(augmented_points)
    
    # 获取可见点的索引
    visible_indices = [idx for idx in hull.vertices if idx < len(points)]
    is_visible = np.zeros(len(points), dtype=bool)
    is_visible[visible_indices] = True

    # 3. 绘图设置
    plt.figure(figsize=(14, 8))
    
    # --- 绘制原始点云 (Original Points) ---
    plt.scatter(points[is_visible, 0], points[is_visible, 1], 
                c='blue', s=50, label='Visible $p_k$', edgecolors='white', linewidth=0.5, zorder=5)
    plt.scatter(points[~is_visible, 0], points[~is_visible, 1], 
                c='lightcoral', s=20, alpha=0.7, label='Hidden $p_k$', zorder=4)
    
    # --- 绘制映射点云 (Reflected Points) ---
    # 可见的映射点：深蓝色/亮蓝色，突出显示
    plt.scatter(reflected_points[is_visible, 0], reflected_points[is_visible, 1], 
                c='navy', s=60, marker='o', edgecolors='cyan', label='Visible Reflected $\hat{p}_k$', zorder=3)
    # 隐藏的映射点：极淡灰色，几乎不可见但保留轮廓
    plt.scatter(reflected_points[~is_visible, 0], reflected_points[~is_visible, 1], 
                c='lightgray', s=15, alpha=0.3, label='Hidden Reflected $\hat{p}_k$', zorder=2)
    
    # --- 绘制视点 (Viewpoint c) ---
    plt.scatter(viewpoint[0], viewpoint[1], marker='*', s=400, 
                c='gold', edgecolors='black', label='Viewpoint $c$ (Observer)', zorder=6)
    
    # --- 绘制投影包络线 (Tangent Lines) ---
    # 找到凸包中连接视点的两条边界边
    view_idx = len(augmented_points) - 1
    for simplex in hull.simplices:
        if view_idx in simplex:
            # 找到 simplex 中另一个不是视点的点
            target_idx = simplex[simplex != view_idx][0]
            target_point = augmented_points[target_idx]
            plt.plot([viewpoint[0], target_point[0]], [viewpoint[1], target_point[1]], 
                     color='darkkhaki', linestyle='-', linewidth=1.2, alpha=0.8, zorder=1)

    # 图表修饰
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.axis('equal')
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # 适当放大显示区域以包含所有点
    plt.margins(0.1)
    plt.show()

# --- 生成 Q1 要求的圆周数据 ---
m = 80
theta = np.linspace(0, 2*np.pi, m, endpoint=False)
circle_points = np.column_stack([4 + np.cos(theta), np.sin(theta)])

# 运行三个核心示例
visualize_hpr_enhanced(circle_points, [0, 0], 2.0, "HPR Visibility: Example 1 - Far Origin View")
visualize_hpr_enhanced(circle_points, [4, 5], 1.8, "HPR Visibility: Example 2 - Top View")
visualize_hpr_enhanced(circle_points, [6, 2], 2.5, "HPR Visibility: Example 3 - Side Oblique View")