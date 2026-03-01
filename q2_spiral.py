import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def visibility_hpr(points, viewpoint, alpha):
    """
    实现 Hidden Point Removal 算子
    """
    # 1. 平移点集，使视点位于原点
    p_adj = points - viewpoint
    norms = np.linalg.norm(p_adj, axis=1, keepdims=True)
    
    # 2. 计算反射半径 R
    R = alpha * np.max(norms)
    
    # 3. 球面反射映射 (Spherical Inversion)
    # 映射公式: P' = P + 2*(R - ||P||) * (P / ||P||)
    p_reflected = p_adj + 2 * (R - norms) * (p_adj / norms)
    
    # 4. 将视点(原点)加入点集并计算凸包
    points_with_origin = np.vstack([p_reflected, [0, 0]])
    hull = ConvexHull(points_with_origin)
    
    # 5. 可见性判定：在凸包顶点上的点（排除最后加入的原点）是可见的
    visible_indices = [idx for idx in hull.vertices if idx < len(points)]
    
    return visible_indices, p_reflected

# --- Q2: 螺旋线数据生成 ---
def generate_spiral(n_points=500):
    theta = np.linspace(0, 4 * np.pi, n_points)
    r = 0.5 * theta
    x = r * np.cos(theta) + 5  # 平移到中心 (5,0)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

# 运行实验
viewpoint = np.array([0, 0])
points = generate_spiral()
alphas = [1.1, 10, 100, 1000]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, alpha in enumerate(alphas):
    vis_idx, _ = visibility_hpr(points, viewpoint, alpha)
    
    axes[i].scatter(points[:, 0], points[:, 1], c='gray', s=5, alpha=0.3, label='Hidden')
    axes[i].scatter(points[vis_idx, 0], points[vis_idx, 1], c='blue', s=10, label='Visible')
    axes[i].scatter(viewpoint[0], viewpoint[1], c='red', marker='x', s=100, label='Viewpoint')
    axes[i].set_title(f"Alpha = {alpha}")
    axes[i].legend()

plt.tight_layout()
plt.show()