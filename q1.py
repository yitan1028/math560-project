import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def visibility_hpr(points, viewpoint, alpha):
    """
    HPR (Hidden Point Removal) 算法实现
    """
    # 1. 以视点为原点进行平移
    p_centered = points - viewpoint
    
    # 2. 计算反演半径 R_max
    norms = np.linalg.norm(p_centered, axis=1)
    R = np.max(norms)
    R_max = alpha * R
    
    # 3. 球面反演变换 (Spherical Inversion)
    # 该变换将点映射到包围球外部，使得遮挡点陷入“内部”
    norms_fixed = np.where(norms == 0, 1e-10, norms)
    reflection_factor = 2 * (R_max - norms_fixed)
    p_reflected = p_centered + (p_centered / norms_fixed[:, np.newaxis]) * reflection_factor[:, np.newaxis]
    
    # 4. 计算包含原点(视点)在内的凸包
    points_with_origin = np.vstack([p_reflected, [0, 0]])
    hull = ConvexHull(points_with_origin)
    
    # 5. 提取凸包顶点的索引 (排除原点)
    visible_indices = [idx for idx in hull.vertices if idx < len(points)]
    return visible_indices

def generate_circle_points(m=100):
    """生成 Q1 指定的圆采样点"""
    theta = np.linspace(0, 2 * np.pi, m, endpoint=False)
    p_x = 4 + np.cos(theta)
    p_y = np.sin(theta)
    return np.vstack((p_x, p_y)).T

# --- 实验：生成三个例子 ---
examples = [
    {'c': np.array([0, 0]), 'alpha': 1.2, 'title': 'Example 1: Far Viewpoint'},
    {'c': np.array([2.5, 0.5]), 'alpha': 2.0, 'title': 'Example 2: Close & Off-center'},
    {'c': np.array([4, 3]), 'alpha': 1.1, 'title': 'Example 3: Top Viewpoint'}
]

points = generate_circle_points(m=100)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, ex in enumerate(examples):
    visible_idx = visibility_hpr(points, ex['c'], ex['alpha'])
    
    # 绘图
    axes[i].scatter(points[:, 0], points[:, 1], color='lightgray', s=15, label='Hidden')
    axes[i].scatter(points[visible_idx, 0], points[visible_idx, 1], color='blue', s=25, label='Visible')
    axes[i].scatter(ex['c'][0], ex['c'][1], color='red', marker='x', s=100, label='Viewpoint c')
    
    axes[i].set_title(f"{ex['title']}\n(c={ex['c']}, alpha={ex['alpha']})")
    axes[i].axis('equal')
    axes[i].grid(True, linestyle='--', alpha=0.6)
    axes[i].legend(loc='lower right')

plt.tight_layout()
plt.show()