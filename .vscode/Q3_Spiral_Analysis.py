import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import os

def visibility_hpr(points, viewpoint, alpha):
    # 1. 平移
    p_adj = points - viewpoint
    norms = np.linalg.norm(p_adj, axis=1, keepdims=True)
    
    # 2. 计算反射半径 R (基于 alpha)
    R = alpha * np.max(norms)
    
    # 3. 执行球面反射 (Spherical Inversion)
    # 使用公式: P_hat = P + 2*(R - ||P||) * (P / ||P||)
    p_reflected = p_adj + 2 * (R - norms) * (p_adj / norms)
    
    # 4. 加入视点(原点)计算凸包
    points_with_origin = np.vstack([p_reflected, [0, 0]])
    hull = ConvexHull(points_with_origin)
    
    # 5. 过滤出可见点索引
    visible_indices = [idx for idx in hull.vertices if idx < len(points)]
    return visible_indices

# --- 生成 Q3 要求的多圈螺旋线 ---
def generate_dense_spiral(n_points=1500, turns=3):
    theta = np.linspace(0, turns * 2 * np.pi, n_points)
    r = 0.6 * theta
    x = r * np.cos(theta) + 5
    y = r * np.sin(theta)
    return np.column_stack((x, y))

# 参数设置
viewpoint = np.array([0, 0])
points = generate_dense_spiral()
alphas = [1.1, 10, 100, 1000] # Q3 要求的不同 alpha 值 

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, alpha in enumerate(alphas):
    vis_idx = visibility_hpr(points, viewpoint, alpha)
    
    axes[i].scatter(points[:, 0], points[:, 1], c='lightgray', s=3, alpha=0.3, label='Hidden')
    axes[i].scatter(points[vis_idx, 0], points[vis_idx, 1], c='blue', s=10, label='Visible')
    axes[i].scatter(viewpoint[0], viewpoint[1], c='red', marker='x', s=150, label='Viewpoint')
    
    axes[i].set_title(f"Q3: Multi-turn Spiral (Alpha = {alpha})")
    axes[i].set_aspect('equal')
    axes[i].legend()

plt.tight_layout()
# 强制保存到当前文件夹
save_path = os.path.join(os.getcwd(), 'Q3_Results.png')
plt.savefig(save_path, dpi=300)
print(f"图像已成功保存至: {save_path}")