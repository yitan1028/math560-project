import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

def visibility_hpr_3d(points, viewpoint, alpha):
    """
    3D 版 Hidden Point Removal 算子
    """
    # 1. 以视点为原点平移
    p_adj = points - viewpoint
    norms = np.linalg.norm(p_adj, axis=1, keepdims=True)
    
    # 2. 计算反射半径 R
    R = alpha * np.max(norms)
    
    # 3. 3D 球面反射映射
    p_reflected = p_adj + 2 * (R - norms) * (p_adj / norms)
    
    # 4. 加入视点(原点)并计算 3D 凸包
    points_with_origin = np.vstack([p_reflected, [0, 0, 0]])
    hull = ConvexHull(points_with_origin)
    
    # 5. 可见点索引
    visible_indices = [idx for idx in hull.vertices if idx < len(points)]
    return visible_indices, p_reflected, hull

# --- Q6: 生成 3D 球面采样点 ---
def generate_sphere_points(n_points=1000, center=(4, 0, 0)):
    # 抽取正态分布的三元组并归一化以获得球面均匀分布
    vec = np.random.randn(n_points, 3)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    # 移动到指定中心
    return vec + np.array(center)

# 参数设置
viewpoint = np.array([0, 0, 0])
points = generate_sphere_points(n_points=1200)
alpha = 3.0

# 运行算法
vis_idx, p_refl, hull = visibility_hpr_3d(points, viewpoint, alpha)

# --- 3D 可视化各步骤 ---
fig = plt.figure(figsize=(18, 6))

# Step 1: 原始点云与视点
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(points[:, 0], points[:, 1], points[:, 2], color='lightgray', s=2, alpha=0.3)
ax1.scatter(viewpoint[0], viewpoint[1], viewpoint[2], color='red', marker='x', s=100, label='Viewpoint')
ax1.set_title("1. Original 3D Sphere Points")

# Step 2: 反射空间与 3D 凸包
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(p_refl[:, 0], p_refl[:, 1], p_refl[:, 2], color='blue', s=1, alpha=0.1)
# 绘制凸包面片
for simplex in hull.simplices:
    # 只绘制不包含原点(视点)的面片
    if len(points) not in simplex:
        pts = p_refl[simplex]
        ax2.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2], color='cyan', alpha=0.2, linewidth=0.1)
ax2.set_title(f"2. Reflected Space & Hull (alpha={alpha})")

# Step 3: 最终可见性结果
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(points[:, 0], points[:, 1], points[:, 2], color='lightgray', s=2, alpha=0.2)
ax3.scatter(points[vis_idx, 0], points[vis_idx, 1], points[vis_idx, 2], color='red', s=5, label='Visible')
ax3.scatter(viewpoint[0], viewpoint[1], viewpoint[2], color='blue', marker='x', s=100)
ax3.set_title("3. Final Visible Points (Red)")

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout()
plt.savefig('Q6_3D_Visibility_Steps.png', dpi=300)
print("Q6_3D_Visibility_Steps.png 已保存。")