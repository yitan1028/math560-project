import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage import data, color, filters, measure

def visibility_hpr(points, viewpoint, alpha):
    # 1. 以视点为原点进行平移
    p_adj = points - viewpoint
    norms = np.linalg.norm(p_adj, axis=1, keepdims=True)
    
    # 2. 计算反射半径 R
    R = alpha * np.max(norms)
    
    # 3. 球面反射映射 (Spherical Inversion)
    # 映射公式: P' = P + 2*(R - ||P||) * (P / ||P||)
    p_reflected = p_adj + 2 * (R - norms) * (p_adj / norms)
    
    # 4. 加入视点并计算凸包
    points_with_origin = np.vstack([p_reflected, [0, 0]])
    hull = ConvexHull(points_with_origin)
    
    # 5. 可见点索引（排除最后的视点点）
    visible_indices = [idx for idx in hull.vertices if idx < len(points)]
    return visible_indices

# --- Q5: 加载“真实”形状 (Horse Silhouette) ---
def get_horse_points(num_samples=1500):
    image = data.horse()
    # 提取轮廓
    contours = measure.find_contours(image, 0.5)
    # 取最长的轮廓（马的主体）
    contour = max(contours, key=len)
    # 为了处理速度进行降采样
    indices = np.linspace(0, len(contour)-1, num_samples, dtype=int)
    points = contour[indices]
    # 归一化并移动到原点附近
    points = points - np.mean(points, axis=0)
    return points

# 准备实验
points = get_horse_points()
# 设置两个不同的视点：一个在侧面，一个在斜上方
viewpoints = [np.array([-150, 0]), np.array([0, 200])]
alphas = [3, 50] # 选择一个适中，一个较大的 alpha

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
titles = [
    f"Side View, Alpha={alphas[0]}", f"Side View, Alpha={alphas[1]}",
    f"Top View, Alpha={alphas[0]}", f"Top View, Alpha={alphas[1]}"
]

for i, vp in enumerate(viewpoints):
    for j, alpha in enumerate(alphas):
        ax = axes[i, j]
        vis_idx = visibility_hpr(points, vp, alpha)
        
        ax.scatter(points[:, 1], points[:, 0], c='lightgray', s=2, alpha=0.3, label='Hidden')
        ax.scatter(points[vis_idx, 1], points[vis_idx, 0], c='red', s=8, label='Visible')
        ax.scatter(vp[1], vp[0], c='blue', marker='x', s=100, label='Viewpoint')
        
        ax.set_title(titles[i*2 + j])
        ax.set_aspect('equal')
        ax.invert_yaxis() # 匹配图像坐标系
        ax.legend()

plt.tight_layout()
plt.savefig('Q5_RealShape_Results.png')
print("Q5_RealShape_Results.png saved.")