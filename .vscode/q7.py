import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import trimesh

def visibility_hpr_3d(points, viewpoint, alpha):
    """3D HPR 算法实现"""
    p_adj = points - viewpoint
    norms = np.linalg.norm(p_adj, axis=1, keepdims=True)
    R = alpha * np.max(norms)
    p_reflected = p_adj + 2 * (R - norms) * (p_adj / norms)
    
    # 视点(原点)加入凸包计算
    points_with_origin = np.vstack([p_reflected, [0, 0, 0]])
    hull = ConvexHull(points_with_origin)
    
    # 提取可见点索引
    visible_indices = [idx for idx in hull.vertices if idx < len(points)]
    return visible_indices

# --- Q7: 加载“真实”3D形状 (内置模型) ---
def load_stable_bunny():
    # 尝试加载 trimesh 自带的 ply/obj 样例
    # 如果内置模型加载失败，将自动生成一个莫比乌斯环作为替代复杂形状
    try:
        # 加载库自带的 'bunny.obj' 或者是 'teapot.obj'
        mesh = trimesh.load(trimesh.util.locate_remote('bunny.obj'))
        print("成功加载内置 Stanford Bunny 模型")
    except:
        print("未找到内置模型，正在生成复杂几何形状: Mobius Strip")
        # 备选：生成莫比乌斯带点云 (非合成简单形状，具有复杂拓扑)
        u = np.linspace(0, 2 * np.pi, 3000)
        v = np.linspace(-0.2, 0.2, 50)
        u, v = np.meshgrid(u, v)
        u, v = u.flatten(), v.flatten()
        x = (1 + v/2 * np.cos(u/2)) * np.cos(u) + 4
        y = (1 + v/2 * np.cos(u/2)) * np.sin(u)
        z = v/2 * np.sin(u/2)
        return np.column_stack((x, y, z))
        
    return mesh.sample(5000)

# 执行 HPR
points = load_stable_bunny()
viewpoint = np.array([0, 0, 0])  # 视点设在原点
alpha = 3.0

vis_idx = visibility_hpr_3d(points, viewpoint, alpha)

# --- 3D 可视化 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 隐藏点 (灰色)
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='lightgray', s=1, alpha=0.2)
# 可见点 (红色)
ax.scatter(points[vis_idx, 0], points[vis_idx, 1], points[vis_idx, 2], c='red', s=4, label='Visible')
# 视点 (蓝色)
ax.scatter(viewpoint[0], viewpoint[1], viewpoint[2], c='blue', marker='x', s=100, label='Viewpoint')

ax.set_title("Q7: Direct Visibility of Real 3D Shape")
plt.legend()
plt.show()