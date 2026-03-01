import numpy as np
import matplotlib.pyplot as plt

def analyze_mapping(alpha_values):
    # 创建一个简单的垂直线段点集（模拟遮挡物）
    points = np.column_stack((np.full(100, 5), np.linspace(-2, 2, 100)))
    viewpoint = np.array([0, 0])
    
    plt.figure(figsize=(12, 5))
    
    for i, alpha in enumerate(alpha_values):
        p_adj = points - viewpoint
        norms = np.linalg.norm(p_adj, axis=1, keepdims=True)
        R = alpha * np.max(norms)
        
        # HPR 核心映射公式
        p_reflected = p_adj + 2 * (R - norms) * (p_adj / norms)
        
        plt.subplot(1, len(alpha_values), i+1)
        plt.scatter(p_reflected[:, 0], p_reflected[:, 1], s=10, label=f'Alpha={alpha}')
        plt.title(f"Reflected Space (Alpha={alpha})")
        plt.axis('equal')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 运行观察映射空间的变化
analyze_mapping([2, 100, 1000])