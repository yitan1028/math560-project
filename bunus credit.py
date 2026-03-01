import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# -------------------------
# Q1 Example 1 (Python)
# -------------------------

def generate_circle(m, center=(4.0, 0.0), radius=1.0):
    theta = np.linspace(0, 2*np.pi, m, endpoint=False)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.column_stack([x, y])

def reflect_points(P, c, R):
    # p_ref = c + (R^2 / ||p-c||^2) * (p-c)
    V = P - c
    denom = np.sum(V**2, axis=1, keepdims=True)
    return c + (R**2) * (V / denom)

def main():
    # ----- parameters -----
    m = 120
    c = np.array([0.0, 0.0])
    alpha = 1.2

    # ----- data -----
    P = generate_circle(m, center=(4.0, 0.0), radius=1.0)

    # reflection radius
    d = np.linalg.norm(P - c, axis=1)
    R = alpha * np.max(d)

    # reflect
    P_ref = reflect_points(P, c, R)

    # convex hull on combined points
    all_pts = np.vstack([P, P_ref])
    hull = ConvexHull(all_pts)
    hull_idx = hull.vertices

    # visible original indices = hull vertices that belong to P
    n = len(P)
    vis_idx_P = np.sort(hull_idx[hull_idx < n])
    visible_P = P[vis_idx_P]

    # "corresponding" reflected points (same indices)
    visible_ref = P_ref[vis_idx_P]

    # ----- choose two extreme rays (min/max angle among visible originals) -----
    ang = np.arctan2(visible_P[:, 1] - c[1], visible_P[:, 0] - c[0])
    p_min = visible_P[np.argmin(ang)]
    p_max = visible_P[np.argmax(ang)]

    # ----- plot -----
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_aspect("equal", adjustable="box")

    # reflection circle
    t = np.linspace(0, 2*np.pi, 400)
    ax.plot(c[0] + R*np.cos(t), c[1] + R*np.sin(t),
            "k--", lw=2, alpha=0.8, label="Reflection Circle", zorder=0)

    # original points
    ax.scatter(P[:, 0], P[:, 1],
               s=40, c="dodgerblue", alpha=0.35, label="Original", zorder=2)

    # reflected points (background)
    ax.scatter(P_ref[:, 0], P_ref[:, 1],
               s=40, c="gray", alpha=0.18, label="Reflected", zorder=1)

    # visible original points (RED)
    ax.scatter(visible_P[:, 0], visible_P[:, 1],
               s=140, c="red", edgecolors="black", linewidths=1.0,
               label="Visible (original)", zorder=6)

    # visible reflected points (PURPLE)  <-- 右边那圈你要的高亮
    ax.scatter(visible_ref[:, 0], visible_ref[:, 1],
               s=120, c="magenta", edgecolors="black", linewidths=1.0,
               alpha=1.0, label="Visible (reflected)", zorder=7)

    # viewpoint
    ax.scatter([c[0]], [c[1]],
               s=220, c="green", edgecolors="black", linewidths=0.8,
               label="Viewpoint c", zorder=8)

    # two extreme rays (YELLOW)
    ax.plot([c[0], p_min[0]], [c[1], p_min[1]], color="gold", lw=4, zorder=5)
    ax.plot([c[0], p_max[0]], [c[1], p_max[1]], color="gold", lw=4, zorder=5)

    # optional: fill wedge for visibility cone
    ax.fill([c[0], p_min[0], p_max[0]], [c[1], p_min[1], p_max[1]],
            color="gold", alpha=0.25, zorder=3)

    ax.grid(True, alpha=0.25)
    ax.set_title(f"Example 1: c=(0,0), alpha={alpha}")
    ax.legend(loc="upper right")

    print("num visible (original) =", len(vis_idx_P))
    plt.show()

if __name__ == "__main__":
    main()