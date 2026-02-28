import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def main():
    # Paper-like typography: serif text with Times-style fallback.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })

    rng = np.random.default_rng(42)

    # Synthetic-but-structured points for visual explanation only
    n_succ = 95
    n_fail = 65

    x_s = np.clip(rng.normal(250, 85, n_succ), 40, 540)   # N_step
    y_s = np.clip(rng.normal(92, 28, n_succ), 15, 190)    # Delta t
    z_s = np.ones(n_succ)

    x_f = np.clip(rng.normal(325, 95, n_fail), 50, 560)
    y_f = np.clip(rng.normal(118, 34, n_fail), 25, 210)
    z_f = np.zeros(n_fail)

    fig = plt.figure(figsize=(9.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    # Background planes to emphasize partition at z=0 and z=1
    xx, yy = np.meshgrid(np.linspace(0, 560, 2), np.linspace(0, 220, 2))
    ax.plot_surface(xx, yy, np.zeros_like(xx), color="#ffe5dc", alpha=0.25, linewidth=0, shade=False)
    ax.plot_surface(xx, yy, np.ones_like(xx), color="#dbeeff", alpha=0.22, linewidth=0, shade=False)

    # Scatter clouds
    ax.scatter(x_f, y_f, z_f, s=25, color="#d1495b", edgecolors="white", linewidths=0.4, alpha=0.95, label="Fail tasks (z=0)")
    ax.scatter(x_s, y_s, z_s, s=25, color="#2b6cb0", edgecolors="white", linewidths=0.4, alpha=0.95, label="Success tasks (z=1)")

    # Centroids
    cxf, cyf = x_f.mean(), y_f.mean()
    cxs, cys = x_s.mean(), y_s.mean()
    ax.scatter([cxf], [cyf], [0], s=130, marker="X", color="#8b1e2d", edgecolors="white", linewidths=1.0)
    ax.scatter([cxs], [cys], [1], s=130, marker="X", color="#1f4f80", edgecolors="white", linewidths=1.0)

    ax.text(cxf + 14, cyf + 8, 0.03, r"$T_{fail}$", color="#7a1626", fontsize=11, weight="bold")
    ax.text(cxs + 14, cys + 8, 1.03, r"$T_{succ}$", color="#173c63", fontsize=11, weight="bold")

    # Axis and styling
    ax.set_xlim(0, 560)
    ax.set_ylim(0, 220)
    ax.set_zlim(-0.1, 1.1)

    ax.set_xlabel(r"$N_{step}$", labelpad=10)
    ax.set_ylabel(r"$\Delta t$", labelpad=10)
    ax.set_zlabel("Execution outcome", labelpad=9)

    ax.set_zticks([0, 1])
    ax.set_zticklabels(["Fail", "Success"])

    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.grid(True, alpha=0.25)

    ax.view_init(elev=23, azim=-57)

    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=9)

    out = "images/nav-tugas/fig_protocol_3d_partition_en_v3.pdf"
    fig.subplots_adjust(left=0.02, right=0.93, bottom=0.03, top=0.98)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
