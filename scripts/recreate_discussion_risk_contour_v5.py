import numpy as np
import matplotlib.pyplot as plt


def main():
    # Observed summary metrics from the thesis table
    mean_nav_sr_obs = 0.9982
    mean_task_sr_obs = 0.9945
    m_obs = 3.21

    # XY domain focused around the relevant operating regime
    x = np.linspace(0.985, 1.0, 450)  # episode-level success
    y = np.linspace(1.0, 7.0, 320)    # effective horizon length
    X, Y = np.meshgrid(x, y)

    # Narrative relation behind the contour surface
    Z = np.power(X, Y)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    # Soft filled contour for paper-style readability
    filled_levels = np.linspace(0.95, 1.0, 16)
    cf = ax.contourf(X, Y, Z, levels=filled_levels, cmap="cividis", alpha=0.9)

    # Keep only key isolines to avoid clutter
    line_levels = [0.985, 0.990, 0.995]
    cs = ax.contour(X, Y, Z, levels=line_levels, colors="white", linewidths=1.0, alpha=0.95)
    ax.clabel(cs, fmt=lambda v: f"{v:.3f}", fontsize=8, inline=True)

    # Observation marker with halo to create a clear visual anchor
    ax.scatter([mean_nav_sr_obs], [m_obs], s=190, facecolor="none", edgecolor="black", lw=1.0, zorder=6, alpha=0.35)
    ax.scatter([mean_nav_sr_obs], [m_obs], s=55, color="#c0392b", edgecolor="white", linewidth=0.8, zorder=7)

    # Short narrative callouts (without explicit equations)
    ax.annotate(
        "Observation point",
        xy=(mean_nav_sr_obs, m_obs),
        xytext=(0.9928, 5.95),
        fontsize=8.8,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.2"),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.95),
    )
    ax.text(
        0.9862,
        1.37,
        "Longer horizon\n-> task-level success drop\nbecomes more visible",
        fontsize=8.3,
        color="0.15",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.92),
    )

    # Axes, title, and colorbar tuned for thesis readability
    ax.set_title("End-to-End Success Landscape in Long-Horizon Tasks", fontsize=11.5, pad=9)
    ax.set_xlabel("Episode-level success")
    ax.set_ylabel("Effective horizon (number of episodes)")
    ax.set_xlim(0.985, 1.0)
    ax.set_ylim(1.0, 7.0)
    ax.grid(True, alpha=0.12, linewidth=0.6)

    cbar = fig.colorbar(cf, ax=ax, pad=0.018, fraction=0.05)
    cbar.set_label("Estimated task-level success")
    cbar.set_ticks([0.95, 0.97, 0.99, 1.00])

    fig.tight_layout()

    out_path = "images/nav-tugas/fig_discussion_risk_contour_2d_v4_dense_left.pdf"
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
