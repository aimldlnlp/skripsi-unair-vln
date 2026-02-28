import numpy as np
import matplotlib.pyplot as plt


def main():
    # Parameters inferred from the existing figure annotations
    mean_nav_sr_obs = 0.9982
    measured_task_sr = 0.9945
    m_values = [2.21, 2.71, 3.21, 3.71, 4.21]

    x = np.linspace(0.99, 1.0, 800)

    # Main curve set uses TaskSR = (MeanNavSR)^m
    curves = {m: x ** m for m in m_values}

    # Observation is compared at m = 3.21 (as labeled in figure)
    m_obs = 3.21
    predicted_task_sr_obs = mean_nav_sr_obs ** m_obs
    delta = measured_task_sr - predicted_task_sr_obs

    fig = plt.figure(figsize=(7.8, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.22)

    ax_top = fig.add_subplot(gs[0, :])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_fixed = fig.add_subplot(gs[1, 1])

    # Top panel
    cmap = plt.get_cmap("viridis")
    colors = [cmap(v) for v in np.linspace(0.15, 0.85, len(m_values))]

    for m, c in zip(m_values, colors):
        ax_top.plot(x, curves[m], color=c, lw=2.0, label=f"m = {m:.2f}")

    ax_top.scatter([mean_nav_sr_obs], [measured_task_sr], color="#d62728", zorder=6, s=35)
    ax_top.scatter([mean_nav_sr_obs], [predicted_task_sr_obs], color="#1f77b4", zorder=6, s=28)

    ann = (
        f"MeanNavSR = {mean_nav_sr_obs:.4f}\n"
        f"Predicted (MeanNavSR)$^m$ = {predicted_task_sr_obs:.4f}\n"
        f"Measured MeanTaskSR = {measured_task_sr:.4f}\n"
        f"$\\Delta$ = {delta:+.7f}"
    )
    ax_top.text(0.9903, 0.9996, ann, va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.95))

    ax_top.set_title("Risk Accumulation In Long-Horizon Tasks: MeanTaskSR $(\\mathrm{MeanNavSR})^m$", fontsize=11)
    ax_top.set_xlabel("MeanNavSR (episode-level success)")
    ax_top.set_ylabel("Predicted TaskSR")
    ax_top.set_xlim(0.99, 1.0)
    ax_top.set_ylim(0.9855, 1.0002)
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="center right", frameon=True, fontsize=8)

    # Bottom-left zoom panel
    for m, c in zip(m_values, colors):
        ax_zoom.plot(x, curves[m], color=c, lw=1.8)

    ax_zoom.scatter([mean_nav_sr_obs], [measured_task_sr], color="#d62728", zorder=7, s=30)
    ax_zoom.scatter([mean_nav_sr_obs], [predicted_task_sr_obs], color="#1f77b4", zorder=7, s=24)
    ax_zoom.axvline(mean_nav_sr_obs, color="0.5", ls="--", lw=1)
    ax_zoom.axhline(measured_task_sr, color="0.7", ls=":", lw=1)

    ax_zoom.set_title("Zoom At Observation", fontsize=10)
    ax_zoom.set_xlabel("MeanNavSR")
    ax_zoom.set_ylabel("TaskSR")
    ax_zoom.set_xlim(0.995, 1.0)
    ax_zoom.set_ylim(0.9918, 1.0002)
    ax_zoom.grid(True, alpha=0.25)

    # Bottom-right fixed MeanNavSR panel
    m_axis = np.linspace(1, 6, 400)
    fixed_curve = mean_nav_sr_obs ** m_axis
    ax_fixed.plot(m_axis, fixed_curve, color="#1f77b4", lw=2.2)
    ax_fixed.scatter([m_obs], [predicted_task_sr_obs], color="#1f77b4", s=28, zorder=6)
    ax_fixed.scatter([m_obs], [measured_task_sr], color="#d62728", s=32, zorder=6)

    ax_fixed.set_title("At Fixed MeanNavSR", fontsize=10)
    ax_fixed.set_xlabel("m")
    ax_fixed.set_ylabel("TaskSR")
    ax_fixed.set_xlim(1, 6)
    ax_fixed.set_ylim(0.976, 1.0005)
    ax_fixed.grid(True, alpha=0.25)

    out_path = "images/nav-tugas/fig_discussion_risk_curve_2d_v4_panels.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
