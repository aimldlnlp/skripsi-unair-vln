import numpy as np
import matplotlib.pyplot as plt


def main():
    # Observed summary metrics from the thesis table
    mean_nav_sr_obs = 0.9982
    measured_task_sr = 0.9945
    m_values = [2.21, 2.71, 3.21, 3.71, 4.21]

    # Domain focused around the operative success regime
    x = np.linspace(0.99, 1.0, 700)

    # Curve family behind the narrative relation
    curves = {m: x ** m for m in m_values}

    # Observation is interpreted at m = 3.21
    m_obs = 3.21
    predicted_task_sr_obs = mean_nav_sr_obs ** m_obs
    delta = measured_task_sr - predicted_task_sr_obs

    fig = plt.figure(figsize=(8.2, 6.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.34, wspace=0.24)

    ax_top = fig.add_subplot(gs[0, :])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_fixed = fig.add_subplot(gs[1, 1])

    # Narrative palette and dual encoding (color + linestyle) for grayscale robustness
    cmap = plt.get_cmap("cividis")
    colors = [cmap(v) for v in np.linspace(0.2, 0.88, len(m_values))]
    linestyles = ["-", "--", "-", "--", "-."]

    for m, c, ls in zip(m_values, colors, linestyles):
        ax_top.plot(x, curves[m], color=c, lw=2.1, ls=ls, label=f"m = {m:.2f}")

    # Observation and model-consistent point
    ax_top.scatter([mean_nav_sr_obs], [measured_task_sr], color="#b22222", zorder=7, s=44, edgecolor="white", linewidth=0.7)
    ax_top.scatter([mean_nav_sr_obs], [predicted_task_sr_obs], color="#1b4f72", zorder=7, s=34, edgecolor="white", linewidth=0.6)

    ann = (
        f"Episode-level success: {mean_nav_sr_obs:.4f}\n"
        f"Estimated task-level success: {predicted_task_sr_obs:.4f}\n"
        f"Measured task-level success: {measured_task_sr:.4f}\n"
        f"Gap: {delta:+.7f}"
    )
    ax_top.text(
        0.99035,
        0.99965,
        ann,
        va="top",
        fontsize=8.4,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.95),
    )
    ax_top.annotate(
        "Observation point",
        xy=(mean_nav_sr_obs, measured_task_sr),
        xytext=(0.9966, 0.9974),
        fontsize=8.3,
        ha="left",
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.25"),
    )

    ax_top.set_title("Risk Accumulation in Long-Horizon Tasks", fontsize=11.3)
    ax_top.set_xlabel("Episode-level success")
    ax_top.set_ylabel("Task-level success")
    ax_top.set_xlim(0.99, 1.0)
    ax_top.set_ylim(0.9855, 1.00025)
    ax_top.grid(True, alpha=0.17, linewidth=0.65)
    ax_top.legend(loc="lower right", frameon=True, fontsize=8)

    # Bottom-left: compact diagnostic zoom near the observation
    for m, c, ls in zip(m_values, colors, linestyles):
        ax_zoom.plot(x, curves[m], color=c, lw=1.7, ls=ls)

    ax_zoom.scatter([mean_nav_sr_obs], [measured_task_sr], color="#b22222", zorder=7, s=34, edgecolor="white", linewidth=0.6)
    ax_zoom.scatter([mean_nav_sr_obs], [predicted_task_sr_obs], color="#1b4f72", zorder=7, s=28, edgecolor="white", linewidth=0.55)
    ax_zoom.axvline(mean_nav_sr_obs, color="0.5", ls="--", lw=1)
    ax_zoom.axhline(measured_task_sr, color="0.7", ls=":", lw=1)

    ax_zoom.set_title("Zoom Around Observation", fontsize=9.8)
    ax_zoom.set_xlabel("Episode-level success")
    ax_zoom.set_ylabel("Task-level success")
    ax_zoom.set_xlim(0.995, 1.0)
    ax_zoom.set_ylim(0.9918, 1.0002)
    ax_zoom.grid(True, alpha=0.17, linewidth=0.6)

    # Bottom-right: trend under fixed episode-level success
    m_axis = np.linspace(1, 6, 400)
    fixed_curve = mean_nav_sr_obs ** m_axis
    ax_fixed.plot(m_axis, fixed_curve, color="#1b4f72", lw=2.2)
    ax_fixed.scatter([m_obs], [predicted_task_sr_obs], color="#1b4f72", s=31, zorder=7, edgecolor="white", linewidth=0.55)
    ax_fixed.scatter([m_obs], [measured_task_sr], color="#b22222", s=34, zorder=7, edgecolor="white", linewidth=0.6)

    ax_fixed.set_title("At Fixed Episode-Level Success", fontsize=9.8)
    ax_fixed.set_xlabel("Effective horizon")
    ax_fixed.set_ylabel("Task-level success")
    ax_fixed.set_xlim(1, 6)
    ax_fixed.set_ylim(0.976, 1.0005)
    ax_fixed.grid(True, alpha=0.17, linewidth=0.6)

    out_path = "images/nav-tugas/fig_discussion_risk_curve_2d_v4_panels.pdf"
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
