import matplotlib.pyplot as plt
import numpy as np

# Source values from Table 1.13 (tab:navtask_summary) in bab4.tex
MEAN_TASK_STEP = 233.70
MEAN_NAV_STEP = 72.85

# Keep failure-rate values consistent with the previous published figure
# (more precise than deriving from rounded SR values in the table)
TASK_FR_X1E3 = 5.464
NAV_FR_X1E3 = 1.821


def style_axes(ax):
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_steps_lollipop(out_path):
    labels = ["Mean Task Step", "Mean Navigation Step"]
    values = [MEAN_TASK_STEP, MEAN_NAV_STEP]
    y = np.arange(len(labels))[::-1]

    ratio = MEAN_TASK_STEP / MEAN_NAV_STEP
    delta = MEAN_TASK_STEP - MEAN_NAV_STEP

    fig, ax = plt.subplots(figsize=(8.6, 3.9))

    for yi, xv in zip(y, values):
        ax.hlines(yi, 0, xv, color="#8ea3b5", linewidth=3.2, alpha=0.9)

    ax.scatter(values, y, s=[260, 220], c=["#0b5fa5", "#2d9cdb"],
               edgecolors="white", linewidths=1.5, zorder=3)

    ax.text(MEAN_TASK_STEP + 4, y[0], f"{MEAN_TASK_STEP:.2f}", va="center", ha="left",
            fontsize=10, color="#0b3a63", fontweight="bold")
    ax.text(MEAN_NAV_STEP + 4, y[1], f"{MEAN_NAV_STEP:.2f}", va="center", ha="left",
            fontsize=10, color="#0b3a63", fontweight="bold")

    x_mid = (MEAN_TASK_STEP + MEAN_NAV_STEP) / 2
    y_mid = (y[0] + y[1]) / 2
    ax.annotate(
        f"Ratio = {ratio:.2f}x\nGap = {delta:.2f} steps",
        xy=(x_mid, y_mid),
        xytext=(x_mid + 16, y_mid + 0.48),
        textcoords="data",
        fontsize=9,
        color="#2f3e46",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#b8c1c8", alpha=0.95),
        arrowprops=dict(arrowstyle="-|>", color="#7a8a99", lw=1.0)
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Steps", fontsize=10)
    ax.set_title("Task-Level vs Episode-Level Mean Steps", fontsize=11, pad=10)
    ax.set_xlim(0, 260)

    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_failure_rate_lollipop(out_path):
    labels = ["Task Failure Rate (1-SR)", "Navigation Failure Rate (1-SR)"]
    task_fr = TASK_FR_X1E3
    nav_fr = NAV_FR_X1E3
    values = [task_fr, nav_fr]
    y = np.arange(len(labels))[::-1]

    ratio = task_fr / nav_fr
    delta = task_fr - nav_fr

    fig, ax = plt.subplots(figsize=(8.6, 3.9))

    for yi, xv in zip(y, values):
        ax.hlines(yi, 0, xv, color="#c4b5a5", linewidth=3.2, alpha=0.95)

    ax.scatter(values, y, s=[260, 220], c=["#a63603", "#e6550d"],
               edgecolors="white", linewidths=1.5, zorder=3)

    ax.text(task_fr + 0.08, y[0], f"{task_fr:.3f}", va="center", ha="left",
            fontsize=10, color="#7f2704", fontweight="bold")
    ax.text(nav_fr + 0.08, y[1], f"{nav_fr:.3f}", va="center", ha="left",
            fontsize=10, color="#7f2704", fontweight="bold")

    x_mid = (task_fr + nav_fr) / 2
    y_mid = (y[0] + y[1]) / 2
    ax.annotate(
        f"Ratio = {ratio:.2f}x\nGap = {delta:.3f}",
        xy=(x_mid, y_mid),
        xytext=(x_mid + 0.35, y_mid + 0.48),
        textcoords="data",
        fontsize=9,
        color="#2f3e46",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d0c2b2", alpha=0.95),
        arrowprops=dict(arrowstyle="-|>", color="#8b6f54", lw=1.0)
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Failure Rate (x10^-3)", fontsize=10)
    ax.set_title("Failure Rate Comparison at Task and Episode Levels", fontsize=11, pad=10)
    ax.set_xlim(0, 6.2)

    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_steps_lollipop("images/nav-tugas/fig_results_steps_lollipop_v2.pdf")
    fig_failure_rate_lollipop("images/nav-tugas/fig_results_failure_rate_lollipop_v2.pdf")
    print("Saved:")
    print("- images/nav-tugas/fig_results_steps_lollipop_v2.pdf")
    print("- images/nav-tugas/fig_results_failure_rate_lollipop_v2.pdf")


if __name__ == "__main__":
    main()
