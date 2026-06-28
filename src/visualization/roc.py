import os
import matplotlib.pyplot as plt


def plot_roc_curve(fpr, tpr, roc_auc, fig_dir, logger):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set")
    ax.legend(loc="lower right")
    path = os.path.join(fig_dir, "roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ROC curve -> {path}")
