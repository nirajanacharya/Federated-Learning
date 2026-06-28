import os
import matplotlib.pyplot as plt


def plot_training_curves(history, fig_dir, logger):
    rounds = range(1, len(history["val_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(rounds, history["val_acc"], marker="o", color="steelblue")
    axes[0].set_title("Val Accuracy per Round")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(rounds, history["val_f1"], marker="s", color="seagreen")
    axes[1].set_title("Val F1 per Round")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("F1 Score")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    path = os.path.join(fig_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved training curves -> {path}")


def plot_client_losses(history, fig_dir, logger):
    fig, ax = plt.subplots(figsize=(10, 4))
    for cid, losses in history["client_losses"].items():
        ax.plot(losses, label=f"Client {cid}", alpha=0.75)
    ax.set_title("Client Local Training Loss per Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("BCE Loss")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = os.path.join(fig_dir, "client_losses.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved client losses -> {path}")
