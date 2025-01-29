import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def plot_weight_evolution(attempt, n_epochs, save_path, architecture="vgg"):
    """
    Plot the evolution of learned weights across training epochs

    Args:
        attempt: Name of the checkpoint directory
        n_epochs: Number of epochs to plot
        save_path: Where to save the resulting plot
        architecture: Network architecture (vgg/alex/squeeze)
    """
    # Set up plot parameters based on architecture
    if architecture == "vgg":
        n_blocks = 5
        block_names = ["conv1_2", "conv2_2", "conv3_3", "conv4_3", "conv5_3"]
    elif architecture == "alex":
        n_blocks = 5
        block_names = ["conv1", "conv2", "conv3", "conv4", "conv5"]
    elif architecture == "squeeze":
        n_blocks = 7
        block_names = ["conv1", "fire3", "fire5", "fire6", "fire7", "fire8", "fire9"]

    # Create figure
    fig, axs = plt.subplots(n_epochs, n_blocks, figsize=(20, 4 * n_epochs))
    fig.suptitle(
        f"Evolution of Learned Weights ({architecture.upper()})", fontsize=16, y=1.00
    )

    # For each epoch
    for epoch in range(1, n_epochs + 1):
        model_path = f"checkpoints/{attempt}/{epoch}_net_.pth"

        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping...")
            continue

        # Load the model weights
        state_dict = torch.load(model_path, map_location="cpu")

        # Plot each block for this epoch
        for block in range(n_blocks):
            key = f"lin{block}.model.1.weight"
            w = state_dict[key]
            w_np = w.squeeze().numpy()
            sorted_weights = np.sort(w_np)[::-1]

            # Get the current axis
            ax = axs[epoch - 1, block]

            # Plot
            ax.plot(range(len(w_np)), sorted_weights)

            # Set titles only for first row
            if epoch == 1:
                ax.set_title(block_names[block])

            # Set labels
            if block == 0:
                ax.set_ylabel(f"Weight - Epoch {epoch}")
            if epoch == n_epochs:
                ax.set_xlabel("Channel number")

            ax.grid(True)

            # Print statistics
            print(f"Epoch {epoch}, Block {block_names[block]}:")
            print(
                f"  Non-zero weights: {(w_np > 0).sum()}/{len(w_np)} "
                f"({(w_np > 0).sum() / len(w_np) * 100:.1f}%)"
            )
            print(f"  Max weight: {w_np.max():.3f}")
            print(f"  Mean weight: {w_np.mean():.3f}\n")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


attempt = "vgg_custom1"

# Weights plot usage
save_path = f"weight_evolution_{attempt}.png"
plot_weight_evolution(
    attempt=attempt, n_epochs=10, save_path=save_path, architecture="vgg"
)

# Load saved data
acc_x = np.load(f"checkpoints/{attempt}/web/acc_r_x.npy")
acc_y = np.load(f"checkpoints/{attempt}/web/acc_r_y.npy")
loss_x = np.load(f"checkpoints/{attempt}/web/loss_total_x.npy")
loss_y = np.load(f"checkpoints/{attempt}/web/loss_total_y.npy")


# Apply moving average filter
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


# Smooth the data
window_size = 15
acc_y_smooth = moving_average(acc_y, window_size)
loss_y_smooth = moving_average(loss_y, window_size)

# Adjust x values for the smoothed data
acc_x_smooth = acc_x[window_size - 1 :]
loss_x_smooth = loss_x[window_size - 1 :]

# Create figure with original and smoothed data
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot original accuracy
ax1.plot(acc_x, acc_y, "b-", alpha=0.3, label="Original")
ax1.plot(acc_x_smooth, acc_y_smooth, "b-", label="Smoothed")
ax1.set_title("Accuracy (Original vs Smoothed)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.grid(True)
ax1.legend()

# Plot original loss
ax2.plot(loss_x, loss_y, "r-", alpha=0.3, label="Original")
ax2.plot(loss_x_smooth, loss_y_smooth, "r-", label="Smoothed")
ax2.set_title("Loss (Original vs Smoothed)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.grid(True)
ax2.legend()

# Plot only smoothed accuracy
ax3.plot(acc_x_smooth, acc_y_smooth, "b-")
ax3.set_title("Smoothed Accuracy")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Accuracy")
ax3.grid(True)

# Plot only smoothed loss
ax4.plot(loss_x_smooth, loss_y_smooth, "r-")
ax4.set_title("Smoothed Loss")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Loss")
ax4.grid(True)

fig.tight_layout()
fig.savefig(f"training_plot_{attempt}.png")
