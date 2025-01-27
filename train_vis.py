import numpy as np
import matplotlib.pyplot as plt

attempt = "vgg_LPIPSplusplus"

# Load saved data
acc_x = np.load(f"checkpoints/{attempt}/web/acc_r_x.npy")
acc_y = np.load(f"checkpoints/{attempt}/web/acc_r_y.npy")
loss_x = np.load(f"checkpoints/{attempt}/web/loss_total_x.npy")
loss_y = np.load(f"checkpoints/{attempt}/web/loss_total_y.npy")

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot accuracy
ax1.plot(acc_x, acc_y, "b-")
ax1.set_title("Accuracy during training")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.grid(True)

# Plot loss
ax2.plot(loss_x, loss_y, "r-")
ax2.set_title("Loss during training")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.grid(True)

fig.tight_layout()
fig.savefig(f"training_plot_{attempt}.png")
