import json
import numpy as np
from pathlib import Path


def calculate_metrics(json_path):
    """
    Calculate MAE and MSE between d0_deswapped/e24 and d1_deswapped/e55

    Args:
        json_path: Path to the deswapped_results.json file

    Returns:
        Dictionary containing the calculated metrics
    """
    # Load the JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract the relevant values into arrays
    d0_deswapped = []
    d1_deswapped = []
    e24_values = []
    e55_values = []

    for img_data in data.values():
        d0_deswapped.append(img_data["d0_deswapped"])
        d1_deswapped.append(img_data["d1_deswapped"])
        e24_values.append(img_data["e24"])
        e55_values.append(img_data["e55"])

    # Convert to numpy arrays
    d0_deswapped = np.array(d0_deswapped)
    d1_deswapped = np.array(d1_deswapped)
    e24_values = np.array(e24_values)
    e55_values = np.array(e55_values)

    # Calculate metrics
    metrics = {
        "d0_e24_mae": float(np.mean(np.abs(d0_deswapped - e24_values))),
        "d0_e24_mse": float(np.mean((d0_deswapped - e24_values) ** 2)),
        "d1_e55_mae": float(np.mean(np.abs(d1_deswapped - e55_values))),
        "d1_e55_mse": float(np.mean((d1_deswapped - e55_values) ** 2)),
    }

    return metrics


def process_network_trial(net, trial):
    """
    Process metrics for a specific network and trial

    Args:
        net: Network name
        trial: Trial number/identifier
    """
    input_path = Path(f"checkpoints/{net}_{trial}/deswapped_results.json")
    output_path = Path(f"checkpoints/{net}_{trial}/result_metrics.json")

    # Calculate metrics
    metrics = calculate_metrics(input_path)

    # Save results
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    process_network_trial("vgg", "custom1")
