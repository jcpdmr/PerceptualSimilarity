import json

path = "checkpoints/vgg_custom0"

# Read the two JSON files
with open(f"{path}/val_results_verbose.json", "r") as f:
    val_results = json.load(f)
    val_results = val_results["results"]  # Access the nested "results" key

with open(f"{path}/h_values_with_swap.json", "r") as f:
    h_values = json.load(f)

# Create the new dictionary
deswapped_results = {}

for img_name in val_results:
    # Convert .png to .jpg if necessary for matching
    h_values_key = img_name.replace(".png", ".jpg")

    if h_values_key in h_values:
        val_data = val_results[img_name]
        h_data = h_values[h_values_key]

        # Determine the deswapped values based on the swapped flag
        if h_data["swapped"]:
            d0_deswapped = val_data["d1"]  # swap
            d1_deswapped = val_data["d0"]  # swap
        else:
            d0_deswapped = val_data["d0"]  # no swap
            d1_deswapped = val_data["d1"]  # no swap
        e24 = h_data["24"]
        e55 = h_data["55"]
        # Create the new dictionary for this image
        deswapped_results[img_name] = {
            "d0_swapped": val_data["d0"],
            "d1_swapped": val_data["d1"],
            "d0_deswapped": d0_deswapped,
            "d1_deswapped": d1_deswapped,
            "e24": e24,
            "e55": e55,
            "e24-e55": e24 - e55,
            "d0-d1": d0_deswapped - d1_deswapped,
            "h_diff_orig": h_data["h_diff"],
            "h_diff_swap": val_data["h (gt)"],  # ["h_diff_new"]
            "score": val_data["score"],
            "was_swapped": h_data["swapped"],
        }

# Save the new JSON
with open(f"{path}/deswapped_results.json", "w") as f:
    json.dump(deswapped_results, f, indent=4, sort_keys=True)
