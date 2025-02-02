import plotly.graph_objects as go
import json

path = "checkpoints/vgg_custom0"

# Read the JSON
with open(f"{path}/deswapped_results.json", "r") as f:
    data = json.load(f)

# Prepare arrays for the three scatter plots
d0_deswapped = []
e24_values = []
d1_deswapped = []
e55_values = []
d_diff = []
e_diff = []

for img_name, values in data.items():
    d0_deswapped.append(values["d0_deswapped"])
    e24_values.append(values["e24"])
    d1_deswapped.append(values["d1_deswapped"])
    e55_values.append(values["e55"])
    d_diff.append((values["d0_deswapped"] - values["d1_deswapped"]))
    e_diff.append((values["e24"] - values["e55"]))

# Create three separate figures
fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=d0_deswapped,
        y=e24_values,
        mode="markers",
        name="d0 vs e24",
        marker=dict(color="blue", size=2),
    )
)
fig1.update_layout(
    title="Relationship between d0 and e24",
    xaxis_title="LPIPS Distance (d0)",
    yaxis_title="e24 Value",
    hovermode="closest",
)

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=d1_deswapped,
        y=e55_values,
        mode="markers",
        name="d1 vs e55",
        marker=dict(color="red", size=2),
    )
)
fig2.update_layout(
    title="Relationship between d1 and e55",
    xaxis_title="LPIPS Distance (d1)",
    yaxis_title="e55 Value",
    hovermode="closest",
)

fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(
        x=d_diff,
        y=e_diff,
        mode="markers",
        name="d0-d1 vs e24-e55",
        marker=dict(color="green", size=2),
    )
)
fig3.update_layout(
    title="Relationship between distance differences",
    xaxis_title="d0-d1",
    yaxis_title="e24-e55",
    hovermode="closest",
)

# Show all figures
# fig1.show()
# fig2.show()
# fig3.show()


# Save all figures
fig1.write_html(f"{path}/error_dist_d0_e24.html")
fig1.write_image(f"{path}/error_dist_d0_e24.png", width=1080, height=1080, scale=2)

fig2.write_html(f"{path}/error_dist_d1_e55.html")
fig2.write_image(f"{path}/error_dist_d1_e55.png", width=1080, height=1080, scale=2)

fig3.write_html(f"{path}/error_dist_diff.html")
fig3.write_image(f"{path}/error_dist_diff.png", width=1080, height=1080, scale=2)
