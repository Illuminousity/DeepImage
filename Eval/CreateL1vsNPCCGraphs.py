import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the dataset
file_path = "./csveval/evaluation_results.csv"
df = pd.read_csv(file_path)

# Determine the filename column
filename_col = None
if "ModelPath" in df.columns:
    filename_col = "ModelPath"
elif "ModelName" in df.columns:
    filename_col = "ModelName"
else:
    raise ValueError("CSV file must contain a 'ModelPath' or 'ModelName' column.")

# Filter dataset size = 60000
def extract_dataset_size(name):
    match = re.search(r"_(\d{5})_", name)
    return int(match.group(1)) if match else None

df["DatasetSize"] = df[filename_col].apply(extract_dataset_size)
df = df[df["DatasetSize"] == 60000]

# Parse loss function from the model name
df["LossFunction"] = df[filename_col].apply(lambda x: "NPCC" if "npcc" in x.lower() else ("L1" if "l1" in x.lower() else None))
df = df[df["LossFunction"].notna()]

# Parse output type from the model name
df["Output"] = df[filename_col].apply(lambda name: "greyscale" if "greyscale" in name.lower() else "binary")

# Compute Unified Evaluation Metric
df["UnifiedScore"] = (df["Average_SSIM"] + (1 - df["Average_LPIPS"])) / 2

# Plotting
metrics = [("Average_SSIM", "Average SSIM (Higher is Better)"), ("Average_LPIPS", "Average LPIPS (Lower is Better)"), ("UnifiedScore", "Unified Evaluation (Higher is Better)")]
grit_levels = sorted(df["GRIT"].unique())
output_types = ["binary", "greyscale"]
architectures = sorted(df["Architecture"].unique())

for metric, ylabel in metrics:
    for grit in grit_levels:
        for output in output_types:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"{ylabel} - GRIT {grit}, {output.capitalize()} Images", fontsize=16)

            for i, architecture in enumerate(architectures):
                subset = df[(df["GRIT"] == grit) & 
                            (df["Output"] == output) & 
                            (df["Architecture"] == architecture)]

                row, col = divmod(i, 2)
                ax = axes[row][col]

                if subset.empty:
                    ax.set_visible(False)
                    continue

                l1_vals = subset[subset["LossFunction"] == "L1"][metric]
                npcc_vals = subset[subset["LossFunction"] == "NPCC"][metric]

                data_to_plot = [l1_vals.mean(), npcc_vals.mean()]
                ax.bar(["L1", "NPCC"], data_to_plot, color=["skyblue", "salmon"])

                ax.set_title(f"{architecture}", fontsize=12)
                ax.set_ylabel(ylabel)
                ax.set_ylim(0, 1)  # Adjust based on your metric scale
                ax.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

