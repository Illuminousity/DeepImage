import os
import re
import matplotlib.pyplot as plt
import pandas as pd

# Path to the evaluation CSV file
csv_file = "./csveval/evaluation_results.csv"

# Check if the CSV file exists
if not os.path.exists(csv_file):
    print(f"Error: file {csv_file} does not exist.")
    exit(1)

# Read the CSV file into a DataFrame.
df = pd.read_csv(csv_file)

# Expected required columns besides "DatasetSize" (which will be extracted):
# We assume CSV should have:
# "GRIT" (numeric), "Output" (string, e.g., "Binary" or "Greyscale"),
# "Average_SSIM" (numeric), "Average_LPIPS" (numeric) and a column containing the model filename.
# We look first for a "Filename" column, and if not found, try "Model".
filename_col = None
if "ModelPath" in df.columns:
    filename_col = "ModelPath"
elif "ModelName" in df.columns:
    filename_col = "ModelName"
else:
    print("Error: CSV file must contain a 'Filename' or 'Model' column to extract the dataset size.")
    exit(1)

df = df[df[filename_col].str.lower().str.contains("l1")]


# If "DatasetSize" column is missing, create it by extracting from the filename.
if "DatasetSize" not in df.columns:
    # Define a helper function to extract the dataset size from the filename.
    def extract_dataset_size(name):
        # The expected pattern: architecture_GRIT_datasetSize_...
        # For example: effnet_rednet_120_10000_l1_greyscale.pth
        pattern = r"(.+?)_(\d+)_(\d+).*\.pth"
        match = re.match(pattern, name)
        if match:
            try:
                return int(match.group(3))
            except ValueError:
                return None
        return None
    df["DatasetSize"] = df[filename_col].apply(extract_dataset_size)

if "Output" not in df.columns:
    # Check for the presence of "greyscale" in the filename (case-insensitive).
    df["Output"] = df[filename_col].apply(lambda name: "greyscale" if "greyscale" in name.lower() else "binary")

# Verify that all required columns now exist.
required_columns = ["GRIT", "Average_SSIM", "Average_LPIPS", "DatasetSize", "Output"]
for col in required_columns:
    if col not in df.columns:
        print(f"Error: Required column '{col}' not found in CSV.")
        exit(1)

# Convert numeric columns.
df["GRIT"] = pd.to_numeric(df["GRIT"], errors='coerce')
df["DatasetSize"] = pd.to_numeric(df["DatasetSize"], errors='coerce')
df["Average_SSIM"] = pd.to_numeric(df["Average_SSIM"], errors='coerce')
df["Average_LPIPS"] = pd.to_numeric(df["Average_LPIPS"], errors='coerce')

# Compute the unified guideline score: (Average_SSIM + (1 - Average_LPIPS)) / 2.
df["UnifiedScore"] = (df["Average_SSIM"] + (1 - df["Average_LPIPS"])) / 2

# Get unique GRIT variants and output types (e.g., "Binary" or "Greyscale")
grit_variants = df["GRIT"].unique()
output_types = df["Output"].unique()

# Check if an "Architecture" column exists.
has_architecture = "Architecture" in df.columns
if has_architecture:
    architectures = df["Architecture"].unique()

# Define the metrics (columns) and associated y-axis labels for plotting.
metrics = [
    ("Average_SSIM", "Average SSIM Performance (Higher is Better)"),
    ("Average_LPIPS", "Average LPIPS Score (Lower is Better)"),
    ("UnifiedScore", "Unified Guideline Score (Higher is Better)")
]

# Generate a plot for each GRIT variant and each output type.
for grit in sorted(grit_variants):
    for output in sorted(output_types):
        # Filter the DataFrame for the current GRIT variant and output type.
        df_group = df[(df["GRIT"] == grit) & (df["Output"] == output)]
        if df_group.empty:
            print(f"No data for GRIT {grit} with output {output}. Skipping...")
            continue

        # Create a plot for each metric.
        for metric, ylabel in metrics:
            plt.figure(figsize=(6, 4))
            if has_architecture:
                # Generate a separate curve per architecture.
                for arch in sorted(architectures):
                    df_subset = df_group[df_group["Architecture"] == arch]
                    if df_subset.empty:
                        continue
                    # Sort by dataset size for an orderly progression.
                    df_subset = df_subset.sort_values("DatasetSize")
                    plt.plot(df_subset["DatasetSize"], df_subset[metric], marker='o', label=str(arch))
                plt.legend()
            else:
                # Plot all data for this group as a single line.
                df_group_sorted = df_group.sort_values("DatasetSize")
                plt.plot(df_group_sorted["DatasetSize"], df_group_sorted[metric], marker='o')
            plt.title(f"GRIT = {grit} | Output = {output} | {ylabel} vs. Dataset Size")
            plt.xlabel("Dataset Size")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.tight_layout()
            plt.show()