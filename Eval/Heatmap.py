import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from glob import glob

# Re-filter to include only L1 models so we can analyze across dataset sizes
eval_df = pd.read_csv("./csveval/evaluation_results_new.csv")
filename_col = "ModelPath" if "ModelPath" in eval_df.columns else "ModelName"

# Extract dataset size
def extract_dataset_size(name):
    match = re.search(r"_(\d{5})_", name)
    return int(match.group(1)) if match else None

eval_df["DatasetSize"] = eval_df[filename_col].apply(extract_dataset_size)
# Only include L1 models
eval_df["LossFunction"] = eval_df[filename_col].apply(lambda x: "L1" if "l1" in x.lower() else "NPCC")
eval_df = eval_df[eval_df["LossFunction"] == "L1"]

# Additional parsing
eval_df["Architecture"] = eval_df[filename_col].apply(lambda x: "_".join(x.split("_")[:2]).lower())
eval_df["Output"] = eval_df[filename_col].apply(lambda x: "greyscale" if "greyscale" in x.lower() else "binary")
eval_df["UnifiedScore"] = (eval_df["Average_SSIM"] + (1 - eval_df["Average_LPIPS"])) / 2

# Load runtime files again
runtime_data = []
runtime_files = glob("./csv/*.csv")
for file in runtime_files:
    df = pd.read_csv(file)
    total_time = df["Time"].sum() if "Time" in df.columns else df.iloc[:, -1].sum()
    base_name = os.path.basename(file).replace(".csv", "")
    runtime_data.append({"Model": base_name, "TotalTime": total_time})

runtime_df = pd.DataFrame(runtime_data)

# Match runtime with evaluation
def match_runtime(model_name):
    for runtime_model in runtime_df["Model"]:
        if model_name.startswith(runtime_model):
            return runtime_model
    return None

eval_df["RuntimeKey"] = eval_df[filename_col].apply(lambda x: match_runtime(os.path.basename(x).replace(".pth", "")))
merged_df = pd.merge(eval_df, runtime_df, left_on="RuntimeKey", right_on="Model", how="left")

# Select relevant numeric columns

figsize = (16, 12)
cols_to_check = ["DatasetSize", "Average_SSIM", "Average_LPIPS", "TotalTime"]
grit_levels = sorted(merged_df["GRIT"].dropna().unique())
outputs = sorted(merged_df["Output"].dropna().unique())

for output in outputs:
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Correlation Heatmaps - {output.capitalize()}', fontsize=16)

    for i, grit in enumerate(grit_levels):
        subset = merged_df[(merged_df["Output"] == output) & (merged_df["GRIT"] == grit)]
        corr_subset = subset[cols_to_check].dropna()

        row, col = divmod(i, 2)
        ax = axes[row][col]

        if corr_subset.empty:
            ax.set_visible(False)
            continue

        corr_matrix = corr_subset.corr(method='spearman')
        heatmap = sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0.5, ax=ax, cbar=True, cbar_kws={"shrink": 0.7}
        )

        ax.set_title(f'GRIT {grit}', fontsize=14, pad=12)
        ax.tick_params(axis='x', labelrotation=45)
        ax.tick_params(axis='y', labelrotation=0)

    # Adjust layout: more space at top for suptitle, space for axis labels
    plt.subplots_adjust(wspace=0.3, hspace=1, top=0.88, bottom=0.1)
    plt.show()
