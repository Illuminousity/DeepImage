import glob
import os
import pandas as pd
import re
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load your CSV
df = pd.read_csv("./csveval/evaluation_results_new.csv")
filename_col = "ModelPath" if "ModelPath" in df.columns else "ModelName"

# --- Extract features from ModelName ---

def extract_dataset_size(name):
    match = re.search(r'_(\d{5})_', name)
    return int(match.group(1)) if match else None

def extract_image_type(name):
    return "greyscale" if "greyscale" in name.lower() else "binary"
def extract_loss_function(name):
    if "l1" in name.lower():
        return "L1"
    elif "npcc" in name.lower():
        return "NPCC"
    else:
        None

df["Dataset_Size"] = df["ModelName"].apply(extract_dataset_size)
df["Image_Type"] = df["ModelName"].apply(extract_image_type)
df["LossFunction"] = df["ModelName"].apply(extract_loss_function)


# Load runtime files again
runtime_data = []
runtime_files = glob.glob("./csv/*.csv")
for file in runtime_files:
    temp_df = pd.read_csv(file)
    total_time = temp_df["Time"].sum() if "Time" in temp_df.columns else temp_df.iloc[:, -1].sum()
    base_name = os.path.basename(file).replace(".csv", "")
    runtime_data.append({"Model": base_name, "TotalTime": total_time})

runtime_df = pd.DataFrame(runtime_data)

#Match runtime with evaluation
def match_runtime(model_name):
    for runtime_model in runtime_df["Model"]:
        if model_name.startswith(runtime_model):
            return runtime_model
    return None

df["RuntimeKey"] = df[filename_col].apply(lambda x: match_runtime(os.path.basename(x).replace(".pth", "")))
merged_df = pd.merge(df, runtime_df, left_on="RuntimeKey", right_on="Model", how="left")



# --- ANOVA for SSIM ---
ssim_model = ols(
    "Average_SSIM ~ C(Architecture) + C(Dataset_Size) + C(Image_Type) + C(TotalTime) + C(LossFunction) + C(GRIT)  +"
    "C(Architecture):C(GRIT) + C(Architecture):C(Image_Type) + C(Architecture):C(LossFunction) + C(Architecture):C(LossFunction):C(Image_Type) +"
    "C(Architecture):C(Image_Type):C(Dataset_Size) + C(Architecture):C(Image_Type):C(GRIT)", 
    data=merged_df
).fit()
ssim_anova = sm.stats.anova_lm(ssim_model, typ=2)

# --- ANOVA for LPIPS ---

lpips_model = ols(
    "Average_LPIPS ~ C(Architecture) + C(Dataset_Size) + C(Image_Type) + C(TotalTime) + C(LossFunction) + C(GRIT)  +"
    "C(Architecture):C(GRIT) + C(Architecture):C(Image_Type) + C(Architecture):C(LossFunction) + C(Architecture):C(LossFunction):C(Image_Type) +"
    "C(Architecture):C(Image_Type):C(Dataset_Size) + C(Architecture):C(Image_Type):C(GRIT)", 
    data=merged_df
).fit()
lpips_anova = sm.stats.anova_lm(lpips_model, typ=2)

# --- Print results ---
print("SSIM ANOVA Results:\n", ssim_anova)
print("\nLPIPS ANOVA Results:\n", lpips_anova)
