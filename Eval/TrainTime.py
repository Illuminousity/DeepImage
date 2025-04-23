import pandas as pd
import glob
import os

# Path to the folder containing CSV files
csv_folder = './csv'

# Initialize total training time
total_training_time = 0

# Iterate through each CSV file in the folder
for csv_file in glob.glob(os.path.join(csv_folder, '*.csv')):
    df = pd.read_csv(csv_file)
    # Sum the 'Train Time (s)' column and add to the total training time
    total_training_time += df['Train Time (s)'].sum()

# Convert total training time from seconds to hours and minutes
total_hours = total_training_time // 3600
total_minutes = (total_training_time % 3600) // 60

print(f"Total Training Time: {total_hours} hours and {total_minutes} minutes ({total_training_time} seconds)")
