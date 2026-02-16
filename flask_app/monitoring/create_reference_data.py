import pandas as pd

# CHANGE THIS path to your actual training dataset
DATASET_PATH = "D:\Projects\MLOps-Project\Real-Time-YouTube-Comment-Analysis\dataset\Data_Cleaning_YT.csv"# example
TEXT_COLUMN = "Comment"                      # example

OUTPUT_PATH = "monitoring/reference_data.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Keep only text
df = df[[TEXT_COLUMN]].dropna()

# Sample to reduce size (adjust if you want)
df_sample = df.sample(n=min(5000, len(df)), random_state=42)

# Save reference data
df_sample.rename(columns={TEXT_COLUMN: "text"}).to_csv(
    OUTPUT_PATH, index=False
)

print("✅ Reference data saved to", OUTPUT_PATH)
