import pandas as pd
import wandb
import os
from sklearn.model_selection import train_test_split

# Set up wandb
os.environ["WANDB_PROJECT"] = "nyc_airbnb"
os.environ["WANDB_RUN_GROUP"] = "development"
wandb.init()

# Get the data
artifact = wandb.use_artifact("clean_sample.csv:latest")
artifact_path = artifact.file()
df = pd.read_csv(artifact_path)

# Split the data
stratify = None
if "neighbourhood_group" in df.columns:
    stratify = df["neighbourhood_group"]

train_val, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=stratify
)

# Save locally
train_val.to_csv("trainval_data.csv", index=False)
test.to_csv("test_data.csv", index=False)

# Create and upload artifacts
trainval_artifact = wandb.Artifact("trainval_data.csv", type="TRAINVAL_DATA")
trainval_artifact.add_file("trainval_data.csv")
wandb.log_artifact(trainval_artifact)

test_artifact = wandb.Artifact("test_data.csv", type="TEST_DATA")
test_artifact.add_file("test_data.csv")
wandb.log_artifact(test_artifact)

print("Data split completed successfully!")
print(f"trainval_data.csv created with {len(train_val)} rows")
print(f"test_data.csv created with {len(test)} rows")

wandb.finish()
