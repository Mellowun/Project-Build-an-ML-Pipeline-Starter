import pandas as pd
import numpy as np
import wandb
import os
import pickle

# Configure wandb
os.environ["WANDB_PROJECT"] = "nyc_airbnb"
os.environ["WANDB_RUN_GROUP"] = "development"

def main():
    # Initialize wandb run
    run = wandb.init(job_type="test_random_forest")

    # Download the model artifact
    model_artifact = run.use_artifact("random_forest_export:prod")
    model_path = model_artifact.download()

    # Load the model
    with open(os.path.join(model_path, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    # Download the test data
    test_artifact = run.use_artifact("test_data.csv:latest")
    test_path = test_artifact.file()

    # Load test data
    X_test = pd.read_csv(test_path)
    y_test = X_test.pop("price")  # Remove price from X and save it as target

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_pred - y_test))
    r2 = model.score(X_test, y_test)

    # Log metrics
    run.summary['test_mae'] = mae
    run.summary['test_r2'] = r2

    print(f"Test MAE: {mae}")
    print(f"Test R2: {r2}")

    wandb.finish()

if __name__ == "__main__":
    main()