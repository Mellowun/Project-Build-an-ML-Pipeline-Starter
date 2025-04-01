import pandas as pd
import numpy as np
import wandb
import os
import json
import shutil
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
import argparse

# Add this at the beginning of main()
parser = argparse.ArgumentParser()
parser.add_argument("--max_depth", type=int, default=15)
parser.add_argument("--n_estimators", type=int, default=100)
args = parser.parse_args()

# Configure wandb
os.environ["WANDB_PROJECT"] = "nyc_airbnb"
os.environ["WANDB_RUN_GROUP"] = "development"

# Helper function for date features
def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()

def main():
    # Initialize wandb run
    run = wandb.init(job_type="train_random_forest")
    
    # Load RF config from the main config
    with open("rf_config.json") as fp:
        rf_config = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": 4,
            "min_samples_leaf": 3,
            "n_jobs": -1,
            "criterion": "squared_error",
            "max_features": 0.5,
            "oob_score": True,
            "random_state": 42  # This will be overwritten below
        }
    
    # Update wandb config
    params = {
        "trainval_artifact": "trainval_data.csv:latest",
        "val_size": 0.2,
        "random_seed": 42,
        "stratify_by": "neighbourhood_group",
        "max_tfidf_features": 5,
        "output_artifact": "random_forest_export"
    }
    run.config.update(params)
    run.config.update(rf_config)
    # Then update rf_config
    run.config.update({
        "max_depth": rf_config["max_depth"],
        "n_estimators": rf_config["n_estimators"]
    })
    
    # Fix random seed
    rf_config['random_state'] = params["random_seed"]
    
    # Get the training data
    trainval_artifact = run.use_artifact(params["trainval_artifact"])
    trainval_path = trainval_artifact.file()
    
    # Load and prepare the data
    X = pd.read_csv(trainval_path)
    y = X.pop("price")  # Remove price from X and save it as target
    
    print(f"Minimum price: {y.min()}, Maximum price: {y.max()}")
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=params["val_size"], 
        stratify=X[params["stratify_by"]], 
        random_state=params["random_seed"]
    )
    
    print("Preparing sklearn pipeline")
    
    # Build preprocessing pipeline
    # Let's handle the categorical features first
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    ordinal_categorical_preproc = OrdinalEncoder()
    
    # Build the non-ordinal categorical pipeline
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder()
    )
    
    # Numerical features
    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
    
    # Date features
    date_imputer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )
    
    # NLP features
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=False,
            max_features=params["max_tfidf_features"],
            stop_words='english'
        ),
    )
    
    # Combine everything into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop",  # Drop columns that we do not transform
    )
    
    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]
    
    # Create random forest
    random_forest = RandomForestRegressor(**rf_config)
    
    # Create the complete pipeline
    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest", random_forest)
        ]
    )
    
    # Train the model
    print("Fitting the model...")
    sk_pipe.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating the model...")
    r_squared = sk_pipe.score(X_val, y_val)
    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    
    print(f"R² Score: {r_squared}")
    print(f"MAE: {mae}")
    
    # Save the model
    print("Saving the model...")
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")
    
    # We'll use pickle instead of mlflow to avoid dependencies
    import pickle
    os.makedirs("random_forest_dir", exist_ok=True)
    with open("random_forest_dir/model.pkl", "wb") as f:
        pickle.dump(sk_pipe, f)
    
    # Create input example file
    with open("random_forest_dir/input_example.csv", "w") as f:
        X_train.iloc[:5].to_csv(f, index=False)
    
    # Upload the model to W&B
    artifact = wandb.Artifact(
        params["output_artifact"],
        type='model_export',
        description='Trained Random Forest model',
        metadata=rf_config
    )
    artifact.add_dir('random_forest_dir')
    run.log_artifact(artifact)
    
    # Plot feature importance
    feat_imp = random_forest.feature_importances_
    
    # Handle the NLP feature importance separately
    base_feat_imp = feat_imp[:len(processed_features)-1]
    nlp_importance = sum(feat_imp[len(processed_features)-1:])
    feat_imp = np.append(base_feat_imp, nlp_importance)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.bar(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    ax.set_xticks(range(feat_imp.shape[0]))
    ax.set_xticklabels(np.array(processed_features), rotation=90)
    fig.tight_layout()
    
    # Log metrics and visualizations to W&B
    wandb.log({"r2": r_squared, "mae": mae})
    
    run.log({
        "feature_importance": wandb.Image(fig)
    })
    
    print("Training completed successfully!")
    wandb.finish()

if __name__ == "__main__":
    main()
