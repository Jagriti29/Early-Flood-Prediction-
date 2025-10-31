"""
Baseline Random Forest Regression Model
---------------------------------------
Trains a RandomForestRegressor on Model_Input.csv
Evaluates performance and saves model & results.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ======================
# Load Data
# ======================
DATA_PATH = "data/processed/model_ready/"
train_file = os.path.join(DATA_PATH, "train.csv")
val_file = os.path.join(DATA_PATH, "val.csv")
test_file = os.path.join(DATA_PATH, "test.csv")

train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)
test_df = pd.read_csv(test_file)

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")

# ======================
# Split Features & Target
# ======================
target_col = "Flood_Risk_Index"
feature_cols = [c for c in train_df.columns if c != target_col and c != "Districts"]

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_val = val_df[feature_cols]
y_val = val_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

# ======================
# Train Random Forest
# ======================
print("\nTraining Random Forest Regressor...")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ======================
# Evaluate Model
# ======================
def evaluate_model(model, X, y, name):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    print(f"{name} Performance:")
    print(f" MAE: {mae:.4f}")
    print(f" RMSE: {rmse:.4f}")
    print(f" R²: {r2:.4f}\n")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

train_metrics = evaluate_model(rf, X_train, y_train, "Train")
val_metrics = evaluate_model(rf, X_val, y_val, "Validation")
test_metrics = evaluate_model(rf, X_test, y_test, "Test")

# ======================
# Save Model and Results
# ======================
os.makedirs("models", exist_ok=True)
model_path = "models/random_forest_baseline.pkl"
joblib.dump(rf, model_path)

results = pd.DataFrame([train_metrics, val_metrics, test_metrics],
                       index=["Train", "Validation", "Test"])
os.makedirs("data/results", exist_ok=True)
results.to_csv("data/results/baseline_results.csv")

print(f"Model saved to: {model_path}")
print(f"Results saved to: data/results/baseline_results.csv")
print("\nBaseline training complete!")
