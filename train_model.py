#!/usr/bin/env python3
"""
Train a RandomForestRegressor pipeline on the provided contracts CSV.

Usage:
  python train_model.py --input data/Contact_spreadsheet_Sheet1.csv --output model.pkl

The script will auto-detect features if the default FEATURE_COLUMNS are missing from the CSV.
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 42

# Reasonable default features (keeps parity with main.py). If missing, script falls back to auto-detect.
FEATURE_COLUMNS = [
    "award_year","award_month","primary_state","geopolitical_zone",
    "latitude_start","longitude_start","estimated_length_km",
    "terrain_type","rainfall_mm_per_year","soil_type","elevation_m",
    "has_bridge","is_dual_carriageway","is_rehabilitation","is_coastal_or_swamp",
    "boq_earthworks_m3_per_km","boq_asphalt_ton_per_km","boq_drainage_km_per_km",
    "boq_bridges_units","boq_culverts_units","boq_premium_percent"
]

TARGET_COLUMN = "cost_ngn_billion"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    return df


def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = load_data(input_path)
    print(f"Loaded data with shape: {df.shape}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found. Found columns: {list(df.columns)}")

    # Determine feature columns
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"Warning: some FEATURE_COLUMNS missing: {missing}")
        feature_cols = [c for c in df.columns if c not in ("project_id", "project_name", TARGET_COLUMN)]
        print(f"Auto-detected feature columns: {feature_cols}")
    else:
        feature_cols = FEATURE_COLUMNS

    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].astype(float).values

    # split numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    print(f"Numeric cols ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical cols ({len(categorical_cols)}): {categorical_cols}")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ], remainder="drop")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=RANDOM_STATE)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"Evaluation: RMSE={rmse:.4f}, R2={r2:.4f}")

    joblib.dump(pipeline, output_path)
    print(f"Saved trained pipeline to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/Contact_spreadsheet_Sheet1.csv")
    parser.add_argument("--output", type=str, default="model.pkl")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
  if not os.path.exists("model.pkl"):
    subprocess.run(["python", "train_model.py"])
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTXlHZrU20uniUkjr-5Pis1pfJSOYDUiFVcML6UqW2Lu176_opvZPQvTGOpQZnNx02HyFf-jRYw3O8o/pub?output=csv"

try:
    df = pd.read_csv(GOOGLE_SHEET_URL)
    y = df['cost_ngn_billion']
    
    # Use only numeric columns that are guaranteed to exist
    numeric_cols = ['estimated_length_km', 'award_year', 'rainfall_mm_per_year', 'elevation_m']
    X = df[numeric_cols].copy()
    
    # Create pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=10, random_state=42))
    ])
    
    pipeline.fit(X, y)
    joblib.dump(pipeline, "model.pkl")
    print("✅ Model trained and saved successfully!")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    # Create dummy model
    from sklearn.dummy import DummyRegressor
    dummy_model = DummyRegressor(strategy='mean')
    dummy_model.fit(, [100])
    joblib.dump(dummy_model, "model.pkl")
    print("✅ Created dummy model as fallback")
