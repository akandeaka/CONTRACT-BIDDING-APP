
```python
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1Mf2ktSMN7dknEb3hhAxA0c0qHd1zV-Jh/edit?gid=1060047915#gid=1060047915:~:text=https%3A//docs.google.com/spreadsheets/d/e/2PACX%2D1vQT5%2DBoqR1dHJ4q0uTkm4W1GAN3lcINVNmCKA9XNIaIN%2Dns6_LSBEbkHHeBMkV7kQ/pub%3Foutput%3Dcsv"

print("Starting model training...")
df = pd.read_csv(GOOGLE_SHEET_URL)
y = df['cost_ngn_billion']

feature_columns = [
    "award_year", "award_month", "primary_state", "geopolitical_zone",
    "latitude_start", "longitude_start", "estimated_length_km",
    "terrain_type", "rainfall_mm_per_year", "soil_type", "elevation_m",
    "has_bridge", "is_dual_carriageway", "is_rehabilitation", "is_coastal_or_swamp",
    "boq_earthworks_m3_per_km", "boq_asphalt_ton_per_km", "boq_drainage_km_per_km",
    "boq_bridges_units", "boq_culverts_units", "boq_premium_percent"
]
X = df[feature_columns].copy()

categorical_features = X.select_dtypes(include=['object', 'bool']).columns
numerical_features = X.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X, y)
joblib.dump(pipeline, "model.pkl")
print("✅ Model trained and saved!")
```

## ✅ **Additional Recommendations:**

1. **Add error handling** for robustness:
```python
try:
    df = pd.read_csv(GOOGLE_SHEET_URL)
    print(f"✅ Loaded {len(df)} rows from Google Sheets")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    # Create dummy data for testing
    df = create_dummy_data()
```

