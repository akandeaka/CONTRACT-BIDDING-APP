

### ✅ `train_model.py` (Cloud-Safe Version)
```python
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Google Sheets CSV URL (same as in main.py)
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTXlHZrU20uniUkjr-5Pis1pfJSOYDUiFVcML6UqW2Lu176_opvZPQvTGOpQZnNx02HyFf-jRYw3O8o/pub?output=csv"

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

---

