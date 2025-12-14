import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("contracts.csv").reset_index(drop=True)

features = [
    "award_year","award_month","latitude_start","longitude_start","estimated_length_km",
    "terrain_type","rainfall_mm_per_year","soil_type","elevation_m",
    "has_bridge","is_dual_carriageway","is_rehabilitation","is_coastal_or_swamp",
    "boq_earthworks_m3_per_km","boq_asphalt_ton_per_km","boq_drainage_km_per_km",
    "boq_bridges_units","boq_culverts_units","boq_premium_percent"
]

X = df[features]
y = df["contract_price"]   # target column

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")
