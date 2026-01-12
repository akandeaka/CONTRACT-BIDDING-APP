from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import sqlite3
import joblib
import os
import numpy as np
from typing import List

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ---------- Configuration ----------
df= pd.read_csv("https://docs.google.com/spreadsheets/d/.../pub?out=csv")
CONTRACTS_PATHS = [
    os.environ.get("CONTRACTS_PATH", ""),  # optional env override
    "contracts.csv",
    "contracts.xlsx",
    "Contact.csv.xlsx",
    "Contact.csv",
    "data/contracts.csv",
]
CONTRACTS_PATHS = [p for p in CONTRACTS_PATHS if p]  # drop empty

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
DB_PATH = os.environ.get("DB_PATH", "bids.db")

# ---------- Helper: load contracts ----------
def load_contracts(paths: List[str]) -> pd.DataFrame:
    last_exc = None
    for p in paths:
        if os.path.exists(p):
            try:
                if p.lower().endswith((".xls", ".xlsx")):
                    df = pd.read_excel(p)
                else:
                    df = pd.read_csv(p)
                df = df.reset_index(drop=True)
                return df
            except Exception as e:
                last_exc = e
    # No file found — return a minimal sample dataframe so the app can run for testing
    sample = pd.DataFrame([
        {
            "project_id": 0,
            "title": "Sample road A-B",
            "award_year": 2023,
            "award_month": 6,
            "primary_state": "State X",
            "geopolitical_zone": "Zone 1",
            "latitude_start": 0.0,
            "longitude_start": 0.0,
            "estimated_length_km": 10.0,
            "terrain_type": "plain",
            "rainfall_mm_per_year": 1000,
            "soil_type": "sandy",
            "elevation_m": 50,
            "has_bridge": 0,
            "is_dual_carriageway": 0,
            "is_rehabilitation": 0,
            "is_coastal_or_swamp": 0,
            "boq_earthworks_m3_per_km": 100.0,
            "boq_asphalt_ton_per_km": 10.0,
            "boq_drainage_km_per_km": 1.0,
            "boq_bridges_units": 0,
            "boq_culverts_units": 1,
            "boq_premium_percent": 5.0,
            "cost_ngn_billion": 1.2
        }
    ])
    print("Warning: no contracts file found. Using sample data. To load real data, add contracts.csv or set CONTRACTS_PATH env var.")
    return sample

df = load_contracts(CONTRACTS_PATHS)
# Use an 'index' as id if project_id not present
if "project_id" not in df.columns:
    df["project_id"] = df.index

# ---------- Helper: load model ----------
class DummyModel:
    # Simple fallback model which returns the mean cost if model.pkl missing
    def __init__(self, fallback_value: float = None):
        self.fallback = fallback_value

    def predict(self, X):
        # return a constant or fallback
        if self.fallback is not None:
            return np.array([self.fallback] * len(X))
        return np.array([1.0] * len(X))

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Warning: failed to load model from {MODEL_PATH}: {e}")
        model = None

if model is None:
    fallback_val = None
    if "cost_ngn_billion" in df.columns:
        try:
            fallback_val = float(df["cost_ngn_billion"].mean())
        except Exception:
            fallback_val = 1.0
    else:
        fallback_val = 1.0
    print(f"Info: model.pkl not found or failed to load. Using DummyModel with fallback {fallback_val}")
    model = DummyModel(fallback_value=fallback_val)

# ---------- SQLite (bids) ----------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS bids (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        contract_id INTEGER,
        email TEXT,
        phone TEXT,
        bid_amount REAL,
        equipment_list TEXT,
        workforce TEXT,
        status TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
)
conn.commit()

# ---------- Utility ----------
FEATURE_COLUMNS = [
    "award_year","award_month","primary_state","geopolitical_zone",
    "latitude_start","longitude_start","estimated_length_km",
    "terrain_type","rainfall_mm_per_year","soil_type","elevation_m",
    "has_bridge","is_dual_carriageway","is_rehabilitation","is_coastal_or_swamp",
    "boq_earthworks_m3_per_km","boq_asphalt_ton_per_km","boq_drainage_km_per_km",
    "boq_bridges_units","boq_culverts_units","boq_premium_percent"
]

def prepare_features(row: pd.Series) -> np.ndarray:
    # If some expected features are missing, fill with zeros or reasonable defaults
    values = []
    for col in FEATURE_COLUMNS:
        if col in row:
            val = row[col]
            # Convert booleans to integers and handle nan
            if pd.isna(val):
                val = 0
            # For categorical values, attempt to encode simply:
            if isinstance(val, str):
                # A trivial hash-to-number encoding to get numeric input for fallback model
                val = float(abs(hash(val)) % 1000)  # simple deterministic encoding
            values.append(float(val))
        else:
            values.append(0.0)
    return np.array(values).reshape(1, -1)

def adjust_for_inflation(base_price, inflation_rate, years):
    try:
        return float(base_price) * ((1 + float(inflation_rate)) ** float(years))
    except Exception:
        return float(base_price)

# ---------- Routes ----------
@app.get("/contracts", response_class=HTMLResponse)
def contracts(request: Request):
    records = df.to_dict(orient="records")
    return templates.TemplateResponse("contracts.html", {"request": request, "contracts": records})

@app.get("/contracts/{contract_id}", response_class=HTMLResponse)
def contract_detail(request: Request, contract_id: int):
    # Try to locate by index first, else by project_id
    if 0 <= contract_id < len(df):
        row = df.iloc[contract_id]
    else:
        matches = df[df.get("project_id") == contract_id]
        if len(matches) == 0:
            raise HTTPException(status_code=404, detail="Contract not found")
        row = matches.iloc[0]
    return templates.TemplateResponse("contract_detail.html", {"request": request, "contract": row.to_dict()})

@app.post("/contracts/{contract_id}/submit_bid", response_class=HTMLResponse)
def submit_bid(contract_id: int, email: str = Form(...), phone: str = Form(...),
               bid_amount: float = Form(...), equipment_list: str = Form(""),
               workforce: str = Form("")):
    # locate row
    if 0 <= contract_id < len(df):
        row = df.iloc[contract_id]
    else:
        matches = df[df.get("project_id") == contract_id]
        if len(matches) == 0:
            raise HTTPException(status_code=404, detail="Contract not found")
        row = matches.iloc[0]

    # prepare features for model
    X = prepare_features(row)
    try:
        base_price = model.predict(X)[0]
    except Exception as e:
        # If model fails, fallback gracefully
        print(f"Model prediction error: {e}")
        base_price = getattr(model, "fallback", 1.0)

    adjusted_price = adjust_for_inflation(base_price, 0.12, 2)
    fair_min, fair_max = adjusted_price * 0.9, adjusted_price * 1.1

    if fair_min <= bid_amount <= fair_max:
        status, explanation = "Approved ✅", "Your bid has been accepted for review."
    else:
        status, explanation = "Rejected ❌", "Your bid did not meet evaluation criteria."

    cursor.execute(
        """
        INSERT INTO bids (contract_id, email, phone, bid_amount, equipment_list, workforce, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (contract_id, email, phone, bid_amount, equipment_list, workforce, status)
    )
    conn.commit()

    return f"<h2>Bid Result</h2><p>Status: {status}</p><p>Explanation: {explanation}</p>"

# Optional: simple health check
@app.get("/health")
def health():
    return {"status": "ok", "contracts": len(df)}
