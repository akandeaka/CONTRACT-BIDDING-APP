
```python
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import sqlite3
import joblib
import os
import hashlib
import subprocess
import sys

app = FastAPI()
templates = Jinja2Templates(directory="templates")

GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTXlHZrU20uniUkjr-5Pis1pfJSOYDUiFVcML6UqW2Lu176_opvZPQvTGOpQZnNx02HyFf-jRYw3O8o/pub?output=csv"
MODEL_PATH = "model.pkl"

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print("Training model...")
        result = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Model training failed:\n{result.stderr}")

ensure_model_exists()

df = pd.read_csv(GOOGLE_SHEET_URL).reset_index(drop=True)
model = joblib.load(MODEL_PATH)

conn = sqlite3.connect("bids.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS bids (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id INTEGER,
    user_id INTEGER NOT NULL,
    email TEXT,
    phone TEXT,
    bid_amount REAL,
    equipment_list TEXT,
    workforce TEXT,
    status TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
)
""")
conn.commit()

def adjust_for_inflation(base_price, inflation_rate=0.12, years=2):
    return base_price * ((1 + inflation_rate) ** years)

@app.post("/register", response_class=HTMLResponse)
async def register_user(email: str = Form(...), password: str = Form(...)):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    try:
        cursor.execute("INSERT INTO users (email, hashed_password) VALUES (?, ?)", (email, hashed))
        conn.commit()
        return "<h1>✅ Registration Successful!</h1><p><a href='/login'>Login here</a></p>"
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")

@app.post("/login", response_class=HTMLResponse)
async def login_user(email: str = Form(...), password: str = Form(...)):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute("SELECT id FROM users WHERE email = ? AND hashed_password = ?", (email, hashed))
    user = cursor.fetchone()
    if user:
        return f"<h1>✅ Login Successful!</h1><p>User ID: {user[0]}</p><p><a href='/contracts'>View Contracts</a></p>"
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/contracts", response_class=HTMLResponse)
def contracts(request: Request):
    return templates.TemplateResponse("contracts.html", {"request": request, "contracts": df.to_dict(orient="records")})

@app.get("/contracts/{contract_id}", response_class=HTMLResponse)
def contract_detail(request: Request, contract_id: int):
    row = df.iloc[contract_id]
    return templates.TemplateResponse("contract_detail.html", {"request": request, "contract": row.to_dict()})

@app.post("/contracts/{contract_id}/submit_bid", response_class=HTMLResponse)
async def submit_bid(
    contract_id: int,
    user_id: int = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    bid_amount: float = Form(...),
    equipment_list: str = Form(...),
    workforce: str = Form(...),
):
    row = df.iloc[contract_id]
    features = row[[
        "award_year", "award_month", "primary_state", "geopolitical_zone",
        "latitude_start", "longitude_start", "estimated_length_km",
        "terrain_type", "rainfall_mm_per_year", "soil_type", "elevation_m",
        "has_bridge", "is_dual_carriageway", "is_rehabilitation", "is_coastal_or_swamp",
        "boq_earthworks_m3_per_km", "boq_asphalt_ton_per_km", "boq_drainage_km_per_km",
        "boq_bridges_units", "boq_culverts_units", "boq_premium_percent"
    ]]

    features_df = pd.DataFrame([features.values], columns=features.index)
    base_price = model.predict(features_df)[0]
    adjusted = adjust_for_inflation(base_price)
    fair_min, fair_max = adjusted * 0.9, adjusted * 1.1

    if fair_min <= bid_amount <= fair_max:
        status_msg, explanation = "Approved ✅", "Your bid has been accepted for review."
    else:
        status_msg, explanation = "Rejected ❌", "Your bid did not meet evaluation criteria."

    cursor.execute("""
    INSERT INTO bids (contract_id, user_id, email, phone, bid_amount, equipment_list, workforce, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (contract_id, user_id, email, phone, bid_amount, equipment_list, workforce, status_msg))
    conn.commit()

    return f"<h2>Bid Result</h2><p>Status: {status_msg}</p><p>Explanation: {explanation}</p>"
```

---

