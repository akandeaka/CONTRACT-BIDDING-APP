from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd, sqlite3, joblib, os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load dataset
df = pd.read_csv("contracts.csv").reset_index(drop=True)

# Load model
if not os.path.exists("model.pkl"):
    raise FileNotFoundError("⚠️ model.pkl not found. Run train_model.py first.")
model = joblib.load("model.pkl")

# SQLite setup
conn = sqlite3.connect("bids.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
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
""")
conn.commit()

def adjust_for_inflation(base_price, inflation_rate, years):
    return base_price * ((1 + inflation_rate) ** years)

@app.get("/contracts", response_class=HTMLResponse)
def contracts(request: Request):
    return templates.TemplateResponse("contracts.html", {"request": request, "contracts": df.to_dict(orient="records")})

@app.get("/contracts/{contract_id}", response_class=HTMLResponse)
def contract_detail(request: Request, contract_id: int):
    row = df.iloc[contract_id]
    return templates.TemplateResponse("contract_detail.html", {"request": request, "contract": row.to_dict(), "index": contract_id})

@app.post("/contracts/{contract_id}/submit_bid", response_class=HTMLResponse)
def submit_bid(contract_id: int, email: str = Form(...), phone: str = Form(...),
               bid_amount: float = Form(...), equipment_list: str = Form(...), workforce: str = Form(...)):

    row = df.iloc[contract_id]
    features = row[["award_year","award_month","latitude_start","longitude_start","estimated_length_km",
                    "terrain_type","rainfall_mm_per_year","soil_type","elevation_m",
                    "has_bridge","is_dual_carriageway","is_rehabilitation","is_coastal_or_swamp",
                    "boq_earthworks_m3_per_km","boq_asphalt_ton_per_km","boq_drainage_km_per_km",
                    "boq_bridges_units","boq_culverts_units","boq_premium_percent"]]

    base_price = model.predict([features])[0]
    adjusted_price = adjust_for_inflation(base_price, 0.12, 2)
    fair_min, fair_max = adjusted_price*0.9, adjusted_price*1.1

    if fair_min <= bid_amount <= fair_max:
        status, explanation = "Approved ✅", "Your bid has been accepted for review."
    else:
        status, explanation = "Rejected ❌", "Your bid did not meet evaluation criteria."

    cursor.execute("""
    INSERT INTO bids (contract_id, email, phone, bid_amount, equipment_list, workforce, status)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (contract_id, email, phone, bid_amount, equipment_list, workforce, status))
    conn.commit()

    return f"<h2>Bid Result</h2><p>Status: {status}</p><p>Explanation: {explanation}</p>"
