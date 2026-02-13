# main.py
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Generator

import jwt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import joblib

from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from passlib.context import CryptContext
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-jwt-key-change-this-in-production-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# ────────────────────────────────────────────────
# FastAPI + Rate Limiter
# ────────────────────────────────────────────────
app = FastAPI(title="AISEC – Contract Bidding Platform")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    lambda req, exc: HTMLResponse("<h2>Too many attempts. Try again later.</h2>", status_code=429)
)

# ────────────────────────────────────────────────
# Password Hashing & JWT
# ────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": subject, "exp": expire}, JWT_SECRET, algorithm=ALGORITHM)

def get_current_user_id(request: Request) -> int:
    token = request.cookies.get("session_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def get_current_admin_id(request: Request) -> int:
    token = request.cookies.get("admin_token")
    if not token:
        raise HTTPException(status_code=401, detail="Admin login required")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except:
        raise HTTPException(status_code=401, detail="Invalid admin token")

# ────────────────────────────────────────────────
# Database
# ────────────────────────────────────────────────
def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect("bids.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with sqlite3.connect("bids.db") as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            company_name TEXT NOT NULL,
            cac_number TEXT NOT NULL
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS bids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            company_name TEXT NOT NULL,
            cac_number TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            bid_amount REAL NOT NULL,
            equipment_list TEXT NOT NULL,
            workforce TEXT NOT NULL,
            status TEXT NOT NULL,
            predicted_min REAL,
            predicted_max REAL,
            submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        )''')

        # Use short password via env var (avoids bcrypt 72-byte limit)
        admin_password = os.getenv("ADMIN_PASSWORD", "AISEC2026!")
        if len(admin_password.encode('utf-8')) > 72:
            admin_password = admin_password[:72]
            print("Warning: ADMIN_PASSWORD truncated to 72 bytes")

        try:
            c.execute("INSERT INTO admins (username, hashed_password) VALUES (?, ?)",
                      ("admin", pwd_context.hash(admin_password)))
            conn.commit()
            print("Admin user created/updated successfully")
        except sqlite3.IntegrityError:
            print("Admin user already exists - skipping creation")
init_db()

# ────────────────────────────────────────────────
# Load contracts from Google Sheet
# ────────────────────────────────────────────────
BIDDING_CONTRACTS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS-nWpM2oCQ5xmda7a3tlLiRmMC2VaAdG4IhoQsypuVvbYDgtDaWn_bYcClrc35XUoHRvvMEISXTvCw/pub?output=csv"

try:
    df_bidding = pd.read_csv(BIDDING_CONTRACTS_URL).reset_index(drop=True)
    print("Loaded sheet with", len(df_bidding), "contracts")
    print("Columns:", list(df_bidding.columns))
except Exception as e:
    print(f"Failed to load Google Sheet: {e}")
    df_bidding = pd.DataFrame()

# ────────────────────────────────────────────────
# Train / Load XGBoost Model
# ────────────────────────────────────────────────
MODEL_FILE = "ai_contract_model.joblib"

def train_model():
    global model
    print("Training XGBoost model...")

    df = df_bidding.copy()

    # Convert categorical to numeric
    categorical_cols = ["terrain_type", "geopolitical_zone"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    # Features (only those that exist in your sheet)
    possible_features = [
        "estimated_length_km", "terrain_type", "latitude", "longitude",
        "rainfall_mm_per_year", "elevation_m", "has_bridge", "is_dual_carriageway"
    ]
    features = [f for f in possible_features if f in df.columns]

    if not features:
        print("ERROR: No valid features found in sheet. Using dummy model.")
        model = xgb.XGBRegressor()  # dummy
        return model

    # Target: use proxy if no real cost
    target_col = "boq_total_cost"
    if target_col not in df.columns:
        print("No real cost column → using proxy target (length * 1.2B)")
        df[target_col] = df["estimated_length_km"] * 1_200_000_000

    X = df[features].fillna(0)
    y = df[target_col]

    if len(X) < 10:
        print("WARNING: Too few rows to train. Using dummy model.")
        model = xgb.XGBRegressor()
        return model

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    print(f"Model trained → MAPE: {mape:.2f}%")

    joblib.dump(model, MODEL_FILE)
    return model

# Load or train
try:
    model = joblib.load(MODEL_FILE)
    print("Loaded existing XGBoost model")
except FileNotFoundError:
    model = train_model()

# ────────────────────────────────────────────────
# Real AI Prediction (robust numeric conversion)
# ────────────────────────────────────────────────
def is_fair_bid(contract_id: int, bid_amount: float) -> tuple:
    if contract_id >= len(df_bidding):
        return "Under Review", 0, 0

    row = df_bidding.iloc[contract_id]

    # Safe numeric input
    input_dict = {
        "estimated_length_km": float(row.get("estimated_length_km", 100)),
        "latitude": float(row.get("latitude", 0)),
        "longitude": float(row.get("longitude", 0)),
        "rainfall_mm_per_year": float(row.get("rainfall_mm_per_year", 800)),
        "elevation_m": float(row.get("elevation_m", 300)),
        "has_bridge": 1.0 if str(row.get("has_bridge", "No")).lower() in ['yes', '1', 'true'] else 0.0,
        "is_dual_carriageway": 1.0 if str(row.get("is_dual_carriageway", "No")).lower() in ['yes', '1', 'true'] else 0.0,
        "terrain_type": float({
            "arid savanna": 0,
            "semi-arid flat": 1,
            "gently rolling savanna": 2,
            "flat arid": 3,
            "arid savanna dunes": 4,
            "savanna plains/hills": 5,
            "hilly savanna": 6,
            "hilly rocky": 7,
            "hilly forested": 8,
            "tropical rainforest": 9,
            "coastal sandy": 10,
            "mangrove swamp": 11,
            "riverine floodplain": 12,
            "delta swamp": 13,
            "hilly erosion-prone": 14,
            "undulating rainforest": 15,
            "rainforest valleys": 16,
            "coastal mangrove": 17,
            "rainforest hills": 18,
        }.get(str(row.get("terrain_type", "semi-arid flat")).lower().strip(), 1)),
    }

    input_df = pd.DataFrame([input_dict])

    # Force numeric
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    predicted_value = model.predict(input_df)[0]

    min_fair = predicted_value * 0.88
    max_fair = predicted_value * 1.12

    status = "Fair" if min_fair <= bid_amount <= max_fair else "Under Review"

    return status, round(min_fair / 1e9, 2), round(max_fair / 1e9, 2)

# ────────────────────────────────────────────────
# Routes (only dashboard part shown – keep your other routes)
# ────────────────────────────────────────────────

# ... your register, login, contracts, bid routes ...

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: sqlite3.Connection = Depends(get_db)):
    get_current_admin_id(request)

    cursor = db.cursor()
    cursor.execute("""
        SELECT id, contract_id, company_name, bid_amount, status, submitted_at,
               predicted_min, predicted_max
        FROM bids
        ORDER BY submitted_at DESC
    """)
    bids = cursor.fetchall()

    rows = ""
    for b in bids:
        contract_id = b["contract_id"]
        project_name = df_bidding.iloc[contract_id].get("Project_name", f"Contract {contract_id}")
        min_fair = b["predicted_min"] if b["predicted_min"] is not None else "N/A"
        max_fair = b["predicted_max"] if b["predicted_max"] is not None else "N/A"
        status_color = "#10b981" if b["status"] == "Approved" else "#ef4444" if b["status"] == "Rejected" else "#f59e0b"

        rows += f"""
        <tr>
            <td>#{b['id']}</td>
            <td>{project_name}</td>
            <td>{b['company_name']}</td>
            <td>₦{b['bid_amount']:,.2f}B</td>
            <td>₦{min_fair}B – ₦{max_fair}B</td>
            <td style="color:{status_color};">{b['status']}</td>
            <td>{b['submitted_at'][:10]}</td>
            <td>
                <button onclick="openReviewModal({b['id']}, '{project_name.replace("'", "\\'")}', {b['bid_amount']}, {min_fair}, {max_fair})"
                        style="background:#3b82f6;color:white;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;">
                    Review Bid
                </button>
            </td>
        </tr>
        """

    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AISEC Admin Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #f8fafc; margin: 0; padding: 20px; }}
            h1 {{ color: #1e40af; text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            th, td {{ padding: 14px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
            th {{ background: #1e40af; color: white; }}
            button {{ padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; color: white; }}
        </style>
    </head>
    <body>
        <h1>Admin Dashboard – All Bids</h1>
        <a href="/admin/logout" style="float:right;color:#ef4444;">Logout</a>
        <table>
            <tr><th>ID</th><th>Project</th><th>Company</th><th>Bid Amount</th><th>AI Fair Range</th><th>Status</th><th>Date</th><th>Action</th></tr>
            {rows}
        </table>
    </body>
    </html>
    """)

# ... your other admin routes ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


