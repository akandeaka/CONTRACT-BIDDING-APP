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

        try:
            c.execute("INSERT INTO admins (username, hashed_password) VALUES (?, ?)",
                      ("admin", pwd_context.hash("AdminSecure2026!")))
            conn.commit()
        except sqlite3.IntegrityError:
            pass

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
# Train / Load XGBoost Model (real prediction)
# ────────────────────────────────────────────────
MODEL_FILE = "ai_contract_model.joblib"

def train_model():
    global model
    print("Training XGBoost model...")

    df = df_bidding.copy()

    # Convert categorical to numeric
    for col in ["terrain_type", "geopolitical_zone"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    features = [
        "estimated_length_km", "terrain_type", "latitude", "longitude",
        "rainfall_mm_per_year", "elevation_m", "has_bridge", "is_dual_carriageway"
    ]

    # Target: use proxy if no real cost column
    if "boq_total_cost" not in df.columns:
        print("No real cost column → using proxy target")
        df["boq_total_cost"] = df["estimated_length_km"] * 1_200_000_000

    X = df[features].fillna(0)
    y = df["boq_total_cost"]

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

# Load or train model
try:
    model = joblib.load(MODEL_FILE)
    print("Loaded existing XGBoost model")
except FileNotFoundError:
    model = train_model()

# ────────────────────────────────────────────────
# Use real model for fair bid prediction
# ────────────────────────────────────────────────
def is_fair_bid(contract_id: int, bid_amount: float) -> tuple:
    if contract_id >= len(df_bidding):
        return "Under Review", 0, 0

    row = df_bidding.iloc[contract_id]

    input_dict = {
        "estimated_length_km": float(row.get("estimated_length_km", 100)),
        "terrain_type": str(row.get("terrain_type", "Semi-arid flat")),
        "latitude": float(row.get("latitude", 0)),
        "longitude": float(row.get("longitude", 0)),
        "rainfall_mm_per_year": float(row.get("rainfall_mm_per_year", 800)),
        "elevation_m": float(row.get("elevation_m", 300)),
        "has_bridge": int(row.get("has_bridge", 0)),
        "is_dual_carriageway": int(row.get("is_dual_carriageway", 0)),
    }

    input_df = pd.DataFrame([input_dict])

    # Encode terrain_type to match training
    terrain_map = {"arid": 0, "semi-arid": 1, "rainforest": 2, "mangrove": 3, "hilly": 4}
    input_df["terrain_type"] = input_df["terrain_type"].str.lower().map(terrain_map).fillna(1)

    predicted_value = model.predict(input_df)[0]

    min_fair = predicted_value * 0.88
    max_fair = predicted_value * 1.12

    status = "Fair" if min_fair <= bid_amount <= max_fair else "Under Review"

    return status, round(min_fair / 1e9, 2), round(max_fair / 1e9, 2)

# ────────────────────────────────────────────────
# Routes (only showing relevant parts – add your other routes as before)
# ────────────────────────────────────────────────

# ... your existing login/register/contracts/bid routes ...

# Admin Dashboard with AI range
@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: sqlite3.Connection = Depends(get_db)):
    get_current_admin_id(request)

    cursor = db.cursor()
    cursor.execute("""
        SELECT b.id, b.contract_id, b.company_name, b.bid_amount, b.status, b.submitted_at,
               b.predicted_min, b.predicted_max,
               df.Project_name, df.description, df.terrain_type, df.estimated_length_km
        FROM bids b
        LEFT JOIN (SELECT ROW_NUMBER() OVER () - 1 as contract_id, * FROM df_bidding) df
               ON b.contract_id = df.contract_id
        ORDER BY b.submitted_at DESC
    """)
    bids = cursor.fetchall()

    rows = ""
    total_bids = len(bids)
    approved = sum(1 for b in bids if b["status"] == "Approved")
    rejected = sum(1 for b in bids if b["status"] == "Rejected")

    for b in bids:
        project_name = b["Project_name"] or f"Contract {b['contract_id']}"
        ai_min = b["predicted_min"] or 0
        ai_max = b["predicted_max"] or 0
        variance = ((b["bid_amount"] - (ai_min + ai_max)/2) / ((ai_min + ai_max)/2)) * 100 if ai_max > 0 else 0

        status_color = "#10b981" if b["status"] == "Approved" else "#ef4444" if b["status"] == "Rejected" else "#f59e0b"
        variance_color = "green" if abs(variance) < 15 else "red"

        rows += f"""
        <tr>
            <td>#{b['id']}</td>
            <td>{project_name}</td>
            <td>{b['company_name']}</td>
            <td>₦{b['bid_amount']:,.2f}B</td>
            <td>₦{ai_min:,.2f}B – ₦{ai_max:,.2f}B</td>
            <td style="color:{status_color};">{b['status']}</td>
            <td style="color:{variance_color};">{variance:+.1f}%</td>
            <td>{b['submitted_at'][:10]}</td>
            <td>
                <button onclick="showReviewModal({b['id']}, '{project_name}', {b['bid_amount']}, {ai_min}, {ai_max})" 
                        style="background:#3b82f6;color:white;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;">
                    Review Bid
                </button>
            </td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AISEC Admin Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f1f5f9; margin: 0; padding: 20px; }}
            .header {{ background: #1e40af; color: white; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 30px; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .stat-card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            th, td {{ padding: 16px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
            th {{ background: #1e40af; color: white; }}
            .action-btn {{ padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; }}
            .approve {{ background: #10b981; color: white; }}
            .reject {{ background: #ef4444; color: white; }}
            .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1000; }}
            .modal-content {{ background: white; margin: 5% auto; padding: 30px; width: 90%; max-width: 600px; border-radius: 12px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>AISEC Admin Dashboard</h1>
            <p>AI-Powered Contract Bidding Oversight</p>
        </div>

        <div class="stats">
            <div class="stat-card"><h3>{total_bids}</h3><p>Total Bids</p></div>
            <div class="stat-card"><h3 style="color:#10b981">{approved}</h3><p>Approved</p></div>
            <div class="stat-card"><h3 style="color:#ef4444">{rejected}</h3><p>Rejected</p></div>
        </div>

        <table>
            <tr>
                <th>ID</th>
                <th>Project</th>
                <th>Company</th>
                <th>Bid Amount</th>
                <th>AI Fair Range</th>
                <th>Status</th>
                <th>Variance</th>
                <th>Date</th>
                <th>Action</th>
            </tr>
            {rows}
        </table>

        <!-- Review Modal -->
        <div id="reviewModal" class="modal">
            <div class="modal-content">
                <h2 id="modalProject"></h2>
                <p><strong>Bid Amount:</strong> <span id="modalBid"></span></p>
                <p><strong>AI Predicted Fair Range:</strong> <span id="modalAIRange"></span></p>
                <textarea id="adminComment" rows="4" style="width:100%;padding:10px;margin-top:15px;" placeholder="Admin comment / reason for decision..."></textarea>
                <br><br>
                <button onclick="approveBid()" class="action-btn approve" style="width:48%;">Approve Bid</button>
                <button onclick="rejectBid()" class="action-btn reject" style="width:48%;">Reject Bid</button>
                <button onclick="closeModal()" style="margin-top:15px;width:100%;padding:12px;background:#64748b;color:white;border:none;border-radius:6px;cursor:pointer;">Cancel</button>
            </div>
        </div>

        <script>
            let currentBidId = null;

            function showReviewModal(bidId, project, bidAmount, aiMin, aiMax) {
                currentBidId = bidId;
                document.getElementById("modalProject").innerText = project;
                document.getElementById("modalBid").innerText = "₦" + Number(bidAmount).toLocaleString() + " Billion";
                document.getElementById("modalAIRange").innerText = "₦" + Number(aiMin).toLocaleString() + "B – ₦" + Number(aiMax).toLocaleString() + "B";
                document.getElementById("reviewModal").style.display = "block";
            }

            function closeModal() {
                document.getElementById("reviewModal").style.display = "none";
            }

            async function approveBid() {
                const comment = document.getElementById("adminComment").value;
                await fetch(`/admin/update-bid/${currentBidId}`, {
                    method: "POST",
                    headers: {"Content-Type": "application/x-www-form-urlencoded"},
                    body: `new_status=Approved&comment=${encodeURIComponent(comment)}`
                });
                location.reload();
            }

            async function rejectBid() {
                const comment = document.getElementById("adminComment").value;
                await fetch(`/admin/update-bid/${currentBidId}`, {
                    method: "POST",
                    headers: {"Content-Type": "application/x-www-form-urlencoded"},
                    body: `new_status=Rejected&comment=${encodeURIComponent(comment)}`
                });
                location.reload();
            }
        </script>
    </body>
    </html>
    """)
# ... your other routes (logout, update-bid, etc.) ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


