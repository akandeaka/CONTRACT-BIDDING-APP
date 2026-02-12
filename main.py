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
# Train / Load Real XGBoost Model
# ────────────────────────────────────────────────
MODEL_FILE = "ai_contract_model.joblib"

def train_model():
    global model
    print("Training XGBoost model...")

    df = df_bidding.copy()
    for col in ["terrain_type", "geopolitical_zone"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    features = ["estimated_length_km", "terrain_type", "latitude", "longitude",
                "rainfall_mm_per_year", "elevation_m", "has_bridge", "is_dual_carriageway"]

    if "boq_total_cost" not in df.columns:
        df["boq_total_cost"] = df["estimated_length_km"] * 1_200_000_000

    X = df[features].fillna(0)
    y = df["boq_total_cost"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
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
# Real AI Prediction
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
    terrain_map = {"arid": 0, "semi-arid": 1, "rainforest": 2, "mangrove": 3, "hilly": 4}
    input_df["terrain_type"] = input_df["terrain_type"].str.lower().map(terrain_map).fillna(1)

    predicted_value = model.predict(input_df)[0]
    min_fair = predicted_value * 0.88
    max_fair = predicted_value * 1.12
    status = "Fair" if min_fair <= bid_amount <= max_fair else "Under Review"

    return status, round(min_fair / 1e9, 2), round(max_fair / 1e9, 2)

# ────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse("/login", status_code=303)

# Register & Login (clean)
@app.get("/register", response_class=HTMLResponse)
async def register_page():
    return """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Register</title>
    <style>body{font-family:Arial;background:#f0f4f8;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;}
    .card{background:white;padding:2.5rem;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,0.15);width:420px;}
    input,button{width:100%;padding:12px;margin:10px 0;border:1px solid #d1d5db;border-radius:6px;}
    button{background:#10b981;color:white;border:none;font-weight:bold;cursor:pointer;}</style></head>
    <body><div class="card"><h2>Create Account</h2><form method="post">
    <input name="company_name" placeholder="Company Name" required>
    <input name="cac_number" placeholder="CAC Number" required>
    <input type="email" name="email" placeholder="Email" required>
    <input type="password" name="password" placeholder="Password" required>
    <button type="submit">Register</button></form></div></body></html>"""

@app.post("/register", response_class=HTMLResponse)
@limiter.limit("5/minute")
async def register(request: Request, company_name: str = Form(...), cac_number: str = Form(...),
                   email: str = Form(...), password: str = Form(...), db: sqlite3.Connection = Depends(get_db)):
    email = email.strip().lower()
    if len(password) < 8:
        return HTMLResponse(register_page() + '<p style="color:red">Password too short</p>', status_code=400)
    hashed = pwd_context.hash(password)
    cursor = db.cursor()
    try:
        cursor.execute("INSERT INTO users (email, hashed_password, company_name, cac_number) VALUES (?,?,?,?)",
                       (email, hashed, company_name.strip(), cac_number.strip()))
        db.commit()
        return HTMLResponse('<h2 style="color:green;text-align:center;margin-top:120px;">Registration successful!<br><a href="/login">Login</a></h2>')
    except sqlite3.IntegrityError:
        return HTMLResponse(register_page() + '<p style="color:red">Email already registered</p>', status_code=409)

# Login routes (same as before)
@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Login</title>
    <style>body{font-family:Arial;background:#f0f4f8;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;}
    .card{background:white;padding:2.5rem;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,0.15);width:380px;}
    input,button{width:100%;padding:12px;margin:10px 0;border:1px solid #d1d5db;border-radius:6px;}
    button{background:#2563eb;color:white;border:none;font-weight:bold;cursor:pointer;}</style></head>
    <body><div class="card"><h2>Login</h2><form method="post">
    <input type="email" name="email" placeholder="Email" required>
    <input type="password" name="password" placeholder="Password" required>
    <button type="submit">Sign In</button></form></div></body></html>"""

@app.post("/login", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def login_user(request: Request, email: str = Form(...), password: str = Form(...),
                     db: sqlite3.Connection = Depends(get_db)):
    email = email.strip().lower()
    cursor = db.cursor()
    cursor.execute("SELECT id, hashed_password FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    if not user or not pwd_context.verify(password, user["hashed_password"]):
        return HTMLResponse(login_page() + '<p style="color:red">Invalid credentials</p>', status_code=401)

    token = create_access_token(str(user["id"]))
    resp = RedirectResponse("/contracts", status_code=303)
    resp.set_cookie(key="session_token", value=token, httponly=True, secure=True, samesite="lax",
                    max_age=ACCESS_TOKEN_EXPIRE_HOURS * 3600)
    return resp

@app.get("/logout")
async def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("session_token")
    return resp

# ── Contracts List ───────────────────────────────
@app.get("/contracts", response_class=HTMLResponse)
@limiter.limit("20/minute")
async def list_contracts(request: Request, db: sqlite3.Connection = Depends(get_db)):
    user_id = get_current_user_id(request)

    cursor = db.cursor()
    cursor.execute("SELECT contract_id FROM bids WHERE user_id = ?", (user_id,))
    already_bid = {r["contract_id"] for r in cursor.fetchall()}

    available = []
    for idx, row in df_bidding.iterrows():
        if idx not in already_bid:
            available.append({
                "id": idx,
                "project_name": row.get("Project_name", f"Contract {idx}"),
                "description": row.get("Description", "No description available"),
                "location": f"Lat: {row.get('latitude','N/A')}, Lon: {row.get('longitude','N/A')}",
                "terrain": row.get("terrain_type", "N/A"),
                "length_km": row.get("estimated_length_km", "N/A"),
            })

    if not available:
        return HTMLResponse("<h2 style='text-align:center;margin-top:120px;'>No contracts available.</h2>")

    items = "".join(f"""
        <div style="border:1px solid #d1d5db;border-radius:8px;padding:1.5rem;margin-bottom:1.5rem;background:white;">
            <h3>{c['project_name']}</h3>
            <p><strong>Description:</strong> {c['description']}</p>
            <p><strong>Location:</strong> {c['location']}</p>
            <p><strong>Terrain:</strong> {c['terrain']} • Length: {c['length_km']} km</p>
            <a href="/bid/{c['id']}" style="color:#2563eb;font-weight:bold;">→ Place Bid</a>
        </div>
    """ for c in available)

    return HTMLResponse(f"""
    <!DOCTYPE html><html><head><title>Available Contracts</title>
    <style>body{{font-family:Arial;background:#f8fafc;padding:2rem;}} h1{{color:#1e40af;}} .container{{max-width:900px;margin:auto;}}</style>
    </head><body><div class="container">
        <h1>Available Contracts</h1>
        <a href="/logout" style="float:right;color:#ef4444;">Logout</a>
        {items}
    </div></body></html>
    """)

# ── Bid Form & Submit ─────────────────────────────
@app.get("/bid/{contract_id}", response_class=HTMLResponse)
def is_fair_bid(contract_id: int, bid_amount: float) -> tuple:
    """
    Predict fair cost range using trained XGBoost model.
    Forces all input columns to be numeric.
    """
    if contract_id >= len(df_bidding):
        return "Under Review", 0, 0

    row = df_bidding.iloc[contract_id]

    # Create input with safe numeric defaults
    terrain_map = {
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
        # Add more known values here
    }

    terrain_str = str(row.get("terrain_type", "semi-arid flat")).lower().strip()
    terrain_code = terrain_map.get(terrain_str, 1)  # default to 1 if unknown

    input_dict = {
        "estimated_length_km": float(row.get("estimated_length_km", 100)),
        "terrain_type": float(terrain_code),               # ← numeric
        "latitude": float(row.get("latitude", 0)),
        "longitude": float(row.get("longitude", 0)),
        "rainfall_mm_per_year": float(row.get("rainfall_mm_per_year", 800)),
        "elevation_m": float(row.get("elevation_m", 300)),
        "has_bridge": 1.0 if str(row.get("has_bridge", "No")).lower() in ["yes", "1", "true", "y"] else 0.0,
        "is_dual_carriageway": 1.0 if str(row.get("is_dual_carriageway", "No")).lower() in ["yes", "1", "true", "y"] else 0.0,
    }

    input_df = pd.DataFrame([input_dict])

    # Final safety: force EVERY column to numeric
    input_df = input_df.astype(float)

    # Optional debug (remove after testing)
    # print("Input dtypes:\n", input_df.dtypes)
    # print("Input values:\n", input_df.iloc[0].to_dict())

    predicted_value = model.predict(input_df)[0]

    min_fair = predicted_value * 0.88
    max_fair = predicted_value * 1.12

    status = "Fair" if min_fair <= bid_amount <= max_fair else "Under Review"

    return status, round(min_fair / 1e9, 2), round(max_fair / 1e9, 2)
# ── Admin Dashboard (Clean & Functional) ─────────────────────────────────────
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
                    Review
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

# ── Admin Login ───────────────────────────────────────────────────────────────
@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page():
    return """
    <!DOCTYPE html><html><head><title>Admin Login</title>
    <style>body{font-family:Arial;background:#eff6ff;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;}
    .card{background:white;padding:3rem;border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,0.15);width:400px;}
    input,button{width:100%;padding:14px;margin:12px 0;border:1px solid #d1d5db;border-radius:8px;}
    button{background:#2563eb;color:white;border:none;font-weight:bold;cursor:pointer;}</style></head>
    <body><div class="card"><h2>Admin Login</h2>
    <form method="post">
      <input type="text" name="username" placeholder="Username" required>
      <input type="password" name="password" placeholder="Password" required>
      <button type="submit">Login</button>
    </form></div></body></html>
    """

@app.post("/admin/login", response_class=HTMLResponse)
async def admin_login(username: str = Form(...), password: str = Form(...),
                      db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT id, hashed_password FROM admins WHERE username = ?", (username,))
    admin = cursor.fetchone()

    if not admin or not pwd_context.verify(password, admin["hashed_password"]):
        return HTMLResponse((await admin_login_page()) + '<p style="color:red;text-align:center;margin-top:1.5rem;">Invalid credentials</p>', status_code=401)

    token = create_access_token(str(admin["id"]))
    resp = RedirectResponse("/admin/dashboard", status_code=303)
    resp.set_cookie(key="admin_token", value=token, httponly=True, secure=True, samesite="lax",
                    max_age=ACCESS_TOKEN_EXPIRE_HOURS * 3600)
    return resp

@app.post("/admin/update-bid/{bid_id}")
async def update_bid_status(request: Request, bid_id: int, new_status: str = Form(...),
                            db: sqlite3.Connection = Depends(get_db)):
    get_current_admin_id(request)
    if new_status not in ["Approved", "Rejected"]:
        raise HTTPException(400, "Invalid status")
    cursor = db.cursor()
    cursor.execute("UPDATE bids SET status = ? WHERE id = ?", (new_status, bid_id))
    db.commit()
    return RedirectResponse("/admin/dashboard", status_code=303)

@app.get("/admin/logout")
async def admin_logout():
    resp = RedirectResponse("/admin/login", status_code=303)
    resp.delete_cookie("admin_token")
    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

