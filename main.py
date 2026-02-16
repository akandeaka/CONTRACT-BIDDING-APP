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

        # Use environment variable with short fallback (avoids bcrypt 72-byte limit)
        admin_password = os.getenv("ADMIN_PASSWORD", "AISEC26!")  # short & safe
        admin_password_bytes = admin_password.encode('utf-8')
        if len(admin_password_bytes) > 72:
            admin_password_bytes = admin_password_bytes[:72]
            admin_password = admin_password_bytes.decode('utf-8', errors='ignore')
            print("Warning: ADMIN_PASSWORD truncated to 72 bytes")

        try:
            c.execute("INSERT INTO admins (username, hashed_password) VALUES (?, ?)",
                      ("admin", pwd_context.hash(admin_password)))
            conn.commit()
            print("Admin user created/updated successfully")
        except sqlite3.IntegrityError:
            print("Admin user already exists - skipping creation")

@app.on_event("startup")
def startup_event():
    init_db()
    print("Database tables initialized on startup")

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

    if df_bidding.empty:
        print("ERROR: No data loaded from Google Sheet. Cannot train model.")
        model = xgb.XGBRegressor()  # dummy fallback
        return model

    df = df_bidding.copy()

    # Convert categorical to numeric safely
    for col in ["terrain_type", "geopolitical_zone"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    # Use only existing features
    possible_features = [
        "estimated_length_km", "terrain_type", "latitude", "longitude",
        "rainfall_mm_per_year", "elevation_m", "has_bridge", "is_dual_carriageway"
    ]
    features = [f for f in possible_features if f in df.columns]

    if not features:
        print("ERROR: No valid features found in sheet. Using dummy model.")
        model = xgb.XGBRegressor()
        return model

    # Proxy target if no real cost column
    target_col = "boq_total_cost"
    if target_col not in df.columns:
        print("No real cost column → using proxy target (length * 1.2B)")
        df[target_col] = df["estimated_length_km"] * 1_200_000_000

    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df[target_col]

    if len(X) < 10:
        print("WARNING: Too few rows for training. Using dummy model.")
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

# Load or train model
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

    # Force all to float64
    input_df = input_df.astype(float)

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

# ── Register ─────────────────────────────────────
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
async def register(
    request: Request,
    company_name: str = Form(...),
    cac_number: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: sqlite3.Connection = Depends(get_db)
):
    try:
        email = email.strip().lower()
        if len(password) < 8:
            return HTMLResponse(register_page() + '<p style="color:red">Password must be at least 8 characters</p>', status_code=400)

        hashed = pwd_context.hash(password)
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO users (email, hashed_password, company_name, cac_number) VALUES (?, ?, ?, ?)",
            (email, hashed, company_name.strip(), cac_number.strip())
        )
        db.commit()
        return HTMLResponse("""
        <h2 style="color:green; text-align:center; margin-top:120px;">
            Registration successful!<br>
            <a href="/login">Login here</a>
        </h2>
        """)
    except sqlite3.OperationalError as e:
        print(f"DB OperationalError during registration: {e}")
        return HTMLResponse(register_page() + f'<p style="color:red">Database error: {str(e)}</p>', status_code=500)
    except Exception as e:
        print(f"Unexpected error during registration: {e}")
        return HTMLResponse(register_page() + '<p style="color:red">Server error. Please try again later.</p>', status_code=500)

# ── Login ────────────────────────────────────────
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

# ── Bid Form ─────────────────────────────────────────────────────────────
@app.get("/bid/{contract_id}", response_class=HTMLResponse)
async def bid_form(request: Request, contract_id: int):
    get_current_user_id(request)
    if contract_id < 0 or contract_id >= len(df_bidding):
        raise HTTPException(404, "Contract not found")
    project = df_bidding.iloc[contract_id].get("Project_name", f"Contract {contract_id}")

    return HTMLResponse(f"""
    <!DOCTYPE html><html><head><title>Bid - {project}</title>
    <style>body{{font-family:Arial;background:#f0f4f8;padding:2rem;}}
    .card{{background:white;max-width:600px;margin:auto;padding:2.5rem;border-radius:12px;box-shadow:0 6px 20px rgba(0,0,0,0.1);}}
    label{{display:block;margin:1rem 0 0.5rem;font-weight:600;}} input,textarea{{width:100%;padding:12px;border:1px solid #d1d5db;border-radius:6px;}}
    button{{margin-top:2rem;width:100%;padding:14px;background:#10b981;color:white;border:none;border-radius:8px;font-size:1.1rem;cursor:pointer;}}</style></head>
    <body><div class="card"><h2>Bid for: {project}</h2>
    <form method="post">
      <label>Company Name</label><input name="company_name" required>
      <label>CAC Number</label><input name="cac_number" required>
      <label>Email</label><input type="email" name="email" required>
      <label>Phone</label><input name="phone" required>
      <label>Bid Amount (₦ Billion)</label><input type="number" step="0.01" name="bid_amount" required>
      <label>Equipment List</label><textarea name="equipment_list" rows="4" required></textarea>
      <label>Workforce</label><textarea name="workforce" rows="4" required></textarea>
      <button type="submit">Submit Bid</button>
    </form></div></body></html>
    """)

# ── Bid Submission ───────────────────────────────────────────────────────────
@app.post("/bid/{contract_id}", response_class=HTMLResponse)
@limiter.limit("3/hour")
async def submit_bid(request: Request, contract_id: int,
                     company_name: str = Form(...), cac_number: str = Form(...),
                     email: str = Form(...), phone: str = Form(...),
                     bid_amount: float = Form(...),
                     equipment_list: str = Form(...), workforce: str = Form(...),
                     db: sqlite3.Connection = Depends(get_db)):
    user_id = get_current_user_id(request)
    if contract_id < 0 or contract_id >= len(df_bidding):
        raise HTTPException(404, "Contract not found")
    if bid_amount <= 0:
        raise HTTPException(400, "Bid amount must be positive")

    status, min_fair, max_fair = is_fair_bid(contract_id, bid_amount)

    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO bids (contract_id, user_id, company_name, cac_number, email, phone,
                          bid_amount, equipment_list, workforce, status, predicted_min, predicted_max)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (contract_id, user_id, company_name.strip(), cac_number.strip(), email.strip(), phone.strip(),
          bid_amount, equipment_list.strip(), workforce.strip(), status, min_fair, max_fair))
    db.commit()

    project_name = df_bidding.iloc[contract_id].get("Project_name", "Unknown project")

    return HTMLResponse(f"""
    <!DOCTYPE html><html><head><title>Success</title>
    <style>body{{font-family:Arial;background:#f0fdf4;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;}}
    .card{{background:white;padding:3rem;border-radius:16px;box-shadow:0 10px 30px rgba(16,185,129,0.25);max-width:600px;text-align:center;}}
    h1{{color:#065f46;}} .btn{{display:inline-block;padding:14px 32px;background:#1e40af;color:white;border-radius:8px;text-decoration:none;font-weight:bold;margin-top:2rem;}}</style></head>
    <body><div class="card">
      <h1>✓ Bid Submitted Successfully</h1>
      <p>Your bid for <strong>{project_name}</strong> of <strong>₦{bid_amount:,.2f} Billion</strong> has been received.</p>
      <p>We will review it shortly.</p>
      <a href="/contracts" class="btn">Back to Contracts</a>
    </div></body></html>
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
    import os
    port = int(os.getenv("PORT", 8000))  # Render sets PORT env var
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
