# main.py
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Generator

import jwt
import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from passlib.context import CryptContext
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────

JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-jwt-key-change-this-in-production-2026-at-least-64-chars")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# ────────────────────────────────────────────────
# FastAPI app & rate limiter
# ────────────────────────────────────────────────

app = FastAPI(title="AISEC – Contract Bidding Platform")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    lambda req, exc: HTMLResponse("<h2>Rate limit exceeded. Try again later.</h2>", status_code=429)
)

# ────────────────────────────────────────────────
# Password hashing
# ────────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ────────────────────────────────────────────────
# JWT helpers
# ────────────────────────────────────────────────

def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode = {"sub": subject, "exp": expire}
    return jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)

def get_current_user_id(request: Request) -> int:
    token = request.cookies.get("session_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

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
            submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()

init_db()

# ────────────────────────────────────────────────
# Real contract data (Google Sheets)
# ────────────────────────────────────────────────

BIDDING_CONTRACTS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS-nWpM2oCQ5xmda7a3tlLiRmMC2VaAdG4IhoQsypuVvbYDgtDaWn_bYcClrc35XUoHRvvMEISXTvCw/pub?output=csv"

try:
    df_bidding = pd.read_csv(BIDDING_CONTRACTS_URL).reset_index(drop=True)
except Exception as e:
    print(f"Failed to load contracts from Google Sheet: {e}")
    df_bidding = pd.DataFrame()  # fallback

# ────────────────────────────────────────────────
# Simple fairness check (placeholder)
# ────────────────────────────────────────────────

def is_fair_bid(bid_amount: float) -> tuple:
    # Replace with your real model logic
    base = 150_000_000_000
    adjusted = base * 1.24
    min_fair = adjusted * 0.88
    max_fair = adjusted * 1.15
    status = "Approved" if min_fair <= bid_amount <= max_fair else "Under Review"
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
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head><meta charset="UTF-8"><title>Register - AISEC</title>
    <style>body{font-family:Arial;background:#f0f4f8;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;}
    .card{background:white;padding:2.5rem;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,0.15);width:420px;}
    h2{text-align:center;color:#1e40af;} input{width:100%;padding:12px;margin:10px 0;border:1px solid #d1d5db;border-radius:6px;}
    button{width:100%;padding:12px;background:#10b981;color:white;border:none;border-radius:6px;font-weight:bold;cursor:pointer;}
    .error{color:#dc2626;text-align:center;}</style></head>
    <body><div class="card">
    <h2>Create Bidder Account</h2>
    <form method="post">
      <input type="text" name="company_name" placeholder="Company Name" required>
      <input type="text" name="cac_number" placeholder="CAC Number" required>
      <input type="email" name="email" placeholder="Email" required>
      <input type="password" name="password" placeholder="Password (min 8 chars)" required>
      <button type="submit">Register</button>
    </form>
    <p style="text-align:center;margin-top:1.2rem;">Already have account? <a href="/login" style="color:#2563eb;">Login</a></p>
    </div></body></html>
    """

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
    email = email.strip().lower()
    if len(password) < 8:
        return HTMLResponse(register_page() + '<p class="error">Password must be ≥8 characters</p>', status_code=400)

    hashed = pwd_context.hash(password)
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (email, hashed_password, company_name, cac_number) VALUES (?,?,?,?)",
            (email, hashed, company_name.strip(), cac_number.strip())
        )
        db.commit()
        return HTMLResponse('<h2 style="color:green;text-align:center;margin-top:120px;">Registered successfully</h2><p style="text-align:center;"><a href="/login">→ Login</a></p>')
    except sqlite3.IntegrityError:
        return HTMLResponse(register_page() + '<p class="error">Email already registered</p>', status_code=409)

# ── Login ────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head><meta charset="UTF-8"><title>AISEC Login</title>
    <style>body{font-family:Arial;background:#f0f4f8;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;}
    .card{background:white;padding:2.5rem;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,0.15);width:380px;}
    h2{text-align:center;color:#1e40af;} input{width:100%;padding:12px;margin:10px 0;border:1px solid #d1d5db;border-radius:6px;}
    button{width:100%;padding:12px;background:#2563eb;color:white;border:none;border-radius:6px;font-weight:bold;cursor:pointer;}
    .error{color:#dc2626;text-align:center;margin-top:1rem;}</style></head>
    <body><div class="card">
    <h2>AISEC Bidder Login</h2>
    <form method="post">
      <input type="email" name="email" placeholder="Email" required>
      <input type="password" name="password" placeholder="Password" required>
      <button type="submit">Sign In</button>
    </form>
    <p style="text-align:center;margin-top:1.2rem;">New? <a href="/register" style="color:#2563eb;">Register here</a></p>
    </div></body></html>
    """

@app.post("/login", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def login_user(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: sqlite3.Connection = Depends(get_db)
):
    email = email.strip().lower()

    cursor = db.cursor()
    cursor.execute("SELECT id, hashed_password FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()

    if not user or not pwd_context.verify(password, user["hashed_password"]):
        return HTMLResponse(login_page() + '<p class="error">Invalid credentials</p>', status_code=401)

    token = create_access_token(str(user["id"]))

    resp = RedirectResponse("/contracts", status_code=303)
    resp.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_HOURS * 3600
    )
    return resp

# ── Logout ───────────────────────────────────────

@app.get("/logout")
async def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie("session_token")
    return resp

# ── Contracts list ───────────────────────────────

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
                "project_name": row.get("project_name", f"Contract {idx}"),
                "estimated_length_km": row.get("estimated_length_km", "—"),
                "terrain_type": row.get("terrain_type", "—"),
            })

    if not available:
        return HTMLResponse("<h2 style='text-align:center;margin-top:120px;'>No contracts available at the moment.</h2>")

    items = "".join(
        f"""
        <div style="border:1px solid #d1d5db;border-radius:8px;padding:1.2rem;margin-bottom:1rem;background:white;">
          <h3>{c['project_name']}</h3>
          <p>Length: {c['estimated_length_km']} km • Terrain: {c['terrain_type']}</p>
          <a href="/bid/{c['id']}" style="color:#2563eb;font-weight:bold;text-decoration:none;">Place Bid →</a>
        </div>
        """ for c in available
    )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Available Contracts - AISEC</title>
    <style>body{{font-family:Arial;background:#f8fafc;margin:0;padding:2rem;}} h1{{color:#1e40af;text-align:center;}} .container{{max-width:900px;margin:auto;}} a.logout{{float:right;color:#ef4444;text-decoration:none;}}</style>
    </head>
    <body>
      <div class="container">
        <h1>Available Road Contracts</h1>
        <a href="/logout" class="logout">Logout</a>
        {items}
      </div>
    </body>
    </html>
    """
    return HTMLResponse(html)

# ── Bid submission ───────────────────────────────

@app.get("/bid/{contract_id}", response_class=HTMLResponse)
async def bid_form(request: Request, contract_id: int):
    get_current_user_id(request)  # auth check

    if contract_id < 0 or contract_id >= len(df_bidding):
        raise HTTPException(404, "Contract not found")

    project = df_bidding.iloc[contract_id].get("project_name", f"Contract {contract_id}")

    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head><meta charset="UTF-8"><title>Bid on {project}</title>
    <style>body{{font-family:Arial;background:#f0f4f8;padding:2rem;}}
    .card{{background:white;max-width:600px;margin:auto;padding:2.5rem;border-radius:12px;box-shadow:0 6px 20px rgba(0,0,0,0.1);}}
    label{{display:block;margin:1.2rem 0 0.5rem;font-weight:600;}} input,textarea{{width:100%;padding:12px;border:1px solid #d1d5db;border-radius:6px;box-sizing:border-box;}}
    button{{margin-top:2rem;width:100%;padding:14px;background:#10b981;color:white;border:none;border-radius:8px;font-size:1.1rem;cursor:pointer;}}
    </style></head>
    <body>
      <div class="card">
        <h2>Bid for: {project}</h2>
        <form method="post">
          <label>Company Name</label><input name="company_name" required>
          <label>CAC Number</label><input name="cac_number" required>
          <label>Email</label><input type="email" name="email" required>
          <label>Phone</label><input name="phone" required>
          <label>Bid Amount (₦ Billion)</label><input type="number" step="0.01" name="bid_amount" min="0.01" required>
          <label>Equipment List</label><textarea name="equipment_list" rows="4" required></textarea>
          <label>Workforce Description</label><textarea name="workforce" rows="4" required></textarea>
          <button type="submit">Submit Bid</button>
        </form>
      </div>
    </body></html>
    """)

@app.post("/bid/{contract_id}", response_class=HTMLResponse)
@limiter.limit("3/hour")
async def submit_bid(
    request: Request,
    contract_id: int,
    company_name: str = Form(...),
    cac_number: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    bid_amount: float = Form(...),
    equipment_list: str = Form(...),
    workforce: str = Form(...)
):
    user_id = get_current_user_id(request)

    if contract_id < 0 or contract_id >= len(df_bidding):
        raise HTTPException(404, "Contract not found")

    if bid_amount <= 0:
        raise HTTPException(400, "Bid amount must be positive")

    status, min_b, max_b = is_fair_bid(bid_amount)

    db = next(get_db())
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO bids (contract_id, user_id, company_name, cac_number, email, phone, bid_amount, equipment_list, workforce, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (contract_id, user_id, company_name.strip(), cac_number.strip(), email.strip(), phone.strip(),
          bid_amount, equipment_list.strip(), workforce.strip(), status))
    db.commit()

    project_name = df_bidding.iloc[contract_id].get("project_name", "Unknown project")

    return HTMLResponse(f"""
    <h1 style="color:#10b981;text-align:center;margin-top:80px;">Bid Submitted Successfully</h1>
    <div style="max-width:600px;margin:auto;padding:2rem;background:#f0fdf4;border-radius:12px;">
      <p><strong>Project:</strong> {project_name}</p>
      <p><strong>Bid Amount:</strong> ₦{bid_amount:,.2f} Billion</p>
      <p><strong>Status:</strong> {status}</p>
      <p><strong>Fair range (AI estimate):</strong> ₦{min_b} – ₦{max_b} Billion</p>
      <p style="margin-top:2rem;text-align:center;">
        <a href="/contracts" style="color:#2563eb;font-weight:bold;">→ Back to contracts</a>
      </p>
    </div>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
