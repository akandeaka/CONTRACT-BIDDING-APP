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

def get_current_admin_id(request: Request) -> int:
    token = request.cookies.get("admin_token")
    if not token:
        raise HTTPException(status_code=401, detail="Admin login required")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Admin session expired")
    except Exception:
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
        
        # Users table
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            company_name TEXT NOT NULL,
            cac_number TEXT NOT NULL
        )''')

        # Bids table
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

        # Admins table ← NEW
        c.execute('''CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        )''')

        # Default admin account (CHANGE PASSWORD after first login!)
        try:
            c.execute(
                "INSERT INTO admins (username, hashed_password) VALUES (?, ?)",
                ("admin", pwd_context.hash("AdminSecure2026!"))
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # already exists

init_db()

# ────────────────────────────────────────────────
# Real contract data (Google Sheets)
# ────────────────────────────────────────────────

BIDDING_CONTRACTS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS-nWpM2oCQ5xmda7a3tlLiRmMC2VaAdG4IhoQsypuVvbYDgtDaWn_bYcClrc35XUoHRvvMEISXTvCw/pub?output=csv"

try:
    df_bidding = pd.read_csv(BIDDING_CONTRACTS_URL).reset_index(drop=True)
    # Ensure all needed columns exist (with fallback)
    required = ["project_name", "description", "latitude", "longitude", "terrain_type", "estimated_length_km"]
    for col in required:
        if col not in df_bidding.columns:
            df_bidding[col] = "N/A"
except Exception as e:
    print(f"Could not load contracts sheet: {e}")
    df_bidding = pd.DataFrame(columns=["project_name", "description", "latitude", "longitude", "terrain_type", "estimated_length_km"])
BIDDING_CONTRACTS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS-nWpM2oCQ5xmda7a3tlLiRmMC2VaAdG4IhoQsypuVvbYDgtDaWn_bYcClrc35XUoHRvvMEISXTvCw/pub?output=csv"

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
# ====================== ADMIN LOGIN ======================

@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head><meta charset="UTF-8"><title>Admin Login - AISEC</title>
    <style>
        body {font-family:Arial,sans-serif; background:#eff6ff; margin:0; display:flex; justify-content:center; align-items:center; min-height:100vh;}
        .card {background:white; padding:3rem; border-radius:12px; box-shadow:0 10px 30px rgba(0,0,0,0.15); width:400px;}
        h2 {text-align:center; color:#1e40af;}
        input {width:100%; padding:14px; margin:12px 0; border:1px solid #d1d5db; border-radius:8px;}
        button {width:100%; padding:14px; background:#2563eb; color:white; border:none; border-radius:8px; font-weight:bold; cursor:pointer;}
    </style></head>
    <body>
      <div class="card">
        <h2>AISEC Admin Portal</h2>
        <form method="post">
          <input type="text" name="username" placeholder="Username" required>
          <input type="password" name="password" placeholder="Password" required>
          <button type="submit">Login as Admin</button>
        </form>
      </div>
    </body>
    </html>
    """
    
    @app.post("/admin/login", response_class=HTMLResponse)
async def admin_login(
    username: str = Form(...),
    password: str = Form(...),
    db: sqlite3.Connection = Depends(get_db)
):
    cursor = db.cursor()
    cursor.execute("SELECT id, hashed_password FROM admins WHERE username = ?", (username,))
    admin = cursor.fetchone()

    if not admin or not pwd_context.verify(password, admin["hashed_password"]):
        return HTMLResponse(
            admin_login_page() + '<p style="color:red;text-align:center;margin-top:20px;">Invalid credentials</p>',
            status_code=401
        )

    token = create_access_token(str(admin["id"]))

    resp = RedirectResponse("/admin/dashboard", status_code=303)
    resp.set_cookie(
        key="admin_token",
        value=token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_HOURS * 3600
    )
    return resp

    @app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: sqlite3.Connection = Depends(get_db)):
    get_current_admin_id(request)  # Only admin can access

    cursor = db.cursor()
    cursor.execute("""
        SELECT b.id, b.contract_id, b.company_name, b.bid_amount, b.status, b.submitted_at,
               df.project_name
        FROM bids b
        LEFT JOIN (SELECT row_number() OVER () - 1 as contract_id, project_name FROM df_bidding) df 
               ON b.contract_id = df.contract_id
        ORDER BY b.submitted_at DESC
    """)
    bids = cursor.fetchall()

    rows = ""
    for b in bids:
        project_name = b["project_name"] or f"Contract {b['contract_id']}"
        status_color = "#10b981" if b["status"] == "Approved" else "#ef4444"

        rows += f"""
        <tr>
            <td><strong>#{b['id']}</strong></td>
            <td>{project_name}</td>
            <td>{b['company_name']}</td>
            <td>₦{b['bid_amount']:,.2f}B</td>
            <td style="color:{status_color};">{b['status']}</td>
            <td>{b['submitted_at']}</td>
            <td>
                <form action="/admin/update-bid/{b['id']}" method="post" style="display:inline;">
                    <input type="hidden" name="new_status" value="Approved">
                    <button type="submit" style="background:#10b981;color:white;border:none;padding:6px 12px;border-radius:4px;">Approve</button>
                </form>
                <form action="/admin/update-bid/{b['id']}" method="post" style="display:inline;">
                    <input type="hidden" name="new_status" value="Rejected">
                    <button type="submit" style="background:#ef4444;color:white;border:none;padding:6px 12px;border-radius:4px;">Reject</button>
                </form>
            </td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html><head><title>AISEC Admin Dashboard</title>
    <style>
        body {{font-family:Arial,sans-serif; background:#f8fafc; margin:0; padding:2rem;}}
        h1 {{color:#1e40af; text-align:center;}}
        table {{width:100%; border-collapse:collapse; background:white; box-shadow:0 4px 12px rgba(0,0,0,0.1);}}
        th, td {{padding:14px; text-align:left; border-bottom:1px solid #e2e8f0;}}
        th {{background:#eff6ff;}}
        .logout {{float:right; color:#ef4444; text-decoration:none; font-weight:bold;}}
    </style>
    </head>
    <body>
        <h1>Admin Dashboard – All Bids</h1>
        <a href="/admin/logout" class="logout">Logout</a>
        <table>
            <tr><th>ID</th><th>Project</th><th>Company</th><th>Bid Amount</th><th>Status</th><th>Date</th><th>Action</th></tr>
            {rows}
        </table>
    </body></html>
    """
    return HTMLResponse(html)

    
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
    <style>body{{font-family:Arial;background:#f8fafc;margin:0;padding:2rem;}} h1{{color:#1e40af;text-align:center;}} .container{{max-width:900px;margin

# ── Bid submission ───────────────────────────────

# ── Bid form (GET) ──────────────────────────────────────────────────────────────
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
                "description": row.get("description", "No description provided"),
                "location": f"Lat: {row.get('latitude', 'N/A')}, Lon: {row.get('longitude', 'N/A')}",
                "terrain": row.get("terrain_type", "N/A"),
                "length_km": row.get("estimated_length_km", "N/A"),
            })

    if not available:
        return HTMLResponse("""
        <h2 style="text-align:center; margin-top:100px; color:#6b7280;">
            No more contracts available to bid on at the moment.
        </h2>
        """)

    items = ""
    for c in available:
        items += f"""
        <div style="border:1px solid #d1d5db; border-radius:8px; padding:1.5rem; margin-bottom:1.5rem; background:white;">
            <h3>{c['project_name']}</h3>
            <p><strong>Description:</strong> {c['description']}</p>
            <p><strong>Location:</strong> {c['location']}</p>
            <p><strong>Terrain:</strong> {c['terrain']} • Length: {c['length_km']} km</p>
            <a href="/bid/{c['id']}" style="color:#2563eb; font-weight:bold; text-decoration:none;">
                → Place Bid
            </a>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Available Contracts - AISEC</title>
    <style>
        body {{font-family:Arial,sans-serif; background:#f8fafc; margin:0; padding:2rem;}}
        h1 {{color:#1e40af; text-align:center;}}
        .container {{max-width:900px; margin:0 auto;}}
        a.logout {{float:right; color:#ef4444; text-decoration:none; font-weight:bold;}}
    </style>
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
    # After successful insert
    project_name = df_bidding.iloc[contract_id].get("project_name", "Unknown project")

    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head><meta charset="UTF-8"><title>Bid Submitted - AISEC</title>
    <style>
        body {{font-family:Arial,sans-serif; background:#f0fdf4; display:flex; justify-content:center; align-items:center; min-height:100vh; margin:0;}}
        .card {{background:white; padding:3rem; border-radius:16px; box-shadow:0 10px 30px rgba(16,185,129,0.25); max-width:600px; text-align:center;}}
        h1 {{color:#065f46;}}
        .btn {{display:inline-block; padding:14px 32px; background:#1e40af; color:white; border-radius:8px; text-decoration:none; font-weight:bold; margin-top:2rem;}}
    </style>
    </head>
    <body>
      <div class="card">
        <h1>✓ Bid Submitted Successfully</h1>
        <p style="font-size:1.1rem; margin:2rem 0;">
            Your bid for <strong>{project_name}</strong><br>
            in the amount of <strong>₦{bid_amount:,.2f} Billion</strong> has been received.
        </p>
        <p>We will review it shortly. You will be notified of the outcome.</p>
        <a href="/contracts" class="btn">Back to Contracts</a>
      </div>
    </body>
    </html>
    """)

    
# ── Submit bid (POST) ───────────────────────────────────────────────────────────

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
    workforce: str = Form(...),
    db: sqlite3.Connection = Depends(get_db)   # ← correct dependency injection
):
    user_id = get_current_user_id(request)

    if contract_id < 0 or contract_id >= len(df_bidding):
        raise HTTPException(404, "Contract not found")

    if bid_amount <= 0:
        raise HTTPException(400, "Bid amount must be positive")

    status, min_b, max_b = is_fair_bid(bid_amount)

    cursor = db.cursor()
    try:
        cursor.execute("""
            INSERT INTO bids (
                contract_id, user_id, company_name, cac_number, email, phone,
                bid_amount, equipment_list, workforce, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            contract_id, user_id, company_name.strip(), cac_number.strip(),
            email.strip(), phone.strip(), bid_amount,
            equipment_list.strip(), workforce.strip(), status
        ))
        db.commit()

        project_name = df_bidding.iloc[contract_id].get("project_name", "Unknown project")

        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head><meta charset="UTF-8"><title>Bid Submitted</title>
        <style>body{{font-family:Arial;background:#f0fdf4;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;}}
        .card{{background:white;padding:3rem;border-radius:16px;box-shadow:0 10px 30px rgba(16,185,129,0.25);max-width:600px;text-align:center;}}
        h1{{color:#065f46;}} .info{{background:#ecfdf5;padding:1.5rem;border-radius:12px;margin:1.5rem 0;}}
        .btn{{display:inline-block;padding:14px 32px;background:#1e40af;color:white;border-radius:8px;text-decoration:none;font-weight:bold;margin-top:1.5rem;}}
        </style></head>
        <body>
          <div class="card">
            <h1>✓ Bid Submitted Successfully</h1>
            <div class="info">
              <p><strong>Project:</strong> {project_name}</p>
              <p><strong>Your Bid:</strong> ₦{bid_amount:,.2f} Billion</p>
              <p><strong>AI Assessment:</strong> {status}</p>
              <p><strong>Estimated fair range:</strong> ₦{min_b} – ₦{max_b} Billion</p>
            </div>
            <a href="/contracts" class="btn">Back to Contracts</a>
          </div>
        </body>
        </html>
        """)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Bid submission failed: {str(e)}")

@app.post("/admin/update-bid/{bid_id}")
async def update_bid_status(
    request: Request,
    bid_id: int,
    new_status: str = Form(...),
    db: sqlite3.Connection = Depends(get_db)
):
    get_current_admin_id(request)   # admin only

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




