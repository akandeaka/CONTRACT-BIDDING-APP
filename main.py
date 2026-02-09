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
            submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        )''')

        try:
            c.execute(
                "INSERT INTO admins (username, hashed_password) VALUES (?, ?)",
                ("admin", pwd_context.hash("AdminSecure2026!"))
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass

init_db()

# ────────────────────────────────────────────────
# Load real contracts from Google Sheet
# ────────────────────────────────────────────────
BIDDING_CONTRACTS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS-nWpM2oCQ5xmda7a3tlLiRmMC2VaAdG4IhoQsypuVvbYDgtDaWn_bYcClrc35XUoHRvvMEISXTvCw/pub?output=csv"

try:
    df_bidding = pd.read_csv(BIDDING_CONTRACTS_URL).reset_index(drop=True)
    required = ["project_name", "description", "latitude", "longitude", "terrain_type", "estimated_length_km"]
    for col in required:
        if col not in df_bidding.columns:
            df_bidding[col] = "N/A"
except Exception as e:
    print(f"Failed to load Google Sheet: {e}")
    df_bidding = pd.DataFrame(columns=["project_name", "description", "latitude", "longitude", "terrain_type", "estimated_length_km"])

# ────────────────────────────────────────────────
# Simple AI fairness check (placeholder)
# ────────────────────────────────────────────────
def is_fair_bid(bid_amount: float) -> tuple:
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

# ── User Routes ─────────────────────────────────

@app.get("/register", response_class=HTMLResponse)
async def register_page():
    return """
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Register</title>
    <style>body{font-family:Arial;background:#f0f4f8;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;}
    .card{background:white;padding:2.5rem;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,0.15);width:420px;}
    input,button{width:100%;padding:12px;margin:10px 0;border:1px solid #d1d5db;border-radius:6px;}
    button{background:#10b981;color:white;border:none;font-weight:bold;cursor:pointer;}</style></head>
    <body><div class="card"><h2>Create Account</h2><form method="post">
    <input name="company_name" placeholder="Company Name" required>
    <input name="cac_number" placeholder="CAC Number" required>
    <input type="email" name="email" placeholder="Email" required>
    <input type="password" name="password" placeholder="Password" required>
    <button type="submit">Register</button></form></div></body></html>
    """

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
        return HTMLResponse('<h2 style="color:green;text-align:center;margin-top:100px;">Registration successful!<br><a href="/login">Login</a></h2>')
    except sqlite3.IntegrityError:
        return HTMLResponse(register_page() + '<p style="color:red">Email already registered</p>', status_code=409)

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return """
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Login</title>
    <style>body{font-family:Arial;background:#f0f4f8;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;}
    .card{background:white;padding:2.5rem;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,0.15);width:380px;}
    input,button{width:100%;padding:12px;margin:10px 0;border:1px solid #d1d5db;border-radius:6px;}
    button{background:#2563eb;color:white;border:none;font-weight:bold;cursor:pointer;}</style></head>
    <body><div class="card"><h2>Login</h2><form method="post">
    <input type="email" name="email" placeholder="Email" required>
    <input type="password" name="password" placeholder="Password" required>
    <button type="submit">Sign In</button></form></div></body></html>
    """

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

# ── Contracts List (with location & description) ───────────────────────────────

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
                "description": row.get("description", "No description"),
                "location": f"Lat: {row.get('latitude','N/A')}, Lon: {row.get('longitude','N/A')}",
                "terrain": row.get("terrain_type", "N/A"),
                "length_km": row.get("estimated_length_km", "N/A"),
                "Duration": row.get("months", "N/A"),
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
    <!DOCTYPE html><html><head><title>Contracts - AISEC</title>
    <style>body{{font-family:Arial;background:#f8fafc;padding:2rem;}} h1{{color:#1e40af;}} .container{{max-width:900px;margin:auto;}}</style>
    </head><body><div class="container">
        <h1>Available Contracts</h1>
        <a href="/logout" style="float:right;color:#ef4444;">Logout</a>
        {items}
    </div></body></html>
    """)

# ── Bid Form + Submit (no AI range shown to bidder) ───────────────────────────

@app.get("/bid/{contract_id}", response_class=HTMLResponse)
async def bid_form(request: Request, contract_id: int):
    get_current_user_id(request)
    if contract_id < 0 or contract_id >= len(df_bidding):
        raise HTTPException(404, "Contract not found")
    project = df_bidding.iloc[contract_id].get("project_name", f"Contract {contract_id}")

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

@app.post("/bid/{contract_id}", response_class=HTMLResponse)
@limiter.limit("3/hour")
async def submit_bid(
    request: Request, contract_id: int,
    company_name: str = Form(...), cac_number: str = Form(...),
    email: str = Form(...), phone: str = Form(...),
    bid_amount: float = Form(...),
    equipment_list: str = Form(...), workforce: str = Form(...),
    db: sqlite3.Connection = Depends(get_db)
):
    user_id = get_current_user_id(request)
    if contract_id < 0 or contract_id >= len(df_bidding):
        raise HTTPException(404, "Contract not found")
    if bid_amount <= 0:
        raise HTTPException(400, "Bid amount must be positive")

    status, _, _ = is_fair_bid(bid_amount)   # calculate for admin only

    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO bids (contract_id, user_id, company_name, cac_number, email, phone,
                          bid_amount, equipment_list, workforce, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (contract_id, user_id, company_name.strip(), cac_number.strip(),
          email.strip(), phone.strip(), bid_amount,
          equipment_list.strip(), workforce.strip(), status))
    db.commit()

    project_name = df_bidding.iloc[contract_id].get("project_name", "Unknown project")

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

# ── Admin Portal ───────────────────────────────────────────────────────────────

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
        return HTMLResponse(admin_login_page() + '<p style="color:red;text-align:center;">Invalid credentials</p>', status_code=401)

    token = create_access_token(str(admin["id"]))
    resp = RedirectResponse("/admin/dashboard", status_code=303)
    resp.set_cookie(key="admin_token", value=token, httponly=True, secure=True, samesite="lax",
                    max_age=ACCESS_TOKEN_EXPIRE_HOURS * 3600)
    return resp

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: sqlite3.Connection = Depends(get_db)):
    get_current_admin_id(request)

    cursor = db.cursor()
    cursor.execute("""
        SELECT b.id, b.contract_id, b.company_name, b.bid_amount, b.status, b.submitted_at,
               COALESCE(df.project_name, 'Contract ' || b.contract_id) as project_name
        FROM bids b
        LEFT JOIN (SELECT ROW_NUMBER() OVER () - 1 as contract_id, project_name FROM df_bidding) df
               ON b.contract_id = df.contract_id
        ORDER BY b.submitted_at DESC
    """)
    bids = cursor.fetchall()

    rows = ""
    for b in bids:
        rows += f"""
        <tr>
            <td>#{b['id']}</td>
            <td>{b['project_name']}</td>
            <td>{b['company_name']}</td>
            <td>₦{b['bid_amount']:,.2f}B</td>
            <td>{b['status']}</td>
            <td>{b['submitted_at']}</td>
            <td>
                <form action="/admin/update-bid/{b['id']}" method="post" style="display:inline;">
                    <input type="hidden" name="new_status" value="Approved">
                    <button style="background:#10b981;color:white;border:none;padding:6px 12px;border-radius:4px;">Approve</button>
                </form>
                <form action="/admin/update-bid/{b['id']}" method="post" style="display:inline;">
                    <input type="hidden" name="new_status" value="Rejected">
                    <button style="background:#ef4444;color:white;border:none;padding:6px 12px;border-radius:4px;">Reject</button>
                </form>
            </td>
        </tr>
        """

    return HTMLResponse(f"""
    <!DOCTYPE html><html><head><title>Admin Dashboard</title>
    <style>body{{font-family:Arial;background:#f8fafc;padding:2rem;}} table{{width:100%;border-collapse:collapse;background:white;}}
    th,td{{padding:12px;text-align:left;border-bottom:1px solid #e2e8f0;}} th{{background:#eff6ff;}}</style></head>
    <body><h1>Admin Dashboard – All Bids</h1>
    <a href="/admin/logout" style="float:right;color:#ef4444;">Logout</a>
    <table><tr><th>ID</th><th>Project</th><th>Company</th><th>Bid Amount</th><th>Status</th><th>Date</th><th>Action</th></tr>
    {rows}</table></body></html>
    """)

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


