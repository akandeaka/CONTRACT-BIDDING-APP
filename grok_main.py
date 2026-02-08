import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from passlib.context import CryptContext
from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

app = FastAPI(title="AISEC – AI Secure Contracting Platform")

# Change these in production (use .env!)
JWT_SECRET = "super-secret-jwt-key-change-this-in-production-2026-at-least-64-chars"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_USER = "your.email@gmail.com"           # ← CHANGE
EMAIL_APP_PASSWORD = "abcd-efgh-ijkl-mnop"    # ← CHANGE (Google App Password)

# Simulated trained model & data (replace with your real ones)
# For demo: fake prediction always returns ~₦150B base
class FakeModel:
    def predict(self, X):
        return [150_000_000_000]  # 150 billion Naira base

model = FakeModel()

# Fake bidding contracts data (replace with pd.read_csv from your Google Sheet)
df_bidding = pd.DataFrame([
    {"project_name": "Lagos–Ibadan Expressway Dualization Phase II",
     "estimated_length_km": 120, "terrain_type": "Lowland", "has_bridge": 1},
    {"project_name": "Abuja–Keffi Road Rehabilitation",
     "estimated_length_km": 65, "terrain_type": "Savanna", "has_bridge": 0},
    {"project_name": "Calabar–Ogoja Highway Construction",
     "estimated_length_km": 180, "terrain_type": "Coastal Swamp", "has_bridge": 1},
], index=[0,1,2])

# ────────────────────────────────────────────────
# SECURITY & AUTH HELPERS
# ────────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": subject, "exp": expire}, JWT_SECRET, algorithm=ALGORITHM)

def get_current_user_id(request: Request) -> int:
    token = request.cookies.get("session_token")
    if not token:
        raise HTTPException(status_code=401, detail="Please login")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
        return user_id
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
# DATABASE SETUP
# ────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect("aisec_bids.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        company_name TEXT NOT NULL,
        cac_number TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS admins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS bids (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        contract_id INTEGER NOT NULL,
        user_id INTEGER,
        company_name TEXT,
        cac_number TEXT,
        email TEXT,
        phone TEXT,
        bid_amount REAL NOT NULL,
        equipment_list TEXT,
        workforce TEXT,
        status TEXT NOT NULL,
        submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')

    # Default admin (CHANGE PASSWORD after first login!)
    try:
        c.execute("INSERT INTO admins (username, hashed_password) VALUES (?, ?)",
                  ("admin", pwd_context.hash("admin2026!")))
        conn.commit()
    except sqlite3.IntegrityError:
        pass

    conn.close()

init_db()

# ────────────────────────────────────────────────
# EMAIL HELPER
# ────────────────────────────────────────────────

def send_email(to_email: str, subject: str, body: str) -> bool:
    if not EMAIL_USER or not EMAIL_APP_PASSWORD:
        print("[EMAIL] Credentials missing – skipping send")
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_APP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return False

# ────────────────────────────────────────────────
# PRICE FAIRNESS LOGIC (simplified)
# ────────────────────────────────────────────────

def is_fair_bid(bid_amount: float, contract_features: dict) -> tuple[str, float, float]:
    base = model.predict([[1]])[0]  # placeholder – use real features
    adjusted = base * 1.24  # ~2 years inflation @12%/yr
    min_fair = adjusted * 0.88
    max_fair = adjusted * 1.15
    status = "Approved" if min_fair <= bid_amount <= max_fair else "Under Review"
    return status, min_fair/1e9, max_fair/1e9  # in billions ₦

# ────────────────────────────────────────────────
# HTML PAGES (embedded)
# ────────────────────────────────────────────────

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AISEC Login</title>
  <style>body{font-family:Arial;background:#f0f4f8;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;}
  .card{background:white;padding:2.5rem;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,0.15);width:380px;}
  h2{text-align:center;color:#1e40af;} input{width:100%;padding:12px;margin:10px 0;border:1px solid #d1d5db;border-radius:6px;}
  button{width:100%;padding:12px;background:#2563eb;color:white;border:none;border-radius:6px;font-weight:bold;cursor:pointer;}
  button:hover{background:#1d4ed8;} .error{color:#dc2626;text-align:center;margin-top:1rem;}
  </style>
</head>
<body>
  <div class="card">
    <h2>AISEC Bidder Login</h2>
    <form method="post">
      <input type="email" name="email" placeholder="Email" required>
      <input type="password" name="password" placeholder="Password" required>
      <button type="submit">Sign In</button>
    </form>
    <p style="text-align:center;margin-top:1.2rem;">New? <a href="/register" style="color:#2563eb;">Register here</a></p>
  </div>
</body>
</html>
"""

REGISTER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AISEC Register</title>
  <style>body{font-family:Arial;background:#f0f4f8;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;}
  .card{background:white;padding:2.5rem;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,0.15);width:420px;}
  h2{text-align:center;color:#1e40af;} input{width:100%;padding:12px;margin:10px 0;border:1px solid #d1d5db;border-radius:6px;}
  button{width:100%;padding:12px;background:#10b981;color:white;border:none;border-radius:6px;font-weight:bold;cursor:pointer;}
  .error{color:#dc2626;text-align:center;}
  </style>
</head>
<body>
  <div class="card">
    <h2>Create Bidder Account</h2>
    <form method="post">
      <input type="text" name="company_name" placeholder="Company Name" required>
      <input type="text" name="cac_number" placeholder="CAC Number" required>
      <input type="email" name="email" placeholder="Email" required>
      <input type="password" name="password" placeholder="Password (min 8 chars)" required>
      <button type="submit">Register</button>
    </form>
    <p style="text-align:center;margin-top:1.2rem;">Already have account? <a href="/login" style="color:#2563eb;">Login</a></p>
  </div>
</body>
</html>
"""

BID_SUCCESS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Bid Submitted</title>
<style>body{font-family:Arial;background:#f0fdf4;display:flex;justify-content:center;align-items:center;min-height:100vh;}
.card{background:white;padding:3rem;border-radius:16px;box-shadow:0 10px 30px rgba(16,185,129,0.25);max-width:600px;text-align:center;}
h1{color:#065f46;} .info{background:#ecfdf5;padding:1.5rem;border-radius:12px;margin:1.5rem 0;}
.btn{display:inline-block;padding:14px 32px;background:#1e40af;color:white;border-radius:8px;text-decoration:none;font-weight:bold;margin-top:1.5rem;}
</style>
</head>
<body>
  <div class="card">
    <h1>✓ Bid Submitted Successfully</h1>
    <div class="info">{content}</div>
    <a href="/contracts" class="btn">View Other Contracts</a>
  </div>
</body>
</html>
"""

# ────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse("/login")

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return HTMLResponse(LOGIN_HTML)

@app.post("/login", response_class=HTMLResponse)
async def login(response: Response, email: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect("aisec_bids.db")
    c = conn.cursor()
    c.execute("SELECT id, hashed_password FROM users WHERE email = ?", (email.lower(),))
    row = c.fetchone()
    conn.close()

    if row and pwd_context.verify(password, row[1]):
        token = create_access_token(str(row[0]))
        resp = RedirectResponse("/contracts")
        resp.set_cookie("session_token", token, httponly=True, max_age=ACCESS_TOKEN_EXPIRE_HOURS*3600, samesite="lax")
        return resp

    return HTMLResponse(LOGIN_HTML + '<p class="error">Invalid email or password</p>', status_code=401)

@app.get("/register", response_class=HTMLResponse)
async def register_page():
    return HTMLResponse(REGISTER_HTML)

@app.post("/register", response_class=HTMLResponse)
async def register(
    company_name: str = Form(...),
    cac_number: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    if len(password) < 8:
        return HTMLResponse(REGISTER_HTML + '<p class="error">Password must be at least 8 characters</p>', status_code=400)

    hashed = pwd_context.hash(password)
    conn = sqlite3.connect("aisec_bids.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email, hashed_password, company_name, cac_number) VALUES (?,?,?,?)",
                  (email.lower(), hashed, company_name.strip(), cac_number.strip()))
        conn.commit()
        return HTMLResponse('<h2 style="color:#10b981;text-align:center;margin-top:100px;">Registration successful!<br><a href="/login">Login now</a></h2>')
    except sqlite3.IntegrityError:
        return HTMLResponse(REGISTER_HTML + '<p class="error">Email already registered</p>', status_code=409)
    finally:
        conn.close()

@app.get("/logout")
async def logout(response: Response):
    resp = RedirectResponse("/login")
    resp.delete_cookie("session_token")
    return resp

@app.get("/contracts", response_class=HTMLResponse)
async def contracts(request: Request):
    user_id = get_current_user_id(request)

    conn = sqlite3.connect("aisec_bids.db")
    c = conn.cursor()
    c.execute("SELECT contract_id FROM bids WHERE user_id = ?", (user_id,))
    bid_ids = {r[0] for r in c.fetchall()}
    conn.close()

    available = []
    for idx, row in df_bidding.iterrows():
        if idx not in bid_ids:
            available.append({"id": idx, **row})

    if not available:
        return HTMLResponse("<h2 style='text-align:center;margin-top:120px;'>All available contracts have been bid on.</h2>")

    items = ""
    for c in available:
        items += f"""
        <div style="border:1px solid #d1d5db;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;background:white;">
          <h3>{c['project_name']}</h3>
          <p>Length: {c['estimated_length_km']} km • Terrain: {c['terrain_type']}</p>
          <a href="/bid/{c['id']}" style="color:#2563eb;font-weight:bold;">Place Bid →</a>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html><head><title>Available Contracts - AISEC</title>
    <style>body{{font-family:Arial;background:#f8fafc;padding:2rem;}} h1{{color:#1e40af;}} .container{{max-width:900px;margin:auto;}}</style>
    </head><body>
    <div class="container">
      <h1>Available Road Contracts</h1>
      <a href="/logout" style="float:right;color:#ef4444;">Logout</a>
      {items}
    </div>
    </body></html>
    """
    return HTMLResponse(html)

@app.get("/bid/{contract_id}", response_class=HTMLResponse)
async def bid_page(request: Request, contract_id: int):
    get_current_user_id(request)  # auth check

    if contract_id < 0 or contract_id >= len(df_bidding):
        raise HTTPException(404, "Contract not found")

    project = df_bidding.iloc[contract_id]["project_name"]

    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head><meta charset="UTF-8"><title>Bid on {project}</title>
    <style>body{{font-family:Arial;background:#f0f4f8;padding:2rem;}}
    .form-card{{background:white;max-width:600px;margin:auto;padding:2.5rem;border-radius:12px;box-shadow:0 6px 20px rgba(0,0,0,0.1);}}
    label{{display:block;margin:1rem 0 0.4rem;font-weight:600;}} input,textarea{{width:100%;padding:12px;border:1px solid #d1d5db;border-radius:6px;}}
    button{{margin-top:1.5rem;width:100%;padding:14px;background:#10b981;color:white;border:none;border-radius:8px;font-size:1.1rem;cursor:pointer;}}
    </style></head>
    <body>
      <div class="form-card">
        <h2>Bid for: {project}</h2>
        <form method="post">
          <label>Company Name</label><input name="company_name" required>
          <label>CAC Number</label><input name="cac_number" required>
          <label>Email</label><input type="email" name="email" required>
          <label>Phone</label><input name="phone" required>
          <label>Bid Amount (₦ Billion)</label><input type="number" step="0.01" name="bid_amount" required>
          <label>Equipment List</label><textarea name="equipment_list" rows="4" required></textarea>
          <label>Workforce Description</label><textarea name="workforce" rows="4" required></textarea>
          <button type="submit">Submit Bid</button>
        </form>
      </div>
    </body></html>
    """)

@app.post("/bid/{contract_id}", response_class=HTMLResponse)
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

    status, min_b, max_b = is_fair_bid(bid_amount, {})

    conn = sqlite3.connect("aisec_bids.db")
    c = conn.cursor()
    c.execute("""
    INSERT INTO bids (contract_id, user_id, company_name, cac_number, email, phone, bid_amount, equipment_list, workforce, status)
    VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (contract_id, user_id, company_name, cac_number, email, phone, bid_amount, equipment_list, workforce, status))
    bid_id = c.lastrowid
    conn.commit()
    conn.close()

    body = f"""Bid ID: {bid_id}
Project: {df_bidding.iloc[contract_id]['project_name']}
Amount: ₦{bid_amount:.2f} Billion
Status: {status}
Fair range: ₦{min_b:.2f} – ₦{max_b:.2f} Billion"""

    send_email(email, "AISEC Bid Confirmation", body)

    content = f"""
    <p><strong>Bid ID:</strong> {bid_id}</p>
    <p><strong>Amount:</strong> ₦{bid_amount:.2f} Billion</p>
    <p><strong>AI Assessment:</strong> {status}</p>
    <p><strong>Fair Price Range:</strong> ₦{min_b:.2f} – ₦{max_b:.2f} Billion</p>
    <p>Email confirmation has been sent.</p>
    """

    return HTMLResponse(BID_SUCCESS_HTML.format(content=content))

@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html><head><title>Admin – AISEC</title>
    <style>body{font-family:Arial;background:#eff6ff;display:flex;justify-content:center;align-items:center;min-height:100vh;}
    .card{background:white;padding:2.5rem;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,0.15);width:380px;}
    h2{text-align:center;color:#1e40af;}</style></head>
    <body><div class="card">
    <h2>Admin Portal</h2>
    <form method="post">
      <input type="text" name="username" placeholder="Username" required style="width:100%;padding:12px;margin:10px 0;border-radius:6px;border:1px solid #d1d5db;">
      <input type="password" name="password" placeholder="Password" required style="width:100%;padding:12px;margin:10px 0;border-radius:6px;border:1px solid #d1d5db;">
      <button type="submit" style="width:100%;padding:12px;background:#2563eb;color:white;border:none;border-radius:6px;font-weight:bold;">Login</button>
    </form>
    </div></body></html>
    """)

@app.post("/admin/login", response_class=HTMLResponse)
async def admin_login(response: Response, username: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect("aisec_bids.db")
    c = conn.cursor()
    c.execute("SELECT id, hashed_password FROM admins WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()

    if row and pwd_context.verify(password, row[1]):
        token = create_access_token(str(row[0]))
        resp = RedirectResponse("/admin/dashboard")
        resp.set_cookie("admin_token", token, httponly=True, max_age=ACCESS_TOKEN_EXPIRE_HOURS*3600, samesite="lax")
        return resp

    return HTMLResponse("<h2 style='color:red;text-align:center'>Invalid admin credentials</h2>")

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    get_current_admin_id(request)

    conn = sqlite3.connect("aisec_bids.db")
    c = conn.cursor()
    c.execute("SELECT * FROM bids ORDER BY submitted_at DESC")
    bids = c.fetchall()
    conn.close()

    rows = ""
    for b in bids:
        rows += f"<tr><td>{b[0]}</td><td>{b[1]}</td><td>{b[3]}</td><td>{b[6]:,.2f}</td><td>{b[9]}</td><td>{b[10]}</td></tr>"

    html = f"""
    <!DOCTYPE html>
    <html><head><title>AISEC Admin</title>
    <style>body{{font-family:Arial;padding:2rem;background:#f8fafc;}} table{{width:100%;border-collapse:collapse;}} th,td{{padding:12px;border:1px solid #e2e8f0;}} th{{background:#eff6ff;}}</style>
    </head><body>
    <h1>Admin Dashboard – Bids Overview</h1>
    <table>
      <tr><th>ID</th><th>Contract</th><th>Company</th><th>Amount (₦B)</th><th>Status</th><th>Submitted</th></tr>
      {rows}
    </table>
    <p style="margin-top:2rem;"><a href="/admin/logout">Logout</a></p>
    </body></html>
    """
    return HTMLResponse(html)

@app.get("/admin/logout")
async def admin_logout(response: Response):
    resp = RedirectResponse("/admin/login")
    resp.delete_cookie("admin_token")
    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)