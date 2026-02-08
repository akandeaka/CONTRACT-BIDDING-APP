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

EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_USER = os.getenv("EMAIL_USER", "your.email@gmail.com")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "your-app-password-here")

# ────────────────────────────────────────────────
# FastAPI app & rate limiter
# ────────────────────────────────────────────────

app = FastAPI(title="AISEC – AI Secure Contracting Platform")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    lambda request, exc: HTMLResponse(
        "<h2>Too many attempts. Please try again later.</h2>",
        status_code=429
    )
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
        raise HTTPException(status_code=401, detail="Please login")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# ────────────────────────────────────────────────
# Database dependency
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

        # Default admin – CHANGE THIS PASSWORD AFTER FIRST LOGIN
        try:
            c.execute(
                "INSERT INTO admins (username, hashed_password) VALUES (?, ?)",
                ("admin", pwd_context.hash("admin2026!"))
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass

init_db()

# ────────────────────────────────────────────────
# Fake model & data (replace with real model + Google Sheets loading)
# ────────────────────────────────────────────────

class FakeModel:
    def predict(self, X):
        return [150_000_000_000]  # placeholder

model = FakeModel()

df_bidding = pd.DataFrame([
    {"project_name": "Lagos–Ibadan Expressway Dualization Phase II",
     "estimated_length_km": 120, "terrain_type": "Lowland", "has_bridge": 1},
    {"project_name": "Abuja–Keffi Road Rehabilitation",
     "estimated_length_km": 65, "terrain_type": "Savanna", "has_bridge": 0},
    {"project_name": "Calabar–Ogoja Highway Construction",
     "estimated_length_km": 180, "terrain_type": "Coastal Swamp", "has_bridge": 1},
], index=[0,1,2])

# ────────────────────────────────────────────────
# Email helper
# ────────────────────────────────────────────────

def send_email(to_email: str, subject: str, body: str) -> bool:
    if not EMAIL_USER or not EMAIL_APP_PASSWORD:
        print("[EMAIL] Credentials missing – skipping send")
        return False
    try:
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        import smtplib

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
# Bid fairness check (placeholder – replace with real logic)
# ────────────────────────────────────────────────

def is_fair_bid(bid_amount: float) -> tuple:
    base = model.predict([[1]])[0]
    adjusted = base * 1.24           # rough inflation adjustment
    min_fair = adjusted * 0.88
    max_fair = adjusted * 1.15
    status = "Approved" if min_fair <= bid_amount <= max_fair else "Under Review"
    return status, round(min_fair / 1e9, 2), round(max_fair / 1e9, 2)

# ────────────────────────────────────────────────
# HTML templates (simple embedded versions)
# ────────────────────────────────────────────────

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AISEC Login</title>
  <style>
    body {font-family:Arial,sans-serif; background:#f0f4f8; margin:0; display:flex; justify-content:center; align-items:center; min-height:100vh;}
    .card {background:white; padding:2.5rem; border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,0.15); width:380px;}
    h2 {text-align:center; color:#1e40af;}
    input {width:100%; padding:12px; margin:10px 0; border:1px solid #d1d5db; border-radius:6px; box-sizing:border-box;}
    button {width:100%; padding:12px; background:#2563eb; color:white; border:none; border-radius:6px; font-weight:bold; cursor:pointer;}
    button:hover {background:#1d4ed8;}
    .error {color:#dc2626; text-align:center; margin-top:1rem;}
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
    <p style="text-align:center; margin-top:1.2rem;">
      New user? <a href="/register" style="color:#2563eb;">Register here</a>
    </p>
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
  <style>
    body {font-family:Arial,sans-serif; background:#f0f4f8; margin:0; display:flex; justify-content:center; align-items:center; min-height:100vh;}
    .card {background:white; padding:2.5rem; border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,0.15); width:420px;}
    h2 {text-align:center; color:#1e40af;}
    input {width:100%; padding:12px; margin:10px 0; border:1px solid #d1d5db; border-radius:6px; box-sizing:border-box;}
    button {width:100%; padding:12px; background:#10b981; color:white; border:none; border-radius:6px; font-weight:bold; cursor:pointer;}
    .error {color:#dc2626; text-align:center;}
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
    <p style="text-align:center; margin-top:1.2rem;">
      Already have account? <a href="/login" style="color:#2563eb;">Login</a>
    </p>
  </div>
</body>
</html>
"""

# ────────────────────────────────────────────────
# Routes – Authentication
# ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse("/login", status_code=303)

@app.get("/register", response_class=HTMLResponse)
async def register_page():
    return HTMLResponse(REGISTER_HTML)

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
    company_name = company_name.strip()
    cac_number = cac_number.strip()

    if len(password) < 8:
        return HTMLResponse(
            REGISTER_HTML + '<p class="error">Password must be at least 8 characters</p>',
            status_code=400
        )

    hashed = pwd_context.hash(password)

    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (email, hashed_password, company_name, cac_number) VALUES (?, ?, ?, ?)",
            (email, hashed, company_name, cac_number)
        )
        db.commit()
        return HTMLResponse(
            '<h2 style="color:#198754; text-align:center; margin-top:120px;">'
            'Registration successful!<br><a href="/login">→ Login</a></h2>'
        )
    except sqlite3.IntegrityError:
        return HTMLResponse(
            REGISTER_HTML + '<p class="error">Email already registered</p>',
            status_code=409
        )

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return HTMLResponse(LOGIN_HTML)

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
        return HTMLResponse(
            LOGIN_HTML + '<p class="error">Invalid email or password</p>',
            status_code=401
        )

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

# ────────────────────────────────────────────────
# Placeholder – add your other routes here
# (contracts, bid submission, admin panel, etc.)
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
