import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import FastAPI, Request, Form, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import sqlite3
import joblib
import os
import hashlib
import subprocess
import sys
import secrets

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://akandeaka.github.io",
        "http://localhost:8000",
        "https://aisec.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URLs for datasets
TRAINING_DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTXlHZrU20uniUkjr-5Pis1pfJSOYDUiFVcML6UqW2Lu176_opvZPQvTGOpQZnNx02HyFf-jRYw3O8o/pub?output=csv"
BIDDING_CONTRACTS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS-nWpM2oCQ5xmda7a3tlLiRmMC2VaAdG4IhoQsypuVvbYDgtDaWn_bYcClrc35XUoHRvvMEISXTvCw/pub?output=csv"

MODEL_PATH = "model.pkl"

def ensure_model_and_data():
    if not os.path.exists(MODEL_PATH):
        print("Training model...")
        subprocess.run([sys.executable, "train_model.py"], check=True)

ensure_model_and_data()

df_training = pd.read_csv(TRAINING_DATA_URL).reset_index(drop=True)
df_bidding = pd.read_csv(BIDDING_CONTRACTS_URL).reset_index(drop=True)
model = joblib.load(MODEL_PATH)

# ===== DATABASE SETUP (ABSOLUTE PATH) =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "bids.db")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    company_name TEXT NOT NULL,
    cac_number TEXT NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS bids (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id INTEGER NOT NULL,
    user_id INTEGER,
    company_name TEXT NOT NULL,
    cac_number TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT NOT NULL,
    bid_amount REAL NOT NULL,
    equipment_list TEXT NOT NULL,
    workforce TEXT NOT NULL,
    status TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS admins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL
)
""")

try:
    cursor.execute(
        "INSERT INTO admins (username, hashed_password) VALUES (?, ?)",
        ("admin", hashlib.sha256("admin123".encode()).hexdigest())
    )
    conn.commit()
except sqlite3.IntegrityError:
    pass

conn.commit()

# ===== EMAIL SENDER =====
def send_bid_notification(email, company_name, contract_name, status, bid_amount):
    try:
        EMAIL_HOST = "smtp.gmail.com"
        EMAIL_PORT = 587
        EMAIL_USER = "aisec2025.notifications@gmail.com"
        EMAIL_PASSWORD = "YOUR_16_CHAR_APP_PASSWORD"
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = email
        msg['Subject'] = f"AISEC Bid Submission - {status}"
        
        body = f"""Dear {company_name},

Your bid for "{contract_name}" has been successfully submitted!

Bid Amount: ₦{bid_amount:.2f} Billion
Status: {status}

Thank you for using AISEC.

Best regards,
AISEC Team"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)
        
        print(f"✓ Email sent to {email}")
        return True
    except Exception as e:
        print(f"✗ Email FAILED: {str(e)}")
        return False

# ===== SESSION MANAGEMENT =====
sessions = {}

def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    sessions[token] = user_id
    return token

def get_current_user(request: Request):
    token = request.cookies.get("session_token")
    if not token or token not in sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return sessions[token]

def get_admin_user(request: Request):
    token = request.cookies.get("admin_token")
    if not token or token not in sessions:
        raise HTTPException(status_code=401, detail="Admin not authenticated")
    return sessions[token]

# ===== PRICE RANGE CALC =====
def adjust_for_inflation(base_price, inflation_rate=0.12, years=2):
    return base_price * ((1 + inflation_rate) ** years)

def get_fair_price_range(contract_row):
    feature_columns = [
        "award_year", "award_month", "primary_state", "geopolitical_zone",
        "latitude_start", "longitude_start", "estimated_length_km",
        "terrain_type", "rainfall_mm_per_year", "soil_type", "elevation_m",
        "has_bridge", "is_dual_carriageway", "is_rehabilitation", "is_coastal_or_swamp",
        "boq_earthworks_m3_per_km", "boq_asphalt_ton_per_km", "boq_drainage_km_per_km",
        "boq_bridges_units", "boq_culverts_units", "boq_premium_percent"
    ]
    features = contract_row[feature_columns]
    features_df = pd.DataFrame([features.values], columns=features.index)
    base_price = model.predict(features_df)[0]
    adjusted = adjust_for_inflation(base_price)
    return adjusted * 0.9, adjusted * 1.1

# ===== ROUTES =====
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return RedirectResponse(url="/contracts")

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
async def register_user(
    company_name: str = Form(...),
    cac_number: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    try:
        hashed = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute(
            "INSERT INTO users (company_name, cac_number, email, hashed_password) VALUES (?, ?, ?, ?)", 
            (company_name, cac_number, email, hashed)
        )
        conn.commit()
        return "<h2 style='color:green;text-align:center;'>Registration successful!</h2>"
    except sqlite3.IntegrityError:
        return "<h2 style='color:red;text-align:center;'>Email already registered</h2>"

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login_user(response: Response, email: str = Form(...), password: str = Form(...)):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute("SELECT id FROM users WHERE email = ? AND hashed_password = ?", (email, hashed))
    user = cursor.fetchone()
    if user:
        session_token = create_session(user[0])
        resp = RedirectResponse(url="/contracts", status_code=303)
        resp.set_cookie(key="session_token", value=session_token, httponly=True, max_age=3600)
        return resp
    return "<h2 style='color:red;text-align:center;'>Invalid credentials</h2>"

@app.get("/logout", response_class=HTMLResponse)
async def logout(response: Response):
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_token")
    return response

# ===== CONTRACT LIST =====
@app.get("/contracts", response_class=HTMLResponse)
def contracts(request: Request):
    try:
        user_id = get_current_user(request)
        cursor.execute("SELECT company_name FROM users WHERE id = ?", (user_id,))
        company = cursor.fetchone()[0]

        cursor.execute("SELECT contract_id FROM bids WHERE company_name = ?", (company,))
        existing = [row[0] for row in cursor.fetchall()]

        all_contracts = df_bidding.to_dict(orient="records")
        available = [c for i, c in enumerate(all_contracts) if i not in existing]

        return templates.TemplateResponse("contracts_fragment.html", {
            "request": request,
            "contracts": available,
            "user_id": user_id
        })
    except:
        return RedirectResponse(url="/login", status_code=303)

# ===== CONTRACT DETAIL =====
@app.get("/contracts/{contract_id}", response_class=HTMLResponse)
def contract_detail(request: Request, contract_id: int):
    try:
        get_current_user(request)
        row = df_bidding.iloc[contract_id]
        return templates.TemplateResponse("contract_detail.html", {
            "request": request,
            "contract": row.to_dict()
        })
    except:
        return RedirectResponse(url="/login", status_code=303)

# ===== BID SUBMISSION (FULLY FIXED) =====
@app.post("/contracts/{contract_id}/submit_bid", response_class=HTMLResponse)
async def submit_bid(
    request: Request,
    contract_id: int,
    company_name: str = Form(...),
    cac_number: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    bid_amount: str = Form(...),
    equipment_list: str = Form(...),
    workforce: str = Form(...),
):
    try:
        # Get logged-in user
        try:
            user_id = get_current_user(request)
        except:
            user_id = None

        # Override company data from DB
        if user_id:
            cursor.execute("SELECT company_name, cac_number, email FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            company_name, cac_number, email = row

        # Convert bid amount
        try:
            bid_amount_value = float(bid_amount.replace(",", "").strip())
        except:
            return "<h2 style='color:red;text-align:center;'>Invalid bid amount</h2>"

        # AI price check
        contract = df_bidding.iloc[contract_id]
        fair_min, fair_max = get_fair_price_range(contract)
        status_msg = "Approved" if fair_min <= bid_amount_value <= fair_max else "Rejected"

        # Save bid
        cursor.execute("""
            INSERT INTO bids (contract_id, user_id, company_name, cac_number, email, phone, bid_amount, equipment_list, workforce, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (contract_id, user_id, company_name, cac_number, email, phone, bid_amount_value, equipment_list, workforce, status_msg))
        conn.commit()

        bid_id = cursor.lastrowid
        print(f"✓ BID SAVED: {bid_id}")

        # Email
        send_bid_notification(email, company_name, contract['project_name'], status_msg, bid_amount_value)

        return f"<h2 style='color:green;text-align:center;'>Bid submitted successfully! ID: {bid_id}</h2>"

    except Exception as e:
        print("ERROR:", e)
        return "<h2 style='color:red;text-align:center;'>Submission failed</h2>"

# ===== ADMIN LOGIN =====
@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_page(request: Request):
    return """
    <h2>Admin Login</h2>
    <form method='POST'>
        <input name='username' placeholder='Username'>
        <input name='password' placeholder='Password' type='password'>
        <button>Login</button>
    </form>
    """

@app.post("/admin/login", response_class=HTMLResponse)
async def admin_login(username: str = Form(...), password: str = Form(...)):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute("SELECT id FROM admins WHERE username = ? AND hashed_password = ?", (username, hashed))
    admin = cursor.fetchone()
    if admin:
        token = create_session(admin[0])
        resp = RedirectResponse(url="/admin/dashboard", status_code=303)
        resp.set_cookie(key="admin_token", value=token, httponly=True)
        return resp
    return "<h2>Invalid admin credentials</h2>"

# ===== ADMIN DASHBOARD =====
@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    try:
        get_admin_user(request)

        cursor.execute("""
            SELECT id, contract_id, company_name, cac_number, email, phone, bid_amount, equipment_list, workforce, status, timestamp
            FROM bids ORDER BY timestamp DESC
        """)
        bids = cursor.fetchall()

        html = "<h1>Admin Dashboard</h1>"
        html += f"<p>Total bids: {len(bids)}</p>"

        html += "<table border='1' cellpadding='5'>"
        html += "<tr><th>ID</th><th>Contract</th><th>Company</th><th>Email</th><th>Phone</th><th>Amount</th><th>Status</th><th>Time</th></tr>"

        for b in bids:
            contract_name = df_bidding.iloc[b[1]]['project_name']
            html += f"<tr><td>{b[0]}</td><td>{contract_name}</td><td>{b[2]}</td><td>{b[4]}</td><td>{b[5]}</td><td>{b[6]}</td><td>{b[9]}</td><td>{b[10]}</td></tr>"

        html += "</table>"
        return HTMLResponse(html)

    except:
        return RedirectResponse(url="/admin/login", status_code=303)
