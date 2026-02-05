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
import re

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# CORS middleware (NO trailing spaces - CRITICAL FIX)
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

# URLs for datasets (NO trailing spaces - CRITICAL FIX)
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

# Database setup
conn = sqlite3.connect("bids.db", check_same_thread=False)
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
    contract_id INTEGER,
    user_id INTEGER NOT NULL,
    company_name TEXT,
    cac_number TEXT,
    email TEXT,
    phone TEXT,
    bid_amount REAL,
    equipment_list TEXT,
    workforce TEXT,
    status TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
)
""")

# Add admin table
cursor.execute("""
CREATE TABLE IF NOT EXISTS admins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL
)
""")

# Add default admin (run once)
try:
    cursor.execute("INSERT INTO admins (username, hashed_password) VALUES (?, ?)", 
                   ("admin", hashlib.sha256("admin123".encode()).hexdigest()))
    conn.commit()
except sqlite3.IntegrityError:
    pass  # Admin already exists

conn.commit()

# Email function (GMAIL - CORRECTED)
def send_bid_notification(email, company_name, contract_name, status, bid_amount):
    try:
        # Gmail SMTP configuration (MUST use App Password)
        EMAIL_HOST = "smtp.gmail.com"
        EMAIL_PORT = 587
        EMAIL_USER = "aisec2025.notifications@gmail.com"
        EMAIL_PASSWORD = "Qwerasd@()34"  # ← REPLACE WITH 16-CHAR APP PASSWORD
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = email
        msg['Subject'] = f"AISEC Bid Submission - {status}"
        
        body = f"""Dear {company_name},

Your bid for "{contract_name}" has been successfully submitted!

Bid Amount: ₦{bid_amount:.2f} Billion
Status: {status}

You can log in to your AISEC dashboard to view more details.

Thank you for using AISEC - AI for Secure and Efficient Contracting.

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
        print(f"✗ Email FAILED to {email}: {str(e)}")
        return False

# Session management
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

def adjust_for_inflation(base_price, inflation_rate=0.12, years=2):
    return base_price * ((1 + inflation_rate) ** years)

# ===== ROUTES =====
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return RedirectResponse(url="/contracts")

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
async def register_user(company_name: str = Form(...), cac_number: str = Form(...), email: str = Form(...), password: str = Form(...)):
    try:
        hashed = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute("INSERT INTO users (company_name, cac_number, email, hashed_password) VALUES (?, ?, ?, ?)", 
                       (company_name, cac_number, email, hashed))
        conn.commit()
        return "<h2 style='color:green;text-align:center;'>✅ Registration successful!</h2><p style='text-align:center;'><a href='/login'>Login here</a></p>"
    except sqlite3.IntegrityError:
        return "<h2 style='color:red;text-align:center;'>❌ Email already registered</h2><p style='text-align:center;'><a href='/login'>Login here</a></p>"

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
    else:
        return "<h2 style='color:red;text-align:center;'>❌ Invalid credentials</h2><p style='text-align:center;'><a href='/login'>Try again</a></p>"

@app.get("/logout", response_class=HTMLResponse)
async def logout(response: Response):
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_token")
    return response

@app.get("/contracts", response_class=HTMLResponse)
def contracts(request: Request):
    try:
        user_id = get_current_user(request)
        return templates.TemplateResponse("contracts_fragment.html", {
            "request": request, 
            "contracts": df_bidding.to_dict(orient="records"),
            "user_id": user_id
        })
    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)

@app.get("/contracts/{contract_id}", response_class=HTMLResponse)
def contract_detail(request: Request, contract_id: int):
    try:
        user_id = get_current_user(request)
        row = df_bidding.iloc[contract_id]
        return templates.TemplateResponse("contract_detail.html", {
            "request": request, 
            "contract": row.to_dict(),
            "user_id": user_id
        })
    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)

@app.post("/contracts/{contract_id}/submit_bid", response_class=HTMLResponse)
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
):
    try:
        user_id = get_current_user(request)
        bidding_contract = df_bidding.iloc[contract_id]
        
        feature_columns = [
            "award_year", "award_month", "primary_state", "geopolitical_zone",
            "latitude_start", "longitude_start", "estimated_length_km",
            "terrain_type", "rainfall_mm_per_year", "soil_type", "elevation_m",
            "has_bridge", "is_dual_carriageway", "is_rehabilitation", "is_coastal_or_swamp",
            "boq_earthworks_m3_per_km", "boq_asphalt_ton_per_km", "boq_drainage_km_per_km",
            "boq_bridges_units", "boq_culverts_units", "boq_premium_percent"
        ]
        
        features = bidding_contract[feature_columns]
        features_df = pd.DataFrame([features.values], columns=features.index)
        base_price = model.predict(features_df)[0]
        adjusted = adjust_for_inflation(base_price)
        fair_min, fair_max = adjusted * 0.9, adjusted * 1.1
        status_msg = "Approved ✅" if fair_min <= bid_amount <= fair_max else "Rejected ❌"

        cursor.execute("""
        INSERT INTO bids (contract_id, user_id, company_name, cac_number, email, phone, bid_amount, equipment_list, workforce, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (contract_id, user_id, company_name, cac_number, email, phone, bid_amount, equipment_list, workforce, status_msg))
        conn.commit()

        email_sent = send_bid_notification(email, company_name, bidding_contract['project_name'], status_msg, bid_amount)
        email_status = "<p style='color:green;text-align:center;'>📧 Email notification sent!</p>" if email_sent else "<p style='color:#f59e0b;text-align:center;'>⚠️ Bid submitted (email failed)</p>"

        return f"""
        <div style='max-width:600px;margin:30px auto;background:#f0fdf4;border:2px solid #10b981;border-radius:12px;padding:25px;text-align:center;box-shadow:0 4px 12px rgba(16,185,129,0.2)'>
            <div style='width:60px;height:60px;background:#10b981;border-radius:50%;margin:0 auto 15px;display:flex;align-items:center;justify-content:center;color:white;font-size:28px'>✓</div>
            <h2 style='color:#065f46;margin:0 0 15px'>Bid Submitted Successfully!</h2>
            <div style='background:white;padding:15px;border-radius:8px;margin:15px 0;text-align:left'>
                <p><strong>Contract:</strong> {bidding_contract['project_name']}</p>
                <p><strong>Company:</strong> {company_name}</p>
                <p><strong>Bid Amount:</strong> ₦{bid_amount:.2f} Billion</p>
                <p style='font-weight:bold;color:{'#10b981' if 'Approved' in status_msg else '#ef4444'}'><strong>Status:</strong> {status_msg}</p>
            </div>
            {email_status}
            <a href='/contracts' style='display:inline-block;margin-top:20px;padding:12px 30px;background:#2563eb;color:white;text-decoration:none;border-radius:8px;font-weight:600'>View More Contracts</a>
        </div>
        """
        
    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)
    except Exception as e:
        error_msg = "Submission failed. Please try again."
        if "FOREIGN KEY" in str(e):
            error_msg = "Invalid contract selection."
        elif "UNIQUE" in str(e):
            error_msg = "This bid was already submitted."
        
        return f"""
        <div style='max-width:600px;margin:30px auto;background:#fef2f2;border:2px solid #ef4444;border-radius:12px;padding:25px;text-align:center;box-shadow:0 4px 12px rgba(239,68,68,0.2)'>
            <div style='width:60px;height:60px;background:#ef4444;border-radius:50%;margin:0 auto 15px;display:flex;align-items:center;justify-content:center;color:white;font-size:28px'>!</div>
            <h2 style='color:#991b1b;margin:0 0 15px'>Bid Submission Failed</h2>
            <p style='color:#7f1d1d;margin:15px 0'>{error_msg}</p>
            <a href='/contracts' style='display:inline-block;margin-top:20px;padding:12px 30px;background:#2563eb;color:white;text-decoration:none;border-radius:8px;font-weight:600'>Try Again</a>
        </div>
        """

@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_page(request: Request):
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Admin Login - AISEC</title></head>
    <body style="font-family:Arial,sans-serif;background:#f0f9ff;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0">
        <div style="background:white;padding:40px;border-radius:16px;box-shadow:0 10px 30px rgba(0,0,0,0.1);width:100%;max-width:400px;text-align:center">
            <div style="font-size:48px;margin-bottom:20px">🛡️</div>
            <h2 style="color:#1e40af;margin-bottom:30px">AISEC Admin Portal</h2>
            <form method="POST" style="display:flex;flex-direction:column;gap:15px">
                <input type="text" name="username" placeholder="Username" required style="padding:12px;border:1px solid #ddd;border-radius:8px;font-size:16px">
                <input type="password" name="password" placeholder="Password" required style="padding:12px;border:1px solid #ddd;border-radius:8px;font-size:16px">
                <button type="submit" style="padding:12px;background:#2563eb;color:white;border:none;border-radius:8px;font-size:16px;font-weight:600;cursor:pointer">Login to Dashboard</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.post("/admin/login", response_class=HTMLResponse)
async def admin_login(username: str = Form(...), password: str = Form(...)):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute("SELECT id FROM admins WHERE username = ? AND hashed_password = ?", (username, hashed))
    admin = cursor.fetchone()
    if admin:
        session_token = create_session(admin[0])
        resp = RedirectResponse(url="/admin/dashboard", status_code=303)
        resp.set_cookie(key="admin_token", value=session_token, httponly=True, max_age=3600)
        return resp
    else:
        return "<h2 style='color:red;text-align:center'>❌ Invalid credentials</h2><p style='text-align:center'><a href='/admin/login' style='color:#2563eb;text-decoration:none'>Try again</a></p>"
@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    try:
        admin_id = get_admin_user(request)
        
        # CORRECT QUERY (NO JOIN - matches your actual database)
        cursor.execute("""
            SELECT id, contract_id, user_id, company_name, cac_number, email, 
                   phone, bid_amount, equipment_list, workforce, status, timestamp
            FROM bids 
            ORDER BY timestamp DESC
        """)
        bids = cursor.fetchall()
        
        # Prepare enhanced bid data with AI predictions
        enhanced_bids = []
        for bid in bids:
            contract_id = bid[1]  # contract_id at index 1
            try:
                contract_row = df_bidding.iloc[contract_id]
                contract_name = contract_row['project_name']
                
                # Recalculate AI prediction
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
                fair_min = adjusted * 0.9
                fair_max = adjusted * 1.1
                
                # CORRECT INDEXING:
                bid_amount = bid[7]  # bid_amount at index 7
                is_fair = fair_min <= bid_amount <= fair_max
                
                enhanced_bids.append({
                    'contract_name': contract_name,
                    'company_name': bid[3] or 'N/A',  # company_name at index 3
                    'cac_number': bid[4] or 'N/A',   # cac_number at index 4
                    'bid_amount': bid_amount,
                    'fair_min': fair_min,
                    'fair_max': fair_max,
                    'is_fair': is_fair,
                    'status': bid[10],               # status at index 10
                    'timestamp': bid[11],             # timestamp at index 11
                    'email': bid[5]                   # email at index 5
                })
            except Exception as e:
                print(f"Error processing bid {bid[0]}: {str(e)}")
                enhanced_bids.append({
                    'contract_name': f'Contract ID {contract_id} (Error)',
                    'company_name': bid[3] or 'N/A',
                    'cac_number': bid[4] or 'N/A',
                    'bid_amount': bid[7],
                    'fair_min': 0,
                    'fair_max': 0,
                    'is_fair': False,
                    'status': bid[10],
                    'timestamp': bid[11],
                    'email': bid[5]
                })
        
        # Build enhanced HTML dashboard (same as before)
        admin_html = """..."""  # Keep your existing HTML template
        return admin_html
        
    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)

        
        # Prepare enhanced bid data with AI predictions
        enhanced_bids = []
        for bid in bids:
            contract_id = bid[1]  # contract_id at index 1
            try:
                contract_row = df_bidding.iloc[contract_id]
                contract_name = contract_row['project_name']
                
                # Recalculate AI prediction
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
                fair_min = adjusted * 0.9
                fair_max = adjusted * 1.1
                
                # CORRECT INDEXING (critical fix):
                bid_amount = bid[6]  # bid_amount is at index 6 in this query
                is_fair = fair_min <= bid_amount <= fair_max
                
                enhanced_bids.append({
                    'contract_name': contract_name,
                    'company_name': bid[2] or 'N/A',  # company_name at index 2
                    'cac_number': bid[3] or 'N/A',    # cac_number at index 3
                    'bid_amount': bid_amount,
                    'fair_min': fair_min,
                    'fair_max': fair_max,
                    'is_fair': is_fair,
                    'status': bid[9],                 # status at index 9
                    'timestamp': bid[10],             # timestamp at index 10
                    'email': bid[4]                   # email at index 4
                })
            except Exception as e:
                print(f"Error processing bid {bid[0]}: {str(e)}")
                enhanced_bids.append({
                    'contract_name': f'Contract ID {contract_id} (Error)',
                    'company_name': bid[2] or 'N/A',
                    'cac_number': bid[3] or 'N/A',
                    'bid_amount': bid[6],
                    'fair_min': 0,
                    'fair_max': 0,
                    'is_fair': False,
                    'status': bid[9],
                    'timestamp': bid[10],
                    'email': bid[4]
                })
        
        # Build enhanced HTML dashboard (same as before - no changes needed here)
        admin_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AISEC Admin Dashboard</title>
            <style>
                :root { --primary: #2563eb; --success: #10b981; --warning: #f59e0b; --danger: #ef4444; }
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f8fafc; margin: 0; }
                .header { background: linear-gradient(135deg, #1e40af, #0c4a6e); color: white; padding: 20px 40px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
                .container { max-width: 1800px; margin: 30px auto; padding: 0 20px; }
                .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin-bottom: 30px; }
                .stat-card { background: white; border-radius: 16px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); text-align: center; border-top: 4px solid var(--primary); }
                .stat-value { font-size: 36px; font-weight: 800; margin: 10px 0; color: var(--primary); }
                .stat-label { color: #475569; font-size: 16px; }
                .table-container { background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-top: 20px; }
                table { width: 100%; border-collapse: collapse; }
                th { background: linear-gradient(135deg, #f8fafc, #e2e8f0); padding: 16px 12px; text-align: left; font-weight: 700; color: #1e40af; font-size: 14px; position: sticky; top: 0; }
                td { padding: 14px 12px; border-bottom: 1px solid #f1f5f9; font-size: 14px; }
                tr:hover { background: #f8fafc; }
                .fair { background: #f0fdf4; border-left: 4px solid var(--success); }
                .unfair { background: #fff7ed; border-left: 4px solid var(--warning); }
                .status-approved { color: var(--success); font-weight: 600; }
                .status-rejected { color: var(--danger); font-weight: 600; }
                .range-good { color: var(--success); font-weight: 600; }
                .range-bad { color: var(--danger); font-weight: 600; }
                .logout-btn { background: var(--danger); color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; font-weight: 600; }
                .logo { font-size: 28px; font-weight: bold; display: flex; align-items: center; gap: 12px; }
                .highlight { background: #dbeafe; padding: 2px 8px; border-radius: 6px; font-weight: 600; }
            </style>
        </head>
        <body>
            <div class="header">
                <div class="logo">🛡️ AISEC ADMIN DASHBOARD</div>
                <a href="/admin/logout" class="logout-btn">Logout</a>
            </div>
            
            <div class="container">
                <h1 style="color: #1e293b; margin: 20px 0 30px; font-size: 32px;">📊 Bid Analysis & AI Comparison</h1>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">""" + str(len(enhanced_bids)) + """</div>
                        <div class="stat-label">Total Bids</div>
                    </div>
                    <div class="stat-card" style="border-top-color: #10b981;">
                        <div class="stat-value">""" + str(sum(1 for b in enhanced_bids if b['is_fair'])) + """</div>
                        <div class="stat-label">Fair Bids (AI Approved)</div>
                    </div>
                    <div class="stat-card" style="border-top-color: #f59e0b;">
                        <div class="stat-value">""" + str(sum(1 for b in enhanced_bids if not b['is_fair'])) + """</div>
                        <div class="stat-label">Inflated Bids (AI Flagged)</div>
                    </div>
                    <div class="stat-card" style="border-top-color: #0ea5e9;">
                        <div class="stat-value">₦""" + f"{sum(b['bid_amount'] for b in enhanced_bids):.2f}" + """B</div>
                        <div class="stat-label">Total Bid Value</div>
                    </div>
                </div>
                
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Contract Name</th>
                                <th>Company / CAC</th>
                                <th>Email</th>
                                <th>Bid Amount (₦B)</th>
                                <th>AI Fair Range (₦B)</th>
                                <th>AI Assessment</th>
                                <th>Status</th>
                                <th>Submitted</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for bid in enhanced_bids:
            row_class = "fair" if bid['is_fair'] else "unfair"
            ai_assessment = "✅ FAIR PRICE" if bid['is_fair'] else "⚠️ INFLATED"
            ai_class = "range-good" if bid['is_fair'] else "range-bad"
            status_class = "status-approved" if "Approved" in str(bid['status']) else "status-rejected"
            
            admin_html += f"""
            <tr class="{row_class}">
                <td><strong>{bid['contract_name']}</strong></td>
                <td>
                    <div>{bid['company_name']}</div>
                    <div style="color: #64748b; font-size: 13px;">CAC: {bid['cac_number']}</div>
                </td>
                <td>{bid['email']}</td>
                <td><strong>₦{bid['bid_amount']:.2f}</strong></td>
                <td><span class="{ai_class}">₦{bid['fair_min']:.2f} - ₦{bid['fair_max']:.2f}</span></td>
                <td><span class="{ai_class}">{ai_assessment}</span></td>
                <td><span class="{status_class}">{bid['status']}</span></td>
                <td style="color: #64748b; font-size: 13px;">{bid['timestamp']}</td>
            </tr>
            """
        
        admin_html += """
                        </tbody>
                    </table>
                </div>
                
                <div style="background: white; border-radius: 16px; padding: 25px; margin-top: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
                    <h2 style="color: #1e40af; margin-top: 0;">🔍 How AI Assessment Works</h2>
                    <ul style="line-height: 1.8; color: #475569; padding-left: 25px;">
                        <li><span class="highlight">Green rows</span> = Bid amount falls within AI's calculated fair price range (±10% of inflation-adjusted prediction)</li>
                        <li><span class="highlight">Orange rows</span> = Bid amount exceeds AI's fair range (potential fraud indicator)</li>
                        <li>AI considers: Terrain type, location coordinates, soil conditions, bridge requirements, historical pricing, and inflation</li>
                        <li>All predictions use the same model that evaluated bids at submission time</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        return admin_html
        
    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)
        
        admin_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AISEC Admin Dashboard</title>
            <style>
                :root { --primary: #2563eb; --success: #10b981; --warning: #f59e0b; --danger: #ef4444; }
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f8fafc; margin: 0; }
                .header { background: linear-gradient(135deg, #1e40af, #0c4a6e); color: white; padding: 20px 40px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
                .container { max-width: 1800px; margin: 30px auto; padding: 0 20px; }
                .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin-bottom: 30px; }
                .stat-card { background: white; border-radius: 16px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); text-align: center; border-top: 4px solid var(--primary); }
                .stat-value { font-size: 36px; font-weight: 800; margin: 10px 0; color: var(--primary); }
                .stat-label { color: #475569; font-size: 16px; }
                .table-container { background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-top: 20px; }
                table { width: 100%; border-collapse: collapse; }
                th { background: linear-gradient(135deg, #f8fafc, #e2e8f0); padding: 16px 12px; text-align: left; font-weight: 700; color: #1e40af; font-size: 14px; position: sticky; top: 0; }
                td { padding: 14px 12px; border-bottom: 1px solid #f1f5f9; font-size: 14px; }
                tr:hover { background: #f8fafc; }
                .fair { background: #f0fdf4; border-left: 4px solid var(--success); }
                .unfair { background: #fff7ed; border-left: 4px solid var(--warning); }
                .status-approved { color: var(--success); font-weight: 600; }
                .status-rejected { color: var(--danger); font-weight: 600; }
                .range-good { color: var(--success); font-weight: 600; }
                .range-bad { color: var(--danger); font-weight: 600; }
                .logout-btn { background: var(--danger); color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; font-weight: 600; }
                .logo { font-size: 28px; font-weight: bold; display: flex; align-items: center; gap: 12px; }
                .highlight { background: #dbeafe; padding: 2px 8px; border-radius: 6px; font-weight: 600; }
            </style>
        </head>
        <body>
            <div class="header">
                <div class="logo">🛡️ AISEC ADMIN DASHBOARD</div>
                <a href="/admin/logout" class="logout-btn">Logout</a>
            </div>
            
            <div class="container">
                <h1 style="color: #1e293b; margin: 20px 0 30px; font-size: 32px;">📊 Bid Analysis & AI Comparison</h1>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">""" + str(len(enhanced_bids)) + """</div>
                        <div class="stat-label">Total Bids</div>
                    </div>
                    <div class="stat-card" style="border-top-color: #10b981;">
                        <div class="stat-value">""" + str(sum(1 for b in enhanced_bids if b['is_fair'])) + """</div>
                        <div class="stat-label">Fair Bids (AI Approved)</div>
                    </div>
                    <div class="stat-card" style="border-top-color: #f59e0b;">
                        <div class="stat-value">""" + str(sum(1 for b in enhanced_bids if not b['is_fair'])) + """</div>
                        <div class="stat-label">Inflated Bids (AI Flagged)</div>
                    </div>
                    <div class="stat-card" style="border-top-color: #0ea5e9;">
                        <div class="stat-value">₦""" + f"{sum(b['bid_amount'] for b in enhanced_bids):.2f}" + """B</div>
                        <div class="stat-label">Total Bid Value</div>
                    </div>
                </div>
                
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Contract Name</th>
                                <th>Company / CAC</th>
                                <th>Email</th>
                                <th>Bid Amount (₦B)</th>
                                <th>AI Fair Range (₦B)</th>
                                <th>AI Assessment</th>
                                <th>Status</th>
                                <th>Submitted</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for bid in enhanced_bids:
            row_class = "fair" if bid['is_fair'] else "unfair"
            ai_assessment = "✅ FAIR PRICE" if bid['is_fair'] else "⚠️ INFLATED"
            ai_class = "range-good" if bid['is_fair'] else "range-bad"
            status_class = "status-approved" if "Approved" in str(bid['status']) else "status-rejected"
            
            admin_html += f"""
            <tr class="{row_class}">
                <td><strong>{bid['contract_name']}</strong></td>
                <td>
                    <div>{bid['company_name']}</div>
                    <div style="color: #64748b; font-size: 13px;">CAC: {bid['cac_number']}</div>
                </td>
                <td>{bid['email']}</td>
                <td><strong>₦{bid['bid_amount']:.2f}</strong></td>
                <td><span class="{ai_class}">₦{bid['fair_min']:.2f} - ₦{bid['fair_max']:.2f}</span></td>
                <td><span class="{ai_class}">{ai_assessment}</span></td>
                <td><span class="{status_class}">{bid['status']}</span></td>
                <td style="color: #64748b; font-size: 13px;">{bid['timestamp']}</td>
            </tr>
            """
        
        admin_html += """
                        </tbody>
                    </table>
                </div>
                
                <div style="background: white; border-radius: 16px; padding: 25px; margin-top: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
                    <h2 style="color: #1e40af; margin-top: 0;">🔍 How AI Assessment Works</h2>
                    <ul style="line-height: 1.8; color: #475569; padding-left: 25px;">
                        <li><span class="highlight">Green rows</span> = Bid amount falls within AI's calculated fair price range (±10% of inflation-adjusted prediction)</li>
                        <li><span class="highlight">Orange rows</span> = Bid amount exceeds AI's fair range (potential fraud indicator)</li>
                        <li>AI considers: Terrain type, location coordinates, soil conditions, bridge requirements, historical pricing, and inflation</li>
                        <li>All predictions use the same model that evaluated bids at submission time</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        return admin_html
        
    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)

@app.get("/admin/logout", response_class=HTMLResponse)
async def admin_logout(response: Response):
    response = RedirectResponse(url="/admin/login", status_code=303)
    response.delete_cookie("admin_token")
    return response




