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

# CORS middleware (NO TRAILING SPACES - CRITICAL)
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

# URLs for datasets (NO TRAILING SPACES - CRITICAL)
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

# Database setup (CORRECTED SCHEMA - NO FOREIGN KEY CONSTRAINT)
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
    contract_id INTEGER NOT NULL,  -- STORES INTEGER INDEX FROM DATAFRAME
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

# Email function (UPDATE CREDENTIALS BEFORE DEPLOYING)
def send_bid_notification(email, company_name, contract_name, status, bid_amount):
    try:
        EMAIL_HOST = "smtp.gmail.com"
        EMAIL_PORT = 587
        EMAIL_USER = "aisec2025.notifications@gmail.com"  # ← REPLACE WITH YOUR GMAIL
        EMAIL_PASSWORD = "YOUR_16_CHAR_APP_PASSWORD"      # ← GET FROM GOOGLE ACCOUNT SECURITY
        
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
        print(f"✗ Email FAILED: {str(e)}")
        return False

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

# CRITICAL FIX: REMOVED ALL TRAILING SPACES IN FEATURE COLUMNS
def get_fair_price_range(contract_row):
    feature_columns = [
        "award_year", "award_month", "primary_state", "geopolitical_zone",
        "latitude_start", "longitude_start", "estimated_length_km",
        "terrain_type", "rainfall_mm_per_year", "soil_type", "elevation_m",
        "has_bridge", "is_dual_carriageway", "is_rehabilitation", "is_coastal_or_swamp",
        "boq_earthworks_m3_per_km", "boq_asphalt_ton_per_km", "boq_drainage_km_per_km",
        "boq_bridges_units", "boq_culverts_units", "boq_premium_percent"
    ]  # ← NO TRAILING SPACES (CRITICAL FIX)
    features = contract_row[feature_columns]
    features_df = pd.DataFrame([features.values], columns=features.index)
    base_price = model.predict(features_df)[0]
    adjusted = adjust_for_inflation(base_price)
    return adjusted * 0.9, adjusted * 1.1

# ===== CORE ROUTES =====
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

# ===== CONTRACTS WITH FILTERING (CRITICAL FIX) =====
@app.get("/contracts", response_class=HTMLResponse)
def contracts(request: Request):
    try:
        user_id = get_current_user(request)
        cursor.execute("SELECT company_name FROM users WHERE id = ?", (user_id,))
        user_data = cursor.fetchone()
        if not user_data:
            return RedirectResponse(url="/login", status_code=303)
        
        current_company = user_data[0]
        # CRITICAL FIX: Query uses INTEGER contract_id (matches dataframe index)
        cursor.execute("SELECT contract_id FROM bids WHERE company_name = ?", (current_company,))
        existing_bids = cursor.fetchall()
        bid_contract_ids = [bid[0] for bid in existing_bids]  # List of INTEGER indices
        
        all_contracts = df_bidding.to_dict(orient="records")
        # CRITICAL FIX: Filter using INTEGER index comparison
        available_contracts = [
            contract for idx, contract in enumerate(all_contracts) 
            if idx not in bid_contract_ids
        ]
        
        if not available_contracts:
            return """
            <div style='max-width:700px;margin:50px auto;background:white;border-radius:16px;padding:40px;text-align:center;box-shadow:0 5px 20px rgba(0,0,0,0.1)'>
                <div style='font-size:64px;margin-bottom:20px'>✅</div>
                <h2 style='color:#1e40af;margin-bottom:15px'>All Contracts Bid Successfully!</h2>
                <p style='color:#475569;font-size:18px;margin-bottom:25px'>
                    Your company has submitted bids for all available contracts.<br>
                    Administrators will review your submissions shortly.
                </p>
                <div style='background:#f0fdf4;border-radius:12px;padding:20px;margin:25px 0;text-align:left'>
                    <p style='font-weight:600;color:#065f46;margin-bottom:10px'>📌 What's Next:</p>
                    <ul style='color:#065f46;line-height:1.8;text-align:left;padding-left:20px'>
                        <li>Monitor your email for status updates</li>
                        <li>Admins may contact you for clarification</li>
                        <li>Check admin dashboard for AI assessment results</li>
                    </ul>
                </div>
                <a href='/logout' style='display:inline-block;padding:12px 30px;background:#ef4444;color:white;text-decoration:none;border-radius:8px;font-weight:600'>
                    Logout
                </a>
            </div>
            """
        
        return templates.TemplateResponse("contracts_fragment.html", {
            "request": request, 
            "contracts": available_contracts,
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

# ===== BID SUBMISSION WITH VISIBLE SUCCESS & PERSISTENCE (CRITICAL FIX) =====
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
        # Get current user (if logged in)
        try:
            user_id = get_current_user(request)
        except HTTPException:
            user_id = None

        # 🔥 If user is logged in, always use company data from the database
        if user_id is not None:
            cursor.execute(
                "SELECT company_name, cac_number, email FROM users WHERE id = ?",
                (user_id,)
            )
            user_row = cursor.fetchone()
            if user_row:
                company_name = user_row[0]
                cac_number = user_row[1]
                email = user_row[2]

        # 🔢 Safely convert bid_amount to float
        try:
            clean_bid_amount = bid_amount.replace(",", "").strip()
            bid_amount_value = float(clean_bid_amount)
        except ValueError:
            return """
            <div style='max-width:650px;margin:30px auto;background:#fef2f2;border:3px solid #ef4444;border-radius:20px;padding:30px;text-align:center;box-shadow:0 10px 30px rgba(239, 68, 68, 0.25)'>
                <h1 style='color:#991b1b;margin-bottom:15px;font-size:24px'>❌ Invalid Bid Amount</h1>
                <p style='color:#991b1b;font-size:16px;margin-bottom:20px'>
                    Please enter a valid numeric bid amount (e.g. 10.5).
                </p>
                <a href='/contracts' style='display:inline-block;margin-top:15px;padding:12px 30px;background:#1e40af;color:white;text-decoration:none;border-radius:10px;font-weight:600;font-size:16px'>
                    ⇦ Go Back & Retry
                </a>
            </div>
            """

        bidding_contract = df_bidding.iloc[contract_id]
        fair_min, fair_max = get_fair_price_range(bidding_contract)
        status_msg = "Approved ✅" if fair_min <= bid_amount_value <= fair_max else "Rejected ❌"

        # CRITICAL FIX: Save bid with EXPLICIT COMMIT (persists to disk)
        cursor.execute("""
        INSERT INTO bids (contract_id, user_id, company_name, cac_number, email, phone, bid_amount, equipment_list, workforce, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (contract_id, user_id, company_name, cac_number, email, phone, bid_amount_value, equipment_list, workforce, status_msg))
        conn.commit() 
        
        bid_id = cursor.lastrowid
        print(f"✓✓✓ BID SAVED SUCCESSFULLY! ID:{bid_id} Contract:{contract_id} Amount:₦{bid_amount_value:.2f}B")

        # Send email notification (non-blocking)
        try:
            send_bid_notification(email, company_name, bidding_contract['project_name'], status_msg, bid_amount_value)
            email_status = "<p style='color:#10b981;font-weight:600;margin:15px 0;font-size:18px'>📧 Email confirmation sent!</p>"
        except Exception as e:
            print(f"⚠️ Email failed (bid saved): {str(e)}")
            email_status = "<p style='color:#f59e0b;font-weight:600;margin:15px 0;font-size:18px'>⚠️ Bid saved (email failed)</p>"

        # VISIBLE SUCCESS MESSAGE (USER WILL SEE THIS IMMEDIATELY)
        return f"""
        <div style='max-width:750px;margin:30px auto;background:linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);border:3px solid #10b981;border-radius:20px;padding:30px;text-align:center;box-shadow:0 10px 30px rgba(16, 185, 129, 0.25);animation:fadeIn 0.6s'>
            <style>@keyframes fadeIn {{ from {{ opacity:0; transform: translateY(20px); }} to {{ opacity:1; transform: translateY(0); }} }}</style>
            <div style='width:70px;height:70px;background:#10b981;border-radius:50%;margin:0 auto 20px;display:flex;align-items:center;justify-content:center;color:white;font-size:32px'>✓</div>
            <h1 style='color:#065f46;margin:0 0 12px;font-size:28px'>🎉 BID SUBMITTED SUCCESSFULLY!</h1>
            <p style='color:#0f766e;font-size:18px;margin-bottom:20px'>Your bid has been recorded in the AISEC system</p>
            
            <div style='background:white;padding:20px;border-radius:16px;margin:20px 0;box-shadow:0 4px 12px rgba(0,0,0,0.08);text-align:left'>
                <p style='margin:10px 0'><strong>📝 Contract:</strong> <span style='color:#1e40af;font-weight:600'>{bidding_contract['project_name']}</span></p>
                <p style='margin:10px 0'><strong>🏢 Company:</strong> {company_name}</p>
                <p style='margin:10px 0'><strong>🆔 CAC Number:</strong> {cac_number}</p>
                <p style='margin:10px 0'><strong>💰 Bid Amount:</strong> <span style='font-size:20px;font-weight:bold;color:#065f46'>₦{bid_amount_value:.2f} Billion</span></p>
                <p style='margin:10px 0;font-weight:bold;color:{'#10b981' if 'Approved' in status_msg else '#ef4444'};font-size:17px'>
                    <strong>📊 AI Assessment:</strong> {status_msg}
                </p>
                <div style='background:#f0fdf4;border-left:3px solid #10b981;padding:10px;margin-top:15px;font-size:15px;color:#065f46'>
                    <strong>🔖 Bid ID:</strong> {bid_id} • <strong>⏰ Submitted:</strong> {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
                </div>
            </div>
            
            {email_status}
            
            <div style='background:#dbeafe;padding:15px;border-radius:12px;margin:20px 0;text-align:left'>
                <p style='margin:8px 0;color:#1e40af;font-weight:600;font-size:16px'>✅ Next Steps:</p>
                <ul style='text-align:left;margin-left:20px;color:#1e40af;line-height:1.7;font-size:15px'>
                    <li>Your bid is <strong>visible to administrators</strong> in the AISEC dashboard</li>
                    <li>AI has compared your bid against fair market pricing</li>
                    <li>Admins may contact you using the email/phone provided</li>
                    <li>This contract will <strong>no longer appear</strong> in your available contracts list</li>
                </ul>
            </div>
            
            <a href='/contracts' style='display:inline-block;margin-top:15px;padding:14px 40px;background:linear-gradient(135deg, #1e40af, #1e3a8a);color:white;text-decoration:none;border-radius:12px;font-weight:700;font-size:17px;box-shadow:0 4px 12px rgba(30, 64, 175, 0.3);transition:all 0.3s'>
                📋 View Remaining Contracts
            </a>
        </div>
        """
        
    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)
    except Exception as e:
        error_detail = str(e)
        print(f"✗✗✗ BID SUBMISSION FAILED: {error_detail}")
        if "no such table" in error_detail:
            error_msg = "Database error. Contact administrator."
        elif "FOREIGN KEY" in error_detail:
            error_msg = "Session expired. Please login again."
        else:
            error_msg = "Submission failed. Please try again."
        
        return f"""
        <div style='max-width:650px;margin:30px auto;background:#fef2f2;border:3px solid #ef4444;border-radius:20px;padding:30px;text-align:center;box-shadow:0 10px 30px rgba(239, 68, 68, 0.25)'>
            <div style='width:70px;height:70px;background:#ef4444;border-radius:50%;margin:0 auto 20px;display:flex;align-items:center;justify-content:center;color:white;font-size:32px'>!</div>
            <h1 style='color:#991b1b;margin:0 0 12px;font-size:28px'>❌ SUBMISSION FAILED</h1>
            <p style='color:#991b1b;font-size:18px;margin-bottom:20px'>{error_msg}</p>
            <div style='background:#fee2e2;border-radius:10px;padding:15px;margin:15px 0;font-family:monospace;font-size:14px;color:#b91c1c;max-height:100px;overflow:auto;text-align:left'>
                {error_detail[:150]}
            </div>
            <a href='/contracts' style='display:inline-block;margin-top:15px;padding:14px 35px;background:#1e40af;color:white;text-decoration:none;border-radius:10px;font-weight:600;font-size:16px'>
                ⇦ Go Back & Retry
            </a>
        </div>
        """

# ===== ADMIN ROUTES (SINGLE DEFINITION) =====
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
        
        # CORRECT QUERY: NO JOIN, EXPLICIT COLUMNS, INTEGER contract_id
        cursor.execute("""
            SELECT id, contract_id, company_name, cac_number, email, phone, 
                   bid_amount, equipment_list, workforce, status, timestamp
            FROM bids 
            ORDER BY timestamp DESC
        """)
        bids = cursor.fetchall()
        total_bids = len(bids)
        print(f"✓✓✓ ADMIN DASHBOARD: Loaded {total_bids} bids from database")
        
        # Process bids with AI comparison (using INTEGER contract_id)
        enhanced_bids = []
        for bid in bids:
            try:
                # CRITICAL FIX: Use INTEGER contract_id to index dataframe
                contract_row = df_bidding.iloc[bid[1]]  # bid[1] = contract_id (INTEGER)
                fair_min, fair_max = get_fair_price_range(contract_row)
                is_fair = fair_min <= bid[6] <= fair_max  # bid[6] = bid_amount
                
                enhanced_bids.append({
                    'contract_name': contract_row['project_name'],
                    'company_name': bid[2] or 'N/A',
                    'cac_number': bid[3] or 'N/A',
                    'bid_amount': bid[6],
                    'fair_min': fair_min,
                    'fair_max': fair_max,
                    'is_fair': is_fair,
                    'status': bid[9],
                    'timestamp': bid[10],
                    'email': bid[4],
                    'phone': bid[5],
                    'bid_id': bid[0]
                })
            except Exception as e:
                print(f"⚠️ Error processing bid ID {bid[0]}: {str(e)}")
                enhanced_bids.append({
                    'contract_name': f'Contract ID {bid[1]} (Error)',
                    'company_name': bid[2] or 'N/A',
                    'cac_number': bid[3] or 'N/A',
                    'bid_amount': bid[6],
                    'fair_min': 0,
                    'fair_max': 0,
                    'is_fair': False,
                    'status': bid[9],
                    'timestamp': bid[10],
                    'email': bid[4],
                    'phone': bid[5],
                    'bid_id': bid[0]
                })
        
        # Build dashboard HTML
        admin_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AISEC Admin Dashboard</title>
            <style>
                :root {{ --primary: #2563eb; --success: #10b981; --warning: #f59e0b; --danger: #ef4444; }}
                * {{ margin:0; padding:0; box-sizing:border-box; }}
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f9ff; }}
                .header {{ background: linear-gradient(135deg, #1e40af, #0c4a6e); color: white; padding: 20px 40px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15); position: sticky; top: 0; z-index: 100; }}
                .container {{ max-width: 1800px; margin: 30px auto; padding: 0 20px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                .stat-card {{ background: white; border-radius: 20px; padding: 25px; box-shadow: 0 6px 20px rgba(0,0,0,0.08); text-align: center; border-top: 5px solid var(--primary); transition: transform 0.3s; }}
                .stat-card:hover {{ transform: translateY(-5px); }}
                .stat-value {{ font-size: 42px; font-weight: 800; margin: 10px 0; background: linear-gradient(135deg, var(--primary), #0ea5e9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
                .stat-label {{ color: #334155; font-size: 16px; font-weight: 600; }}
                .table-container {{ background: white; border-radius: 20px; overflow: hidden; box-shadow: 0 10px 40px rgba(0,0,0,0.12); margin-top: 10px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th {{ background: linear-gradient(135deg, #f8fafc, #e2e8f0); padding: 18px 15px; text-align: left; font-weight: 800; color: #1e40af; font-size: 15px; position: sticky; top: 70px; z-index: 90; }}
                td {{ padding: 16px 15px; border-bottom: 1px solid #f1f5f9; font-size: 15px; }}
                tr:hover {{ background: #f8fafc; }}
                .fair {{ background: linear-gradient(to right, #f0fdf4 95%, #bbf7d0 5%); border-left: 5px solid var(--success); }}
                .unfair {{ background: linear-gradient(to right, #fff7ed 95%, #fcd34d 5%); border-left: 5px solid var(--warning); }}
                .status-approved {{ color: var(--success); font-weight: 700; font-size: 16px; }}
                .status-rejected {{ color: var(--danger); font-weight: 700; font-size: 16px; }}
                .range-good {{ color: var(--success); font-weight: 700; }}
                .range-bad {{ color: var(--danger); font-weight: 700; }}
                .logout-btn {{ background: linear-gradient(135deg, var(--danger), #b91c1c); color: white; padding: 12px 28px; border-radius: 12px; text-decoration: none; font-weight: 700; font-size: 16px; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4); transition: all 0.3s; }}
                .logout-btn:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(239, 68, 68, 0.6); }}
                .logo {{ font-size: 28px; font-weight: 800; display: flex; align-items: center; gap: 12px; }}
                .highlight {{ background: #dbeafe; padding: 3px 10px; border-radius: 8px; font-weight: 700; }}
                .empty-state {{ text-align: center; padding: 60px 20px; color: #64748b; }}
                .empty-state i {{ font-size: 64px; margin-bottom: 20px; opacity: 0.3; }}
                .empty-state h3 {{ font-size: 28px; margin: 15px 0; color: #334155; }}
                .timestamp {{ color: #475569; font-family: monospace; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="logo">🛡️ AISEC ADMIN DASHBOARD</div>
                <a href="/admin/logout" class="logout-btn">🚪 Logout</a>
            </div>
            
            <div class="container">
                <h1 style="color: #0f172a; margin: 25px 0 30px; font-size: 36px; font-weight: 800;">📊 Real-Time Bid Analysis & AI Comparison</h1>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div style="font-size: 24px; margin-bottom: 8px">📋</div>
                        <div class="stat-value">{total_bids}</div>
                        <div class="stat-label">TOTAL BIDS SUBMITTED</div>
                    </div>
                    <div class="stat-card" style="border-top-color: #10b981;">
                        <div style="font-size: 24px; margin-bottom: 8px">✅</div>
                        <div class="stat-value">{sum(1 for b in enhanced_bids if b['is_fair'])}</div>
                        <div class="stat-label">FAIR BIDS (AI APPROVED)</div>
                    </div>
                    <div class="stat-card" style="border-top-color: #f59e0b;">
                        <div style="font-size: 24px; margin-bottom: 8px">⚠️</div>
                        <div class="stat-value">{sum(1 for b in enhanced_bids if not b['is_fair'])}</div>
                        <div class="stat-label">INFLATED BIDS (AI FLAGGED)</div>
                    </div>
                    <div class="stat-card" style="border-top-color: #0ea5e9;">
                        <div style="font-size: 24px; margin-bottom: 8px">💰</div>
                        <div class="stat-value">₦{sum(b['bid_amount'] for b in enhanced_bids):.2f}B</div>
                        <div class="stat-label">TOTAL BID VALUE</div>
                    </div>
                </div>
                
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Bid ID</th>
                                <th>Contract Name</th>
                                <th>Company / CAC</th>
                                <th>Contact</th>
                                <th>Bid Amount (₦B)</th>
                                <th>AI Fair Range (₦B)</th>
                                <th>AI Assessment</th>
                                <th>Submitted</th>
                            </tr>
                        </thead>
                        <tbody>
        """

        if not enhanced_bids:
            admin_html += """
                <tr>
                    <td colspan="8" class="empty-state">
                        <i>📭</i>
                        <h3>No bids have been submitted yet</h3>
                        <p>Once contractors start submitting bids, they will appear here with AI analysis.</p>
                    </td>
                </tr>
            """
        else:
            for b in enhanced_bids:
                row_class = "fair" if b['is_fair'] else "unfair"
                range_class = "range-good" if b['is_fair'] else "range-bad"
                status_class = "status-approved" if "Approved" in b['status'] else "status-rejected"
                admin_html += f"""
                <tr class="{row_class}">
                    <td>{b['bid_id']}</td>
                    <td>{b['contract_name']}</td>
                    <td>{b['company_name']}<br><span class="highlight">{b['cac_number']}</span></td>
                    <td>{b['email']}<br>{b['phone']}</td>
                    <td>₦{b['bid_amount']:.2f}</td>
                    <td class="{range_class}">₦{b['fair_min']:.2f} - ₦{b['fair_max']:.2f}</td>
                    <td class="{status_class}">{b['status']}</td>
                    <td class="timestamp">{b['timestamp']}</td>
                </tr>
                """

        admin_html += """
                        </tbody>
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=admin_html)
    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)
    except Exception as e:
        print(f"✗✗✗ ADMIN DASHBOARD ERROR: {str(e)}")
        return HTMLResponse(
            content=f"<h2 style='color:red;text-align:center;'>Dashboard error: {str(e)}</h2>",
            status_code=500
        )
