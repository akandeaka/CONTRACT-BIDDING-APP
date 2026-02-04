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

# CORS middleware (NO trailing spaces)
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

# URLs for datasets (NO trailing spaces)
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

# Email function
def send_bid_notification(email, company_name, contract_name, status, bid_amount):
    try:
        # Yahoo SMTP configuration
        EMAIL_HOST = "smtp.mail.yahoo.com"
        EMAIL_PORT = 587
        EMAIL_USER = "aiseс.notifications@yahoo.com"  # ← Replace with your Yahoo email
        EMAIL_PASSWORD = "your-16-char-app-password"  # ← Replace with your app password
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = email
        msg['Subject'] = f"AISEC Bid Submission - {status}"
        
        body = f"""
        Dear {company_name},
        
        Your bid for "{contract_name}" has been successfully submitted!
        
        Bid Amount: ₦{bid_amount:.2f} Billion
        Status: {status}
        
        You can log in to your AISEC dashboard to view more details.
        
        Thank you for using AISEC - AI for Secure and Efficient Contracting.
        
        Best regards,
        AISEC Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_USER, email, text)
        server.quit()
        
        print(f"Email sent successfully to {email}")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

# Simple session storage
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
        return "<h2 style='color:green;'>✅ Registration successful!</h2><p><a href='/login'>Login here</a></p>"
    except sqlite3.IntegrityError:
        return "<h2 style='color:red;'>❌ Email already registered</h2><p><a href='/login'>Login here</a></p>"

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
        return "<h2 style='color:red;'>❌ Invalid credentials</h2><p><a href='/login'>Try again</a></p>"

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

        # Send email notification
        send_bid_notification(email, company_name, bidding_contract['project_name'], status_msg, bid_amount)

        return f"<h2>✅ Bid Submitted Successfully!</h2><p>Status: {status_msg}</p><p>An email notification has been sent to {email}</p><br><a href='/contracts' class='btn btn-primary'>Back to Contracts</a>"
        
    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)

@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_page(request: Request):
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Admin Login - AISEC</title></head>
    <body style="font-family: Arial, sans-serif; max-width: 400px; margin: 50px auto;">
        <h2 style="text-align: center;">AISEC Admin Login</h2>
        <form method="POST" style="display: flex; flex-direction: column; gap: 15px;">
            <input type="text" name="username" placeholder="Username" required style="padding: 10px; border: 1px solid #ddd; border-radius: 4px;">
            <input type="password" name="password" placeholder="Password" required style="padding: 10px; border: 1px solid #ddd; border-radius: 4px;">
            <button type="submit" style="padding: 10px; background: #2563eb; color: white; border: none; border-radius: 4px; cursor: pointer;">Login</button>
        </form>
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
        return "<h2 style='color:red;'>Invalid admin credentials</h2><a href='/admin/login' style='color:#2563eb;'>Try again</a>"

@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    try:
        admin_id = get_admin_user(request)
        
        # Get all bids
        cursor.execute("SELECT * FROM bids ORDER BY timestamp DESC")
        bids = cursor.fetchall()
        
        admin_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AISEC Admin Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f8fafc; font-weight: bold; }
                .approved { color: green; font-weight: bold; }
                .rejected { color: red; font-weight: bold; }
                .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
                .logout { color: #ef4444; text-decoration: none; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AISEC Admin Dashboard</h1>
                <a href="/admin/logout" class="logout">Logout</a>
            </div>
            <h2>All Bids</h2>
            <table>
                <tr>
                    <th>Company Name</th>
                    <th>CAC Number</th>
                    <th>Email</th>
                    <th>Bid Amount (₦B)</th>
                    <th>Status</th>
                    <th>Submitted</th>
                </tr>
        """
        
        for bid in bids:
            status_class = "approved" if "Approved" in bid[10] else "rejected"
            admin_html += f"""
                <tr>
                    <td>{bid[3]}</td>
                    <td>{bid[4]}</td>
                    <td>{bid[5]}</td>
                    <td>{bid[8]:.2f}</td>
                    <td class="{status_class}">{bid[10]}</td>
                    <td>{bid[11]}</td>
                </tr>
            """
        
        admin_html += """
            </table>
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
