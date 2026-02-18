import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import FastAPI, Request, Form, HTTPException, Response, Depends
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
from contextlib import contextmanager
from typing import Dict, Any

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ────────────────────────────────────────────────
#  CORS (no trailing spaces!)
# ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://akandeaka.github.io",
        "http://localhost:8000",
        "http://localhost:3000",
        "https://aisec.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────
TRAINING_DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTXlHZrU20uniUkjr-5Pis1pfJSOYDUiFVcML6UqW2Lu176_opvZPQvTGOpQZnNx02HyFf-jRYw3O8o/pub?output=csv"
BIDDING_CONTRACTS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS-nWpM2oCQ5xmda7a3tlLiRmMC2VaAdG4IhoQsypuVvbYDgtDaWn_bYcClrc35XUoHRvvMEISXTvCw/pub?output=csv"

MODEL_PATH = "model.pkl"
DB_PATH = "bids.db"

# In-memory sessions (use Redis in production!)
user_sessions: Dict[str, int] = {}     # token → user_id
admin_sessions: Dict[str, int] = {}    # token → admin_id

# ────────────────────────────────────────────────
#  Database helpers
# ────────────────────────────────────────────────
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
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
            contract_id TEXT NOT NULL,
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

        # Create default admin (CHANGE THIS IN PRODUCTION!)
        try:
            default_pw = hashlib.sha256("admin123".encode()).hexdigest()
            cursor.execute(
                "INSERT INTO admins (username, hashed_password) VALUES (?, ?)",
                ("admin", default_pw)
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass

init_db()

# ────────────────────────────────────────────────
#  Model & Data
# ────────────────────────────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found → running training script...")
        subprocess.run([sys.executable, "train_model.py"], check=True)

ensure_model()

df_training = pd.read_csv(TRAINING_DATA_URL).reset_index(drop=True)
df_bidding   = pd.read_csv(BIDDING_CONTRACTS_URL).reset_index(drop=True)
model = joblib.load(MODEL_PATH)

# ────────────────────────────────────────────────
#  Email
# ────────────────────────────────────────────────
def send_bid_notification(to_email: str, company_name: str, contract_name: str, status: str, bid_amount: float):
    try:
        EMAIL_HOST = "smtp.gmail.com"
        EMAIL_PORT = 587
        EMAIL_USER = "aisec2025.notifications@gmail.com"
        EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD") or "YOUR_16_CHAR_APP_PASSWORD"

        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = f"AISEC Bid Submission - {status}"

        body = f"""Dear {company_name},

Your bid for "{contract_name}" has been successfully submitted!

Bid Amount: ₦{bid_amount:,.2f} Billion
Status: {status}

You can log in to your AISEC dashboard to view more details.

Thank you for using AISEC.

Best regards,
AISEC Team"""
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)

        print(f"Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False

# ────────────────────────────────────────────────
#  Session helpers
# ────────────────────────────────────────────────
def create_user_session(user_id: int) -> str:
    token = secrets.token_urlsafe(40)
    user_sessions[token] = user_id
    return token

def create_admin_session(admin_id: int) -> str:
    token = secrets.token_urlsafe(40)
    admin_sessions[token] = admin_id
    return token

def get_current_user(request: Request) -> int:
    token = request.cookies.get("session_token")
    if not token or token not in user_sessions:
        raise HTTPException(401, "Not authenticated")
    return user_sessions[token]

def get_current_admin(request: Request) -> int:
    token = request.cookies.get("admin_token")
    if not token or token not in admin_sessions:
        raise HTTPException(401, "Admin not authenticated")
    return admin_sessions[token]

# ────────────────────────────────────────────────
#  Price helpers
# ────────────────────────────────────────────────
def adjust_for_inflation(base_price: float, inflation_rate=0.12, years=2) -> float:
    return base_price * ((1 + inflation_rate) ** years)

def get_fair_price_range(contract_row) -> tuple[float, float]:
    feature_columns = [
        "award_year", "award_month", "primary_state", "geopolitical_zone",
        "latitude_start", "longitude_start", "estimated_length_km",
        "terrain_type", "rainfall_mm_per_year", "soil_type", "elevation_m",
        "has_bridge", "is_dual_carriageway", "is_rehabilitation", "is_coastal_or_swamp",
        "boq_earthworks_m3_per_km", "boq_asphalt_ton_per_km", "boq_drainage_km_per_km",
        "boq_bridges_units", "boq_culverts_units", "boq_premium_percent"
    ]
    features = contract_row[feature_columns]
    features_df = pd.DataFrame([features], columns=feature_columns)
    base = model.predict(features_df)[0]
    adjusted = adjust_for_inflation(base)
    return adjusted * 0.9, adjusted * 1.1

# ────────────────────────────────────────────────
#  Routes
# ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    return RedirectResponse("/contracts")

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
    hashed = hashlib.sha256(password.encode()).hexdigest()
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (company_name, cac_number, email, hashed_password) VALUES (?,?,?,?)",
                (company_name, cac_number, email, hashed)
            )
            conn.commit()
        return HTMLResponse("""
        <h2 style="color:green;text-align:center">Registration successful!</h2>
        <p style="text-align:center"><a href="/login">Login here</a></p>
        """)
    except sqlite3.IntegrityError:
        return HTMLResponse("""
        <h2 style="color:red;text-align:center">Email already registered</h2>
        <p style="text-align:center"><a href="/login">Login here</a></p>
        """, status_code=400)

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login_user(response: Response, email: str = Form(...), password: str = Form(...)):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    with get_db() as conn:
        user = conn.execute(
            "SELECT id FROM users WHERE email = ? AND hashed_password = ?",
            (email, hashed)
        ).fetchone()

    if user:
        token = create_user_session(user["id"])
        resp = RedirectResponse("/contracts", status_code=303)
        resp.set_cookie("session_token", token, httponly=True, max_age=86400)
        return resp

    return HTMLResponse("""
    <h2 style="color:red;text-align:center">Invalid credentials</h2>
    <p style="text-align:center"><a href="/login">Try again</a></p>
    """, status_code=401)

@app.get("/logout", response_class=HTMLResponse)
async def logout(response: Response):
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("session_token")
    return response

# ────────────────────────────────────────────────
#  Contracts & Bidding
# ────────────────────────────────────────────────
@app.get("/contracts", response_class=HTMLResponse)
def contracts(request: Request):
    try:
        user_id = get_current_user(request)
        with get_db() as conn:
            company = conn.execute(
                "SELECT company_name FROM users WHERE id = ?",
                (user_id,)
            ).fetchone()["company_name"]

            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT contract_id 
                FROM bids 
                WHERE company_name = ?
            """, (company,))
            already_bid_ids = {row[0] for row in cursor.fetchall()}   # set of project_id / contract_id values

        all_contracts = df_bidding.to_dict(orient="records")

        available_contracts = []
        bid_contracts     = []   # optional – for showing already bid ones

        for contract in all_contracts:
            cid = contract.get("Project_id")   # ← CHANGE to your real column name
            if cid is None:
                continue  # skip broken rows

            if cid in already_bid_ids:
                bid_contracts.append(contract)
            else:
                available_contracts.append(contract)

        if not available_contracts:
            return HTMLResponse("""
            <div style="max-width:700px;margin:80px auto;padding:40px;background:white;border-radius:16px;text-align:center;box-shadow:0 8px 30px #0002;">
                <div style="font-size:80px">🏆</div>
                <h2>All Contracts Have Been Bid On!</h2>
                <p style="font-size:18px;color:#555;margin:20px 0;">
                    Your company has submitted bids for every available contract.<br>
                    Administrators will review submissions shortly.
                </p>
                <a href="/logout" style="padding:14px 40px;background:#e11d48;color:white;border-radius:10px;text-decoration:none;font-weight:bold;">Logout</a>
            </div>
            """)

        return templates.TemplateResponse("contracts_fragment.html", {
            "request": request,
            "contracts": available_contracts,          # only not-yet-bid
            "bid_contracts": bid_contracts,            # optional
            "user_id": user_id,
            "has_bids": len(bid_contracts) > 0
        })

    except HTTPException:
        return RedirectResponse("/login", status_code=303)

@app.get("/contracts/{contract_id}", response_class=HTMLResponse)
def contract_detail(request: Request, contract_id: int):
    try:
        get_current_user(request)  # auth check
        if contract_id < 0 or contract_id >= len(df_bidding):
            raise HTTPException(404, "Contract not found")
        contract = df_bidding.iloc[contract_id].to_dict()
        return templates.TemplateResponse("contract_detail.html", {
            "request": request,
            "contract": contract,
            "contract_index": contract_id
        })
    except HTTPException as e:
        if e.status_code == 401:
            return RedirectResponse("/login", status_code=303)
        return HTMLResponse(f"<h2>Error {e.status_code}</h2><p>{e.detail}</p>", status_code=e.status_code)

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
    workforce: str = Form(...)
):
    try:
        user_id = get_current_user(request)  # will raise 401 if not logged in
    except HTTPException:
        return RedirectResponse("/login", status_code=303)

    if contract_id < 0 or contract_id >= len(df_bidding):
        return HTMLResponse("<h2>Invalid contract</h2>", status_code=400)

    contract = df_bidding.iloc[contract_id]
    fair_min, fair_max = get_fair_price_range(contract)
    status_msg = "Approved ✅" if fair_min <= bid_amount <= fair_max else "Rejected ❌"

    try:
        real_contract_id = contract.get("Project_id")

        if real_contract_id is None:
            raise HTTPException(500, "Contract is missing unique identifier (Project_id column not found)")

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO bids (
                    contract_id, user_id, company_name, cac_number, email, phone,
                    bid_amount, equipment_list, workforce, status
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                real_contract_id, user_id, company_name, cac_number, email, phone,
                bid_amount, equipment_list, workforce, status_msg
            ))
            bid_id = cursor.lastrowid
            conn.commit()

        # Email (non-blocking)
        send_bid_notification(
            email, company_name, contract.get("Project_name", "Unknown"),
            status_msg, bid_amount
        )

        return HTMLResponse(f"""
        <div style="max-width:750px;margin:40px auto;padding:35px;background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:3px solid #10b981;border-radius:20px;text-align:center;box-shadow:0 10px 35px #10b9833;">
            <h1 style="color:#065f46;font-size:2.1rem;margin-bottom:0.8rem;">Bid Submitted Successfully!</h1>
            <p style="font-size:1.15rem;color:#0f766e;margin:1rem 0 2rem;">Bid ID: <strong>#{bid_id}</strong></p>
            <div style="background:white;padding:1.5rem;border-radius:12px;margin:1.5rem 0;text-align:left;">
                <p><strong>Contract:</strong> {contract.get('Project_name','—')}</p>
                <p><strong>Amount:</strong> ₦{bid_amount:,.2f} Billion</p>
                <p><strong>AI Status:</strong> <span style="color:{'#10b981' if 'Approved' in status_msg else '#ef4444'};">{status_msg}</span></p>
            </div>
            <a href="/contracts" style="padding:14px 38px;background:#1d4ed8;color:white;border-radius:10px;text-decoration:none;font-weight:bold;">Back to Contracts</a>
        </div>
        """)

    except Exception as e:
        print(f"Bid submission failed: {e}")
        return HTMLResponse(f"""
        <div style="max-width:600px;margin:60px auto;padding:40px;background:#fef2f2;border:3px solid #ef4444;border-radius:16px;text-align:center;">
            <h1 style="color:#991b1b">Submission Failed</h1>
            <p style="color:#991b1b;margin:1.5rem 0;">{str(e)[:180]}</p>
            <a href="/contracts" style="padding:12px 30px;background:#1e40af;color:white;border-radius:8px;text-decoration:none;">Try Again</a>
        </div>
        """, status_code=500)

# ────────────────────────────────────────────────
#  ADMIN ROUTES
# ────────────────────────────────────────────────
@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_page():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head><title>Admin Login</title></head>
    <body style="margin:0;font-family:sans-serif;background:#f0f9ff;height:100vh;display:grid;place-items:center;">
        <div style="background:white;padding:3rem 2.5rem;border-radius:16px;box-shadow:0 10px 40px #00000022;width:380px;text-align:center;">
            <h2 style="color:#1e40af;margin-bottom:2rem;">AISEC Admin Login</h2>
            <form method="POST">
                <input name="username" placeholder="Username" required style="width:100%;padding:12px;margin:10px 0;border:1px solid #ddd;border-radius:8px;">
                <input name="password" type="password" placeholder="Password" required style="width:100%;padding:12px;margin:10px 0;border:1px solid #ddd;border-radius:8px;">
                <button type="submit" style="width:100%;padding:14px;background:#2563eb;color:white;border:none;border-radius:8px;font-weight:bold;margin-top:1rem;cursor:pointer;">Login</button>
            </form>
        </div>
    </body>
    </html>
    """)

@app.get("/debug/admin-check")
def debug_admin_check():
    with get_db() as conn:
        row = conn.execute("SELECT username, hashed_password FROM admins WHERE username = 'admin'").fetchone()
        if row:
            return {
                "exists": True,
                "username_in_db": row["username"],
                "hashed_password_in_db": row["hashed_password"][:20] + "...",  # partial for safety
            }
        else:
            return {"exists": False}

@app.post("/admin/login", response_class=HTMLResponse)
async def admin_login(response: Response, username: str = Form(...), password: str = Form(...)):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    with get_db() as conn:
        admin = conn.execute(
            "SELECT id FROM admins WHERE username = ? AND hashed_password = ?",
            (username, hashed)
        ).fetchone()

    if admin:
        token = create_admin_session(admin["id"])
        resp = RedirectResponse("/admin/dashboard", status_code=303)
        resp.set_cookie("admin_token", token, httponly=True, max_age=86400)
        return resp

    return HTMLResponse("""
    <h2 style="color:red;text-align:center;margin-top:120px;">Invalid admin credentials</h2>
    <p style="text-align:center"><a href="/admin/login">Back</a></p>
    """, status_code=401)

@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    try:
        get_current_admin(request)

        with get_db() as conn:
            bids = conn.execute("""
                SELECT id, contract_id, company_name, cac_number, email, phone,
                       bid_amount, equipment_list, workforce, status, timestamp
                FROM bids
                ORDER BY timestamp DESC
            """).fetchall()

        enhanced = []
        for row in bids:
            try:
                contract_row = df_bidding[df_bidding["Project_id"] == row["contract_id"]]
                if not contract_row.empty:
                    contract = contract_row.iloc[0]
                    fair_min, fair_max = get_fair_price_range(contract)
                    is_fair = fair_min <= row["bid_amount"] <= fair_max
                    enhanced.append({
                        "bid_id": row["id"],
                        "contract_name": contract.get("Project_name", f"Contract #{row['contract_id']}"),
                        "company_name": row["company_name"],
                        "cac_number": row["cac_number"],
                        "email": row["email"],
                        "phone": row["phone"],
                        "bid_amount": row["bid_amount"],
                        "fair_min": fair_min,
                        "fair_max": fair_max,
                        "is_fair": is_fair,
                        "status": row["status"],
                        "timestamp": row["timestamp"]
                    })
                else:
                    enhanced.append({
                        "bid_id": row["id"],
                        "contract_name": f"Unknown Contract #{row['contract_id']}",
                        "company_name": row["company_name"],
                        "cac_number": row["cac_number"],
                        "email": row["email"],
                        "phone": row["phone"],
                        "bid_amount": row["bid_amount"],
                        "fair_min": 0,
                        "fair_max": 0,
                        "is_fair": False,
                        "status": row["status"],
                        "timestamp": row["timestamp"]
                    })
            except Exception as e:
                print(f"Error enhancing bid {row['id']}: {e}")
                continue

        total_bids = len(enhanced)
        fair_count = sum(1 for b in enhanced if b["is_fair"])
        unfair_count = total_bids - fair_count
        total_value = sum(b["bid_amount"] for b in enhanced)

        body_html = ""
        if not enhanced:
            body_html = '<tr><td colspan="9" class="empty-state">No Bids Submitted Yet</td></tr>'

        for b in enhanced:
            row_class = "fair" if b["is_fair"] else "unfair"
            assessment = "✅ FAIR" if b["is_fair"] else "⚠️ HIGH"
            color_class = "color:#10b981;" if b["is_fair"] else "color:#f59e0b;"
            body_html += f"""
                    <tr class="{row_class}">
                        <td><strong>#{b['bid_id']}</strong></td>
                        <td>{b['contract_name']}</td>
                        <td>{b['company_name']}<br><small>CAC: {b['cac_number']}</small></td>
                        <td>{b['email']}<br><small>{b['phone']}</small></td>
                        <td>₦{b['bid_amount']:,.2f}</td>
                        <td>₦{b['fair_min']:,.2f} – ₦{b['fair_max']:,.2f}</td>
                        <td style="{color_class};font-weight:bold;">{assessment}</td>
                        <td>{b['status']}</td>
                        <td><small>{b['timestamp']}</small></td>
                    </tr>
            """

        # Build HTML (your original styling preserved + fixes)
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AISEC Admin Dashboard</title>
            <style>
                :root {{ --primary:#2563eb; --success:#10b981; --warning:#f59e0b; --danger:#ef4444; }}
                body {{ font-family:Segoe UI,sans-serif; background:#f8fafc; margin:0; }}
                .header {{ background:linear-gradient(135deg,#1e40af,#0c4a6e); color:white; padding:1.2rem 3rem; display:flex; justify-content:space-between; align-items:center; position:sticky; top:0; z-index:100; box-shadow:0 4px 12px #0003; }}
                .container {{ max-width:1800px; margin:2rem auto; padding:0 1.5rem; }}
                .stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:1.5rem; margin-bottom:2.5rem; }}
                .stat-card {{ background:white; border-radius:16px; padding:1.8rem; text-align:center; box-shadow:0 6px 20px #00000011; border-top:5px solid var(--primary); }}
                .stat-value {{ font-size:2.8rem; font-weight:800; background:linear-gradient(90deg,var(--primary),#0ea5e9); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
                table {{ width:100%; border-collapse:collapse; background:white; border-radius:16px; overflow:hidden; box-shadow:0 10px 35px #0000001a; }}
                th {{ background:#f1f5f9; padding:1.1rem; text-align:left; font-weight:700; color:#1e40af; position:sticky; top:70px; z-index:10; }}
                td {{ padding:1rem; border-bottom:1px solid #e2e8f0; }}
                tr:hover {{ background:#f8fafc; }}
                .fair {{ background:linear-gradient(to right,#f0fdf4 96%,#bbf7d0); }}
                .unfair {{ background:linear-gradient(to right,#fffbeb 96%,#fde68a); }}
                .logout-btn {{ background:#ef4444; color:white; padding:0.7rem 1.6rem; border-radius:10px; text-decoration:none; font-weight:bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <div style="font-size:1.6rem;font-weight:800;">🛡️ AISEC Admin</div>
                <a href="/admin/logout" class="logout-btn">Logout</a>
            </div>
            <div class="container">
                <h1 style="margin:2rem 0 1.5rem;color:#0f172a;font-size:2.2rem;">Bid Overview & AI Analysis</h1>
                
                <div class="stats-grid">
                    <div class="stat-card"><div class="stat-value">{total_bids}</div><div>Total Bids</div></div>
                    <div class="stat-card" style="border-top-color:var(--success);"><div class="stat-value">{fair_count}</div><div>Fair Bids</div></div>
                    <div class="stat-card" style="border-top-color:var(--warning);"><div class="stat-value">{unfair_count}</div><div>Flagged Bids</div></div>
                    <div class="stat-card"><div class="stat-value">₦{total_value:,.1f}B</div><div>Total Value</div></div>
                </div>

                <table>
                    <thead>
                        <tr>
                            <th>ID</th><th>Contract</th><th>Company / CAC</th><th>Contact</th>
                            <th>Bid (₦B)</th><th>AI Fair Range</th><th>Assessment</th><th>Status</th><th>Time</th>
                        </tr>
                    </thead>
                    <tbody>
{body_html}
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """)

    except HTTPException:
        return RedirectResponse("/admin/login", status_code=303)
    except Exception as e:
        print(f"Dashboard error: {e}")
        return HTMLResponse(f"<h1>Server Error</h1><pre>{str(e)}</pre>", status_code=500)

@app.get("/admin/logout", response_class=HTMLResponse)
async def admin_logout(response: Response):
    response = RedirectResponse("/admin/login", status_code=303)
    response.delete_cookie("admin_token")
    return response
