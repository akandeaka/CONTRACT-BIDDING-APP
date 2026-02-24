# main.py
import os
import hashlib
import secrets
import subprocess
import sys
import joblib
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from fastapi import FastAPI, Request, Form, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from contextlib import contextmanager
import psycopg2

# ────────────────────────────────────────────────
#  CONFIG & GLOBALS
# ────────────────────────────────────────────────

app = FastAPI()
templates = Jinja2Templates(directory="templates")

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

TRAINING_DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTXlHZrU20uniUkjr-5Pis1pfJSOYDUiFVcML6UqW2Lu176_opvZPQvTGOpQZnNx02HyFf-jRYw3O8o/pub?output=csv"
BIDDING_CONTRACTS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS-nWpM2oCQ5xmda7a3tlLiRmMC2VaAdG4IhoQsypuVvbYDgtDaWn_bYcClrc35XUoHRvvMEISXTvCw/pub?output=csv"

MODEL_PATH = "model.pkl"

sessions = {}  # simple in-memory session store (use redis in production)

# ────────────────────────────────────────────────
#  DATABASE
# ────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL, sslmode="require")

@contextmanager
def get_db():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        yield cur, conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

def init_database():
    with get_db() as (cur, _):
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            company_name TEXT NOT NULL,
            cac_number TEXT NOT NULL
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS bids (
            id SERIAL PRIMARY KEY,
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
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        );
        """)

        # Default admin (idempotent)
        cur.execute("""
        INSERT INTO admins (username, hashed_password)
        VALUES (%s, %s)
        ON CONFLICT (username) DO NOTHING
        """, ("admin", hashlib.sha256("admin123".encode()).hexdigest()))

init_database()

# ────────────────────────────────────────────────
#  MODEL & DATA
# ────────────────────────────────────────────────

def ensure_model_and_data():
    if not os.path.exists(MODEL_PATH):
        print("Training model...")
        subprocess.run([sys.executable, "train_model.py"], check=True)

ensure_model_and_data()

df_training = pd.read_csv(TRAINING_DATA_URL).reset_index(drop=True)
df_bidding   = pd.read_csv(BIDDING_CONTRACTS_URL).reset_index(drop=True)
model = joblib.load(MODEL_PATH)

# ────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────

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

def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    sessions[token] = user_id
    return token

def get_current_user(request: Request) -> int:
    token = request.cookies.get("session_token")
    if not token or token not in sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return sessions[token]

def get_admin_user(request: Request) -> int:
    token = request.cookies.get("admin_token")
    if not token or token not in sessions:
        raise HTTPException(status_code=401, detail="Admin not authenticated")
    return sessions[token]

# ────────────────────────────────────────────────
#  EMAIL
# ────────────────────────────────────────────────

def send_bid_notification(email: str, company_name: str, contract_name: str, status: str, bid_amount: float):
    try:
        EMAIL_HOST = "smtp.gmail.com"
        EMAIL_PORT = 587
        EMAIL_USER = "aisec2025.notifications@gmail.com"           # ← CHANGE
        EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")           # ← use env var!

        if not EMAIL_PASSWORD:
            raise ValueError("EMAIL_APP_PASSWORD not set")

        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = email
        msg['Subject'] = f"AISEC Bid Submission - {status}"

        body = f"""Dear {company_name},

Your bid for "{contract_name}" has been successfully submitted!

Bid Amount: ₦{bid_amount:,.2f} Billion
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

        print(f"Email sent to {email}")
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False

# ────────────────────────────────────────────────
#  ROUTES
# ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    return RedirectResponse(url="/contracts")

# ── Register ─────────────────────────────────────

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
        with get_db() as (cur, _):
            cur.execute(
                """
                INSERT INTO users (company_name, cac_number, email, hashed_password)
                VALUES (%s, %s, %s, %s)
                """,
                (company_name, cac_number, email, hashed)
            )
        return HTMLResponse("""
        <h2 style="color:green;text-align:center;">✅ Registration successful!</h2>
        <p style="text-align:center;"><a href="/login">Login here</a></p>
        """)
    except psycopg2.IntegrityError:
        return HTMLResponse("""
        <h2 style="color:red;text-align:center;">❌ Email already registered</h2>
        <p style="text-align:center;"><a href="/login">Login here</a></p>
        """, status_code=400)
    except Exception as e:
        print(f"Register error: {e}")
        return HTMLResponse("<h2 style='color:red;'>Server error</h2>", status_code=500)

# ── Login ────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login_user(response: Response, email: str = Form(...), password: str = Form(...)):
    hashed = hashlib.sha256(password.encode()).hexdigest()

    try:
        with get_db() as (cur, _):
            cur.execute(
                "SELECT id FROM users WHERE email = %s AND hashed_password = %s",
                (email, hashed)
            )
            user = cur.fetchone()

        if user:
            token = create_session(user[0])
            resp = RedirectResponse(url="/contracts", status_code=303)
            resp.set_cookie(key="session_token", value=token, httponly=True, max_age=3600*24)
            return resp

        return HTMLResponse("""
        <h2 style="color:red;text-align:center;">❌ Invalid credentials</h2>
        <p style="text-align:center;"><a href="/login">Try again</a></p>
        """, status_code=401)

    except Exception as e:
        print(f"Login error: {e}")
        return HTMLResponse("<h2 style='color:red;'>Server error</h2>", status_code=500)

@app.get("/logout", response_class=HTMLResponse)
async def logout(response: Response):
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie("session_token")
    return resp

# ── Contracts list ───────────────────────────────

# ── Contracts list ───────────────────────────────

@app.get("/contracts", response_class=HTMLResponse)
def contracts(request: Request):
    try:
        user_id = get_current_user(request)

        with get_db() as (cur, _):
            cur.execute("SELECT company_name FROM users WHERE id = %s", (user_id,))
            company_row = cur.fetchone()
            if not company_row:
                return RedirectResponse(url="/login", status_code=303)
            company_name = company_row[0]

            # ── DEBUG PRINT ────────────────────────────────
            print(f"[DEBUG /contracts] user_id={user_id} | company_name='{company_name}'")

            cur.execute("SELECT contract_id FROM bids WHERE company_name = %s", (company_name,))
            existing_rows = cur.fetchall()
            existing = {r[0] for r in existing_rows}

            # ── DEBUG PRINT ────────────────────────────────
            print(f"[DEBUG /contracts] Found {len(existing)} existing bids for '{company_name}': {sorted(existing)}")

        all_contracts = df_bidding.to_dict(orient="records")
        available = [c for i, c in enumerate(all_contracts) if i not in existing]

        if not available:
            return HTMLResponse(""" ... same as before ... """)

        return templates.TemplateResponse("contracts_fragment.html", {
            "request": request,
            "contracts": available,
            "user_id": user_id
        })

    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)
    except Exception as e:
        print(f"Contracts error: {e}")
        return RedirectResponse(url="/login", status_code=303)


# ── Bid submission ─────────────────────────────────

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
    workforce: str = Form(...)
):
    try:
        user_id = get_current_user(request)

        # Logged-in user → override form with DB values
        with get_db() as (cur, conn):
            cur.execute(
                "SELECT company_name, cac_number, email FROM users WHERE id = %s",
                (user_id,)
            )
            row = cur.fetchone()
            if row:
                company_name, cac_number, email = row

        # Parse bid amount
        try:
            clean = bid_amount.replace(",", "").strip()
            bid_value = float(clean)
        except ValueError:
            return HTMLResponse("""
            <div style="max-width:650px;margin:50px auto;padding:30px;background:#fef2f2;border:3px solid #ef4444;border-radius:16px;text-align:center;">
                <h1 style="color:#991b1b;">Invalid Bid Amount</h1>
                <p>Please enter a valid number (e.g. 12.5 or 12,500,000,000)</p>
                <a href="/contracts" style="display:inline-block;margin-top:20px;padding:12px 30px;background:#1e40af;color:white;border-radius:8px;text-decoration:none;">← Back</a>
            </div>
            """, status_code=400)

        contract = df_bidding.iloc[contract_id]
        fair_min, fair_max = get_fair_price_range(contract)
        status_msg = "Approved ✅" if fair_min <= bid_value <= fair_max else "Rejected ❌"

        # Save bid + debug
        with get_db() as (cur, conn):
            cur.execute("""
            INSERT INTO bids (
                contract_id, user_id, company_name, cac_number, email, phone,
                bid_amount, equipment_list, workforce, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """, (
                contract_id, user_id, company_name, cac_number, email, phone,
                bid_value, equipment_list, workforce, status_msg
            ))
            bid_id_result = cur.fetchone()
            bid_id = bid_id_result[0] if bid_id_result else None

            # Explicit commit (redundant in context manager but helps visibility)
            conn.commit()

            # ── DEBUG PRINT ────────────────────────────────
            print(f"[DEBUG submit_bid] INSERTED bid_id={bid_id} | contract_id={contract_id} | company='{company_name}' | amount={bid_value} | status={status_msg}")

        # Email notification
        email_ok = send_bid_notification(
            email, company_name, contract.get("project_name", "Unknown contract"), status_msg, bid_value
        )

        color = "#10b981" if "Approved" in status_msg else "#ef4444"
        email_msg = (
            "<p style='color:#10b981;font-weight:600;'>📧 Confirmation email sent!</p>"
            if email_ok else
            "<p style='color:#f59e0b;'>⚠️ Bid saved (email failed)</p>"
        )

        # Success page
        return HTMLResponse(f"""
        <div style="max-width:750px;margin:40px auto;padding:40px;background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:3px solid #10b981;border-radius:20px;text-align:center;box-shadow:0 10px 30px rgba(16,185,129,0.25);">
            <h1 style="color:#065f46;font-size:2.2rem;margin-bottom:0.5rem;">🎉 BID SUBMITTED SUCCESSFULLY</h1>
            <p style="color:#0f766e;font-size:1.2rem;margin-bottom:2rem;">Your bid has been recorded (ID: {bid_id})</p>
            <!-- rest of your success HTML remains the same -->
            ...
        </div>
        """)

    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)
    except Exception as e:
        print(f"[ERROR submit_bid] {type(e).__name__}: {str(e)}")
        return HTMLResponse(f"""
        <div style="max-width:650px;margin:50px auto;padding:30px;background:#fef2f2;border:3px solid #ef4444;border-radius:16px;text-align:center;">
            <h1 style="color:#991b1b;">Submission Failed</h1>
            <p>{str(e)[:240]}</p>
            <a href="/contracts" style="display:inline-block;margin-top:20px;padding:12px 30px;background:#1e40af;color:white;border-radius:8px;text-decoration:none;">← Back</a>
        </div>
        """, status_code=500)
# ── Contract detail & bid submission ─────────────
# main.py
# ... (keep all the imports, config, database setup, helpers, email function exactly as they were)

# ── Contracts list ───────────────────────────────

@app.get("/contracts", response_class=HTMLResponse)
def contracts(request: Request):
    try:
        user_id = get_current_user(request)

        with get_db() as (cur, _):
            cur.execute("SELECT company_name FROM users WHERE id = %s", (user_id,))
            company_row = cur.fetchone()
            if not company_row:
                return RedirectResponse(url="/login", status_code=303)
            company_name = company_row[0].strip()  # .strip() helps avoid trailing spaces issues

            # Debug output – visible in Render logs
            print(f"[DEBUG /contracts] user_id={user_id} | company_name='{company_name}'")

            cur.execute("SELECT contract_id FROM bids WHERE company_name = %s", (company_name,))
            existing_rows = cur.fetchall()
            existing = {r[0] for r in existing_rows}

            print(f"[DEBUG /contracts] Found {len(existing)} existing bid(s) for '{company_name}': {sorted(existing)}")

        all_contracts = df_bidding.to_dict(orient="records")
        available = [c for i, c in enumerate(all_contracts) if i not in existing]

        if not available:
            return HTMLResponse("""
            <div style="max-width:700px;margin:50px auto;background:white;border-radius:16px;padding:40px;text-align:center;box-shadow:0 5px 20px rgba(0,0,0,0.1)">
                <div style="font-size:64px;margin-bottom:20px">✅</div>
                <h2 style="color:#1e40af;margin-bottom:15px">All Contracts Bid Successfully!</h2>
                <p style="color:#475569;font-size:18px;margin-bottom:25px">
                    Your company has submitted bids for all available contracts.<br>
                    Administrators will review your submissions shortly.
                </p>
                <a href="/logout" style="display:inline-block;padding:12px 30px;background:#ef4444;color:white;text-decoration:none;border-radius:8px;font-weight:600">
                    Logout
                </a>
            </div>
            """)

        return templates.TemplateResponse("contracts_fragment.html", {
            "request": request,
            "contracts": available,
            "user_id": user_id
        })

    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)
    except Exception as e:
        print(f"Contracts route error: {type(e).__name__}: {str(e)}")
        return RedirectResponse(url="/login", status_code=303)


# ── Bid submission ─────────────────────────────────

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
    workforce: str = Form(...)
):
    try:
        user_id = get_current_user(request)

        # Use database values for logged-in user (override form values)
        with get_db() as (cur, conn):
            cur.execute(
                "SELECT company_name, cac_number, email FROM users WHERE id = %s",
                (user_id,)
            )
            row = cur.fetchone()
            if row:
                company_name, cac_number, email = [v.strip() if isinstance(v, str) else v for v in row]

        # Parse bid amount
        try:
            clean = bid_amount.replace(",", "").strip()
            bid_value = float(clean)
        except ValueError:
            return HTMLResponse("""
            <div style="max-width:650px;margin:50px auto;padding:30px;background:#fef2f2;border:3px solid #ef4444;border-radius:16px;text-align:center;">
                <h1 style="color:#991b1b;">Invalid Bid Amount</h1>
                <p>Please enter a valid number (e.g. 12.5 or 12500000000)</p>
                <a href="/contracts" style="display:inline-block;margin-top:20px;padding:12px 30px;background:#1e40af;color:white;border-radius:8px;text-decoration:none;">← Back</a>
            </div>
            """, status_code=400)

        contract = df_bidding.iloc[contract_id]
        fair_min, fair_max = get_fair_price_range(contract)
        status_msg = "Approved ✅" if fair_min <= bid_value <= fair_max else "Rejected ❌"

        # Save bid + debug logging
        with get_db() as (cur, conn):
            cur.execute("""
            INSERT INTO bids (
                contract_id, user_id, company_name, cac_number, email, phone,
                bid_amount, equipment_list, workforce, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """, (
                contract_id, user_id, company_name, cac_number, email, phone,
                bid_value, equipment_list, workforce, status_msg
            ))
            bid_id_result = cur.fetchone()
            bid_id = bid_id_result[0] if bid_id_result else None

            conn.commit()

            # Important debug output – will appear in Render logs
            print(f"[DEBUG submit_bid] INSERTED → bid_id={bid_id} | contract={contract_id} | company='{company_name}' | amount={bid_value} | status={status_msg}")

        # Email
        email_ok = send_bid_notification(
            email, company_name, contract.get("project_name", "Unknown"), status_msg, bid_value
        )

        color = "#10b981" if "Approved" in status_msg else "#ef4444"
        email_msg = "<p style='color:#10b981;font-weight:600;'>📧 Confirmation email sent!</p>" if email_ok else "<p style='color:#f59e0b;'>⚠️ Bid saved (email failed)</p>"

        return HTMLResponse(f"""
        <div style="max-width:750px;margin:40px auto;padding:40px;background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:3px solid #10b981;border-radius:20px;text-align:center;box-shadow:0 10px 30px rgba(16,185,129,0.25);">
            <h1 style="color:#065f46;font-size:2.2rem;margin-bottom:0.5rem;">🎉 BID SUBMITTED SUCCESSFULLY</h1>
            <p style="color:#0f766e;font-size:1.2rem;margin-bottom:2rem;">Your bid has been recorded (ID: {bid_id})</p>

            <div style="background:white;padding:1.8rem;border-radius:16px;margin:1.5rem 0;box-shadow:0 4px 12px rgba(0,0,0,0.08);text-align:left;">
                <p><strong>Contract:</strong> {contract.get('project_name', 'N/A')}</p>
                <p><strong>Company:</strong> {company_name}</p>
                <p><strong>CAC:</strong> {cac_number}</p>
                <p><strong>Bid Amount:</strong> <span style="font-size:1.4rem;font-weight:bold;color:#065f46;">₦{bid_value:,.2f} Billion</span></p>
                <p style="font-weight:bold;color:{color};">AI Assessment: {status_msg}</p>
                <div style="margin-top:1rem;padding:0.8rem;background:#f0fdf4;border-left:4px solid #10b981;border-radius:8px;">
                    Bid ID: {bid_id} • Submitted: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
                </div>
            </div>

            {email_msg}

            <a href="/contracts" style="display:inline-block;margin-top:1.5rem;padding:14px 40px;background:#1e40af;color:white;border-radius:12px;font-weight:700;text-decoration:none;box-shadow:0 4px 12px rgba(30,64,175,0.3);">
                View Remaining Contracts
            </a>
        </div>
        """)

    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)
    except Exception as e:
        print(f"[ERROR submit_bid] {type(e).__name__}: {str(e)}")
        return HTMLResponse(f"""
        <div style="max-width:650px;margin:50px auto;padding:30px;background:#fef2f2;border:3px solid #ef4444;border-radius:16px;text-align:center;">
            <h1 style="color:#991b1b;">Submission Failed</h1>
            <p>{str(e)[:240]}</p>
            <a href="/contracts" style="display:inline-block;margin-top:20px;padding:12px 30px;background:#1e40af;color:white;border-radius:8px;text-decoration:none;">← Back</a>
        </div>
        """, status_code=500)


# Keep the debug endpoint if you want it (optional – good for testing)
@app.get("/debug/bids")
def debug_bids():
    with get_db() as (cur, _):
        cur.execute("SELECT id, contract_id, company_name, bid_amount, status FROM bids ORDER BY id DESC LIMIT 5")
        rows = cur.fetchall()
    return {"recent_bids": [dict(zip(["id", "contract_id", "company_name", "bid_amount", "status"], row)) for row in rows]}

     

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
    workforce: str = Form(...)
):
    try:
        user_id = get_current_user(request)

        # Logged-in user → override form with DB values
        with get_db() as (cur, _):
            cur.execute(
                "SELECT company_name, cac_number, email FROM users WHERE id = %s",
                (user_id,)
            )
            row = cur.fetchone()
            if row:
                company_name, cac_number, email = row

        # Parse bid
        try:
            clean = bid_amount.replace(",", "").strip()
            bid_value = float(clean)
        except ValueError:
            return HTMLResponse("""
            <div style="max-width:650px;margin:50px auto;padding:30px;background:#fef2f2;border:3px solid #ef4444;border-radius:16px;text-align:center;">
                <h1 style="color:#991b1b;">Invalid Bid Amount</h1>
                <p>Please enter a valid number (e.g. 12.5 or 12,500,000,000)</p>
                <a href="/contracts" style="display:inline-block;margin-top:20px;padding:12px 30px;background:#1e40af;color:white;border-radius:8px;text-decoration:none;">← Back</a>
            </div>
            """, status_code=400)

        contract = df_bidding.iloc[contract_id]
        fair_min, fair_max = get_fair_price_range(contract)
        status_msg = "Approved ✅" if fair_min <= bid_value <= fair_max else "Rejected ❌"

        # Save bid
        with get_db() as (cur, _):
            cur.execute("""
            INSERT INTO bids (
                contract_id, user_id, company_name, cac_number, email, phone,
                bid_amount, equipment_list, workforce, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """, (
                contract_id, user_id, company_name, cac_number, email, phone,
                bid_value, equipment_list, workforce, status_msg
            ))
            bid_id = cur.fetchone()[0]

        # Email
        email_ok = send_bid_notification(
            email, company_name, contract["project_name"], status_msg, bid_value
        )

        color = "#10b981" if "Approved" in status_msg else "#ef4444"
        email_msg = "<p style='color:#10b981;font-weight:600;'>📧 Confirmation email sent!</p>" if email_ok else "<p style='color:#f59e0b;'>⚠️ Bid saved (email failed)</p>"

        return HTMLResponse(f"""
        <div style="max-width:750px;margin:40px auto;padding:40px;background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:3px solid #10b981;border-radius:20px;text-align:center;box-shadow:0 10px 30px rgba(16,185,129,0.25);">
            <h1 style="color:#065f46;font-size:2.2rem;margin-bottom:0.5rem;">🎉 BID SUBMITTED SUCCESSFULLY</h1>
            <p style="color:#0f766e;font-size:1.2rem;margin-bottom:2rem;">Your bid has been recorded</p>

            <div style="background:white;padding:1.8rem;border-radius:16px;margin:1.5rem 0;box-shadow:0 4px 12px rgba(0,0,0,0.08);text-align:left;">
                <p><strong>Contract:</strong> {contract['project_name']}</p>
                <p><strong>Company:</strong> {company_name}</p>
                <p><strong>CAC:</strong> {cac_number}</p>
                <p><strong>Bid Amount:</strong> <span style="font-size:1.4rem;font-weight:bold;color:#065f46;">₦{bid_value:,.2f} Billion</span></p>
                <p style="font-weight:bold;color:{color};">AI Assessment: {status_msg}</p>
                <div style="margin-top:1rem;padding:0.8rem;background:#f0fdf4;border-left:4px solid #10b981;border-radius:8px;">
                    Bid ID: {bid_id} • Submitted: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
                </div>
            </div>

            {email_msg}

            <a href="/contracts" style="display:inline-block;margin-top:1.5rem;padding:14px 40px;background:#1e40af;color:white;border-radius:12px;font-weight:700;text-decoration:none;box-shadow:0 4px 12px rgba(30,64,175,0.3);">
                View Remaining Contracts
            </a>
        </div>
        """)

    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)
    except Exception as e:
        print(f"Bid submission failed: {e}")
        return HTMLResponse(f"""
        <div style="max-width:650px;margin:50px auto;padding:30px;background:#fef2f2;border:3px solid #ef4444;border-radius:16px;text-align:center;">
            <h1 style="color:#991b1b;">Submission Failed</h1>
            <p>{str(e)[:180]}</p>
            <a href="/contracts" style="display:inline-block;margin-top:20px;padding:12px 30px;background:#1e40af;color:white;border-radius:8px;text-decoration:none;">← Back</a>
        </div>
        """, status_code=500)
with get_db() as (cur, conn):
    cur.execute("...")
    bid_id = cur.fetchone()[0]
    print(f"DEBUG: Bid inserted with ID {bid_id} for contract {contract_id} by {company_name}")
    conn.commit()  # make sure this is inside the context (already is in your version)
# ── Admin ────────────────────────────────────────

@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_page():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Admin Login</title></head>
    <body style="font-family:sans-serif;background:#f0f9ff;display:flex;justify-content:center;align-items:center;min-height:100vh;">
        <div style="background:white;padding:40px;border-radius:16px;box-shadow:0 10px 30px rgba(0,0,0,0.12);max-width:420px;width:100%;">
            <h2 style="text-align:center;color:#1e40af;margin-bottom:30px;">AISEC Admin Login</h2>
            <form method="post" style="display:flex;flex-direction:column;gap:16px;">
                <input type="text" name="username" placeholder="Username" required style="padding:12px;border:1px solid #ddd;border-radius:8px;">
                <input type="password" name="password" placeholder="Password" required style="padding:12px;border:1px solid #ddd;border-radius:8px;">
                <button type="submit" style="padding:14px;background:#2563eb;color:white;border:none;border-radius:8px;font-weight:600;cursor:pointer;">Login</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.post("/admin/login", response_class=HTMLResponse)
async def admin_login(response: Response, username: str = Form(...), password: str = Form(...)):
    hashed = hashlib.sha256(password.encode()).hexdigest()

    try:
        with get_db() as (cur, _):
            cur.execute(
                "SELECT id FROM admins WHERE username = %s AND hashed_password = %s",
                (username, hashed)
            )
            admin = cur.fetchone()

        if admin:
            token = create_session(admin[0])
            resp = RedirectResponse(url="/admin/dashboard", status_code=303)
            resp.set_cookie(key="admin_token", value=token, httponly=True, max_age=3600*24)
            return resp

        return HTMLResponse("<h2 style='color:red;text-align:center;'>Invalid credentials</h2><p><a href='/admin/login'>Try again</a></p>")

    except Exception as e:
        print(f"Admin login error: {e}")
        return HTMLResponse("<h2 style='color:red;'>Server error</h2>")

@app.get("/admin/logout", response_class=HTMLResponse)
def admin_logout(response: Response):
    resp = RedirectResponse(url="/admin/login", status_code=303)
    resp.delete_cookie("admin_token")
    return resp

@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    try:
        get_admin_user(request)

        with get_db() as (cur, _):
            cur.execute("""
            SELECT id, contract_id, company_name, cac_number, email, phone,
                   bid_amount, equipment_list, workforce, status, timestamp
            FROM bids ORDER BY timestamp DESC
            """)
            bids = cur.fetchall()

        enhanced = []
        for bid in bids:
            try:
                row = df_bidding.iloc[bid[1]]
                fmin, fmax = get_fair_price_range(row)
                is_fair = fmin <= bid[6] <= fmax
                enhanced.append({
                    "bid_id": bid[0],
                    "contract_name": row.get("project_name", f"Contract {bid[1]}"),
                    "company_name": bid[2] or "—",
                    "cac_number": bid[3] or "—",
                    "email": bid[4],
                    "phone": bid[5],
                    "bid_amount": bid[6],
                    "fair_min": fmin,
                    "fair_max": fmax,
                    "is_fair": is_fair,
                    "status": bid[9],
                    "timestamp": bid[10]
                })
            except Exception:
                enhanced.append({
                    "bid_id": bid[0],
                    "contract_name": f"Contract {bid[1]} (data missing)",
                    "company_name": bid[2] or "—",
                    "cac_number": bid[3] or "—",
                    "email": bid[4],
                    "phone": bid[5],
                    "bid_amount": bid[6],
                    "fair_min": 0,
                    "fair_max": 0,
                    "is_fair": False,
                    "status": bid[9],
                    "timestamp": bid[10]
                })

        total = len(enhanced)
        fair_count = sum(1 for x in enhanced if x["is_fair"])
        unfair_count = total - fair_count
        total_value = sum(x["bid_amount"] for x in enhanced)

        # ── HTML ───────────────────────────────────── (kept similar structure)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AISEC Admin Dashboard</title>
    <style>
        :root {{ --p:#2563eb; --s:#10b981; --w:#f59e0b; --d:#ef4444; }}
        body {{ font-family:'Segoe UI',sans-serif; background:#f8fafc; margin:0; }}
        .header {{ background:linear-gradient(135deg,#1e40af,#0c4a6e); color:white; padding:1.2rem 3rem; display:flex; justify-content:space-between; align-items:center; position:sticky; top:0; z-index:100; box-shadow:0 4px 12px rgba(0,0,0,0.2); }}
        .container {{ max-width:1800px; margin:2rem auto; padding:0 1.5rem; }}
        .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:1.5rem; margin-bottom:2.5rem; }}
        .card {{ background:white; border-radius:16px; padding:1.8rem; text-align:center; box-shadow:0 6px 20px rgba(0,0,0,0.08); border-top:5px solid var(--p); transition:transform 0.2s; }}
        .card:hover {{ transform:translateY(-6px); }}
        .big {{ font-size:3.2rem; font-weight:800; background:linear-gradient(90deg,var(--p),#0ea5e9); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
        table {{ width:100%; border-collapse:collapse; background:white; border-radius:16px; overflow:hidden; box-shadow:0 10px 30px rgba(0,0,0,0.1); }}
        th {{ background:#f1f5f9; padding:1.1rem; text-align:left; font-weight:700; color:#1e40af; position:sticky; top:70px; z-index:10; }}
        td {{ padding:1rem; border-bottom:1px solid #f1f5f9; }}
        tr:hover {{ background:#f8fafc; }}
        .fair {{ background:#f0fdf4; border-left:5px solid var(--s); }}
        .unfair {{ background:#fff7ed; border-left:5px solid var(--w); }}
        .approved {{ color:var(--s); font-weight:700; }}
        .rejected {{ color:var(--d); font-weight:700; }}
    </style>
</head>
<body>
<div class="header">
    <div style="font-size:1.6rem;font-weight:800;">🛡️ AISEC Admin Dashboard</div>
    <a href="/admin/logout" style="padding:0.7rem 1.6rem;background:var(--d);color:white;border-radius:10px;text-decoration:none;font-weight:600;">Logout</a>
</div>

<div class="container">
<h1 style="color:#0f172a;margin:2rem 0 1.5rem;font-size:2.4rem;">Bid Overview & AI Fairness Check</h1>

<div class="stats">
    <div class="card"><div class="big">{total}</div><div>Total Bids</div></div>
    <div class="card" style="border-top-color:var(--s);"><div class="big">{fair_count}</div><div>Fair (Approved)</div></div>
    <div class="card" style="border-top-color:var(--w);"><div class="big">{unfair_count}</div><div>Flagged (High)</div></div>
    <div class="card" style="border-top-color:#0ea5e9;"><div class="big">₦{total_value:,.2f}B</div><div>Total Value</div></div>
</div>

<table>
<thead><tr>
    <th>Bid ID</th>
    <th>Contract</th>
    <th>Company / CAC</th>
    <th>Contact</th>
    <th>Bid (₦B)</th>
    <th>AI Fair Range (₦B)</th>
    <th>Assessment</th>
    <th>Status</th>
    <th>Time</th>
</tr></thead>
<tbody>
"""

        if not enhanced:
            html += '<tr><td colspan="9" style="text-align:center;padding:3rem;color:#64748b;">No bids yet</td></tr>'
        else:
            for b in enhanced:
                cls = "fair" if b["is_fair"] else "unfair"
                range_cls = "approved" if b["is_fair"] else "rejected"
                html += f"""
                <tr class="{cls}">
                    <td>{b["bid_id"]}</td>
                    <td>{b["contract_name"]}</td>
                    <td>{b["company_name"]}<br><small style="color:#6b7280;">{b["cac_number"]}</small></td>
                    <td>{b["email"]}<br><small style="color:#6b7280;">{b["phone"]}</small></td>
                    <td>₦{b["bid_amount"]:,.2f}B</td>
                    <td class="{range_cls}">₦{b["fair_min"]:,.2f} – ₦{b["fair_max"]:,.2f}B</td>
                    <td class="{'approved' if b['is_fair'] else 'rejected'}">{'Within range' if b['is_fair'] else 'High'}</td>
                    <td>{b["status"]}</td>
                    <td><small>{b["timestamp"]}</small></td>
                </tr>
                """

        html += "</tbody></table></div></body></html>"

        return HTMLResponse(html)

    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)
    except Exception as e:
        print(f"Dashboard error: {e}")
        return HTMLResponse("<h2 style='color:red;text-align:center;'>Error loading dashboard</h2>")

# ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
@app.get("/debug/bids")
def debug_bids():
    with get_db() as (cur, _):
        cur.execute("SELECT id, contract_id, company_name, bid_amount, status FROM bids ORDER BY id DESC LIMIT 5")
        rows = cur.fetchall()
    return {"recent_bids": [dict(row) for row in rows]}



