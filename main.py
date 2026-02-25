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
import re  # For parsing primary_state

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
            admin_status TEXT DEFAULT 'pending',
            comments TEXT,
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

def adjust_for_inflation(base_price, inflation_rate=0.2, years=1):  # Updated to 20% for 2026
    return base_price * ((1 + inflation_rate) ** years)

def parse_primary_state(project_name):
    words = re.findall(r'\w+', project_name)
    if len(words) > 1:
        return words[0]  # e.g. 'Sokoto' from 'Sokoto Katsina road'
    return 'Unknown'

def get_fair_price_range(contract_row):
    contract_row = contract_row.copy()  # Avoid modifying original
    contract_row['award_year'] = 2026
    contract_row['award_month'] = 2  # February
    contract_row['primary_state'] = parse_primary_state(contract_row.get('project_name', ''))
    contract_row['latitude_start'] = contract_row.get('latitude', 0)
    contract_row['longitude_start'] = contract_row.get('longitude', 0)

    feature_columns = [
        "award_year", "award_month", "primary_state", "geopolitical_zone",
        "latitude_start", "longitude_start", "estimated_length_km",
        "terrain_type", "rainfall_mm_per_year", "soil_type", "elevation_m",
        "has_bridge", "is_dual_carriageway", "is_rehabilitation", "is_coastal_or_swamp",
        "boq_earthworks_m3_per_km", "boq_asphalt_ton_per_km", "boq_drainage_km_per_km",
        "boq_bridges_units", "boq_culverts_units", "boq_premium_percent"
    ]

    available = [col for col in feature_columns if col in contract_row.index]

    if not available:
        print("[WARNING] No model features available → fallback range")
        return 0, 0

    features = contract_row[available]
    features_df = pd.DataFrame([features.values], columns=features.index)

    try:
        base_price = model.predict(features_df)[0]
    except Exception as e:
        print(f"[ERROR] Model predict failed: {e}")
        base_price = 0

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

# (keep as is)

# ────────────────────────────────────────────────
#  ROUTES
# ────────────────────────────────────────────────

# (keep home, register, login, logout as is)

@app.get("/contracts", response_class=HTMLResponse)
def contracts(request: Request):
    try:
        user_id = get_current_user(request)

        with get_db() as (cur, _):
            cur.execute("SELECT company_name FROM users WHERE id = %s", (user_id,))
            company_row = cur.fetchone()
            if not company_row:
                return RedirectResponse(url="/login", status_code=303)
            company_name = company_row[0].strip()

            print(f"[DEBUG /contracts] user_id={user_id} | company_name='{company_name}'")

            cur.execute("SELECT contract_id FROM bids WHERE company_name = %s", (company_name,))
            existing_rows = cur.fetchall()
            existing = {r[0] for r in existing_rows}

            print(f"[DEBUG /contracts] Found {len(existing)} existing bid(s) for '{company_name}': {sorted(existing)}")

        all_contracts = df_bidding.to_dict(orient="records")
        available = []
        for i, contract in enumerate(all_contracts):
            if i not in existing:
                fair_min, fair_max = get_fair_price_range(pd.Series(contract))
                contract_with_range = contract.copy()
                contract_with_range['fair_min'] = fair_min
                contract_with_range['fair_max'] = fair_max
                available.append(contract_with_range)

        if not available:
            # (keep as is)

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

# (keep contract_detail as is)

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

        with get_db() as (cur, conn):
            cur.execute(
                "SELECT company_name, cac_number, email FROM users WHERE id = %s",
                (user_id,)
            )
            row = cur.fetchone()
            if row:
                company_name, cac_number, email = [v.strip() if isinstance(v, str) else v for v in row]

        clean = bid_amount.replace(",", "").strip()
        bid_value = float(clean)

        contract = df_bidding.iloc[contract_id]
        fair_min, fair_max = get_fair_price_range(contract)
        status_msg = "Approved ✅" if fair_min <= bid_value <= fair_max else "Rejected ❌"

        if fair_min == 0 and fair_max == 0:
            fair_min = bid_value * 0.7
            fair_max = bid_value * 1.3
            status_msg = "Pending Review (limited data)"

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

            print(f"[DEBUG submit_bid] INSERTED → bid_id={bid_id} | contract={contract_id} | company='{company_name}' | amount={bid_value} | status={status_msg}")

        email_ok = send_bid_notification(
            email, company_name, contract.get('project_name', 'Unknown contract'), status_msg, bid_value
        )

        color = "#10b981" if "Approved" in status_msg else "#ef4444"
        email_msg = "<p style='color:#10b981;font-weight:600;'>📧 Confirmation email sent!</p>" if email_ok else "<p style='color:#f59e0b;'>⚠️ Bid saved (email failed)</p>"

        return HTMLResponse(f"""
        <div style="max-width:750px;margin:40px auto;padding:40px;background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:3px solid #10b981;border-radius:20px;text-align:center;box-shadow:0 10px 30px rgba(16,185,129,0.25);">
            <h1 style="color:#065f46;font-size:2.2rem;margin-bottom:0.5rem;">🎉 BID SUBMITTED SUCCESSFULLY</h1>
            <p style="color:#0f766e;font-size:1.2rem;margin-bottom:2rem;">Your bid for "{contract.get('project_name', 'N/A')}" has been recorded (ID: {bid_id})</p>

            <!-- rest of success HTML as is -->
        </div>
        """)

    except Exception as e:
        print(f"[ERROR submit_bid] {type(e).__name__}: {str(e)}")
        return HTMLResponse(f"""
        <div style="max-width:650px;margin:50px auto;padding:30px;background:#fef2f2;border:3px solid #ef4444;border-radius:16px;text-align:center;">
            <h1 style="color:#991b1b;">Submission Failed</h1>
            <p>{str(e)[:240]}</p>
            <a href="/contracts" style="display:inline-block;margin-top:20px;padding:12px 30px;background:#1e40af;color:white;border-radius:8px;text-decoration:none;">← Back</a>
        </div>
        """, status_code=500)

# ── Admin approve/reject ─────────────────────────

@app.post("/admin/bids/{bid_id}/approve", response_class=RedirectResponse)
async def admin_approve_bid(bid_id: int, comments: str = Form(None)):
    try:
        with get_db() as (cur, conn):
            cur.execute(
                "UPDATE bids SET admin_status = %s, comments = %s WHERE id = %s",
                ('approved', comments, bid_id)
            )
        return RedirectResponse(url="/admin/dashboard", status_code=303)
    except Exception as e:
        print(f"[ERROR admin_approve] {e}")
        return RedirectResponse(url="/admin/dashboard", status_code=303)

@app.post("/admin/bids/{bid_id}/reject", response_class=RedirectResponse)
async def admin_reject_bid(bid_id: int, comments: str = Form(None)):
    try:
        with get_db() as (cur, conn):
            cur.execute(
                "UPDATE bids SET admin_status = %s, comments = %s WHERE id = %s",
                ('rejected', comments, bid_id)
            )
        return RedirectResponse(url="/admin/dashboard", status_code=303)
    except Exception as e:
        print(f"[ERROR admin_reject] {e}")
        return RedirectResponse(url="/admin/dashboard", status_code=303)

# ── Admin dashboard (updated with approve/reject) ──

@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    try:
        get_admin_user(request)

        with get_db() as (cur, _):
            cur.execute("""
            SELECT id, contract_id, company_name, cac_number, email, phone,
                   bid_amount, equipment_list, workforce, status, timestamp, admin_status, comments
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
                    "timestamp": bid[10],
                    "admin_status": bid[11] or "pending",
                    "comments": bid[12] or ""
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
                    "timestamp": bid[10],
                    "admin_status": bid[11] or "pending",
                    "comments": bid[12] or ""
                })

        total = len(enhanced)
        fair_count = sum(1 for x in enhanced if x["is_fair"])
        unfair_count = total - fair_count
        total_value = sum(x["bid_amount"] for x in enhanced)

        # ── HTML with approve/reject buttons ── (add to table rows)

        # (keep existing HTML, but update tbody loop to include forms)
        for b in enhanced:
            html += f"""
                <tr class="{cls}">
                    ...
                    <td>{b["admin_status"]}</td>
                    <td>{b["comments"]}</td>
                    <td>
                        <form method="POST" action="/admin/bids/{b['bid_id']}/approve">
                            <textarea name="comments" placeholder="Comments"></textarea>
                            <button type="submit">Approve ✅</button>
                        </form>
                        <form method="POST" action="/admin/bids/{b['bid_id']}/reject">
                            <textarea name="comments" placeholder="Comments"></textarea>
                            <button type="submit">Reject ❌</button>
                        </form>
                    </td>
                </tr>
                """

        # (keep rest of dashboard HTML)

    except Exception as e:
        print(f"Dashboard error: {e}")
        return HTMLResponse("<h2 style='color:red;text-align:center;'>Error loading dashboard</h2>")

# ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
</xai:function_call>
