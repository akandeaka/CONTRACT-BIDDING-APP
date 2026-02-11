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
            predicted_min REAL,
            predicted_max REAL,
            submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        )''')

        try:
            c.execute("INSERT INTO admins (username, hashed_password) VALUES (?, ?)",
                      ("admin", pwd_context.hash("AdminSecure2026!")))
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
except Exception as e:
    print(f"Failed to load Google Sheet: {e}")
    df_bidding = pd.DataFrame()

# ────────────────────────────────────────────────
# AI fairness check (using real contract features)
# ────────────────────────────────────────────────
def is_fair_bid(contract_id: int, bid_amount: float) -> tuple:
    if contract_id >= len(df_bidding):
        return "Under Review", 0, 0

    row = df_bidding.iloc[contract_id]

    # Base cost per km (placeholder — you can make this more sophisticated)
    base_cost_per_km = 1_000_000_000  # ₦1B per km as starting point

    # Adjust based on real factors
    length_factor = row.get("estimated_length_km", 100) / 100
    terrain_factor = {
        "Arid savanna": 0.8,
        "Semi-arid flat": 0.9,
        "Rainforest": 1.3,
        "Mangrove swamp": 1.6,
        "Hilly savanna": 1.2,
        # Add more terrain types as needed
    }.get(row.get("terrain_type", "Semi-arid flat"), 1.0)

    zone_factor = {
        "North West": 0.95,
        "North East": 1.05,
        "North Central": 1.0,
        "South West": 1.15,
        "South East": 1.1,
        "South South": 1.25,
    }.get(row.get("geopolitical_zone", "North Central"), 1.0)

    # Calculate predicted fair value
    predicted_value = base_cost_per_km * length_factor * terrain_factor * zone_factor * 1.15  # 15% contingency
    min_fair = predicted_value * 0.88
    max_fair = predicted_value * 1.12

    status = "Fair" if min_fair <= bid_amount <= max_fair else \
             "Too Low (Suspicious)" if bid_amount < min_fair else "Too High"

    return status, round(min_fair / 1e9, 2), round(max_fair / 1e9, 2)
# ────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse("/login", status_code=303)

# ── Register & Login (unchanged, kept clean) ─────────────────────────────────

# [Your existing register and login routes are fine – no need to change them]

# ── Contracts List (with description + location) ───────────────────────────────
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
                "project_name": row.get("Project_name", f"Contract {idx}"),
                "description": row.get("Description", "No description available"),
                "location": f"Lat: {row.get('latitude','N/A')}, Lon: {row.get('longitude','N/A')}",
                "terrain": row.get("terrain_type", "N/A"),
                "length_km": row.get("estimated_length_km", "N/A"),
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
    <!DOCTYPE html><html><head><title>Available Contracts</title>
    <style>body{{font-family:Arial;background:#f8fafc;padding:2rem;}} h1{{color:#1e40af;}} .container{{max-width:900px;margin:auto;}}</style>
    </head><body><div class="container">
        <h1>Available Contracts</h1>
        <a href="/logout" style="float:right;color:#ef4444;">Logout</a>
        {items}
    </div></body></html>
    """)

# ── Bid Submission (stores AI prediction) ─────────────────────────────────────
@app.post("/bid/{contract_id}", response_class=HTMLResponse)
@limiter.limit("3/hour")
async def submit_bid(request: Request, contract_id: int,
                     company_name: str = Form(...), cac_number: str = Form(...),
                     email: str = Form(...), phone: str = Form(...),
                     bid_amount: float = Form(...),
                     equipment_list: str = Form(...), workforce: str = Form(...),
                     db: sqlite3.Connection = Depends(get_db)):
    user_id = get_current_user_id(request)
    if contract_id < 0 or contract_id >= len(df_bidding):
        raise HTTPException(404, "Contract not found")
    if bid_amount <= 0:
        raise HTTPException(400, "Bid amount must be positive")


    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO bids (contract_id, user_id, company_name, cac_number, email, phone,
                          bid_amount, equipment_list, workforce, status, predicted_min, predicted_max)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (contract_id, user_id, company_name.strip(), cac_number.strip(), email.strip(), phone.strip(),
          bid_amount, equipment_list.strip(), workforce.strip(), status, min_fair, max_fair))
    db.commit()

    project_name = df_bidding.iloc[contract_id].get("Project_name", "Unknown project")

    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html><head><title>Success</title>
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

# ── Admin Dashboard (Beautiful + AI Range + Review Modal) ─────────────────────
@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, db: sqlite3.Connection = Depends(get_db)):
    get_current_admin_id(request)

    cursor = db.cursor()
    cursor.execute("""
        SELECT id, contract_id, company_name, bid_amount, status, submitted_at,
               predicted_min, predicted_max
        FROM bids
        ORDER BY submitted_at DESC
    """)
    bids = cursor.fetchall()

    total_bids = len(bids)
    approved = sum(1 for b in bids if b["status"] == "Approved")
    rejected = sum(1 for b in bids if b["status"] == "Rejected")
    pending = total_bids - approved - rejected

    rows = ""
    for b in bids:
        contract_id = b["contract_id"]
        project_name = df_bidding.iloc[contract_id]["Project_name"] \
            if 0 <= contract_id < len(df_bidding) \
            else f"Contract {contract_id}"

        min_fair = b["predicted_min"] if b["predicted_min"] is not None else "N/A"
        max_fair = b["predicted_max"] if b["predicted_max"] is not None else "N/A"

        variance = 0
        if isinstance(min_fair, (int, float)) and isinstance(max_fair, (int, float)):
            ai_mid = (min_fair + max_fair) / 2
            variance = ((b["bid_amount"] - ai_mid) / ai_mid) * 100 if ai_mid > 0 else 0

        variance_color = "green" if abs(variance) < 15 else "orange" if abs(variance) < 30 else "red"

        rows += f"""
        <tr>
            <td>#{b['id']}</td>
            <td>{project_name}</td>
            <td>{b['company_name']}</td>
            <td>₦{b['bid_amount']:,.2f}B</td>
            <td>₦{min_fair}B – ₦{max_fair}B</td>
            <td style="color:{'#10b981' if b['status'] == 'Approved' else '#ef4444' if b['status'] == 'Rejected' else '#f59e0b'}">{b['status']}</td>
            <td style="color:{variance_color}">{variance:+.1f}%</td>
            <td>{b['submitted_at'][:19]}</td>
            <td>
                <button onclick="openReviewModal({b['id']}, '{project_name.replace("'", "\\'")}', {b['bid_amount']}, {min_fair if min_fair != 'N/A' else 'null'}, {max_fair if max_fair != 'N/A' else 'null'})"
                        style="background:#3b82f6;color:white;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;">
                    Review & Decide
                </button>
            </td>
        </tr>
        """

    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AISEC Admin Control Panel</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {{ --primary: #1d4ed8; --success: #10b981; --danger: #ef4444; --warning: #f59e0b; }}
            body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f1f5f9; margin: 0; padding: 24px; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            .header {{ background: linear-gradient(135deg, var(--primary), #3b82f6); color: white; padding: 2rem; border-radius: 16px; text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(29,78,216,0.3); }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
            .stat-card {{ background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); text-align: center; transition: transform 0.2s; }}
            .stat-card:hover {{ transform: translateY(-4px); }}
            .stat-number {{ font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0; }}
            table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.08); }}
            th, td {{ padding: 16px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
            th {{ background: var(--primary); color: white; font-weight: 600; }}
            tr:hover {{ background: #f8fafc; }}
            .action-btn {{ padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; transition: all 0.2s; }}
            .action-btn:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
            .modal {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 1000; align-items: center; justify-content: center; }}
            .modal-content {{ background: white; width: 90%; max-width: 600px; border-radius: 16px; padding: 2rem; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }}
            textarea {{ width: 100%; min-height: 120px; padding: 12px; border: 1px solid #cbd5e1; border-radius: 8px; margin: 1rem 0; resize: vertical; font-size: 1rem; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>AISEC Admin Control Center</h1>
                <p>AI-Powered Fair Bidding Oversight System</p>
            </div>

            <div class="stats-grid">
                <div class="stat-card" style="border-left: 6px solid var(--primary);">
                    <div class="stat-number">{total_bids}</div>
                    <div>Total Bids</div>
                </div>
                <div class="stat-card" style="border-left: 6px solid var(--success);">
                    <div class="stat-number" style="color:var(--success)">{approved}</div>
                    <div>Approved</div>
                </div>
                <div class="stat-card" style="border-left: 6px solid var(--danger);">
                    <div class="stat-number" style="color:var(--danger)">{rejected}</div>
                    <div>Rejected</div>
                </div>
                <div class="stat-card" style="border-left: 6px solid var(--warning);">
                    <div class="stat-number" style="color:var(--warning)">{pending}</div>
                    <div>Pending Review</div>
                </div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Project</th>
                        <th>Company</th>
                        <th>Bid Amount</th>
                        <th>AI Fair Range</th>
                        <th>Status</th>
                        <th>Date</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>

        <!-- Review Modal -->
        <div id="reviewModal" class="modal">
            <div class="modal-content">
                <h2 id="modalTitle">Review Bid</h2>
                <p><strong>Project:</strong> <span id="modalProject"></span></p>
                <p><strong>Bid Amount:</strong> <span id="modalBid"></span></p>
                <p><strong>AI Fair Range:</strong> <span id="modalAIRange"></span></p>
                <p><strong>Variance from AI mid-point:</strong> <span id="modalVariance"></span></p>
                <label for="adminComment"><strong>Admin Comment / Rejection Reason (required for reject):</strong></label>
                <textarea id="adminComment" placeholder="Enter your analysis or reason for rejection..."></textarea>
                <div style="margin-top: 1.5rem; display: flex; gap: 1rem;">
                    <button onclick="submitDecision('Approved')" class="action-btn" style="background:var(--success);flex:1;">Approve Bid</button>
                    <button onclick="submitDecision('Rejected')" class="action-btn" style="background:var(--danger);flex:1;">Reject Bid</button>
                    <button onclick="closeModal()" class="action-btn" style="background:#64748b;flex:1;">Cancel</button>
                </div>
            </div>
        </div>

        <script>
            let currentBidId = null;

            function openReviewModal(bidId, project, bidAmount, aiMin, aiMax) {
                currentBidId = bidId;
                document.getElementById("modalProject").innerText = project;
                document.getElementById("modalBid").innerText = "₦" + Number(bidAmount).toLocaleString() + " Billion";
                document.getElementById("modalAIRange").innerText = aiMin === "N/A" ? "N/A" : "₦" + Number(aiMin).toLocaleString() + "B – ₦" + Number(aiMax).toLocaleString() + "B";

                let variance = "N/A";
                if (aiMin !== "N/A" && aiMax !== "N/A") {
                    let mid = (Number(aiMin) + Number(aiMax)) / 2;
                    variance = ((Number(bidAmount) - mid) / mid * 100).toFixed(1) + "%";
                }
                document.getElementById("modalVariance").innerText = variance;

                document.getElementById("reviewModal").style.display = "flex";
            }

            function closeModal() {
                document.getElementById("reviewModal").style.display = "none";
                document.getElementById("adminComment").value = "";
            }

            async function submitDecision(status) {
                const comment = document.getElementById("adminComment").value.trim();

                if (status === "Rejected" && !comment) {
                    alert("Please provide a reason for rejection");
                    return;
                }

                await fetch(`/admin/update-bid/${currentBidId}`, {
                    method: "POST",
                    headers: { "Content-Type": "application/x-www-form-urlencoded" },
                    body: `new_status=${status}&admin_comment=${encodeURIComponent(comment)}`
                });

                closeModal();
                location.reload();
            }
        </script>
    </body>
    </html>
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
