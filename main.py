import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import http.client  # For SMS via Twilio
import json
from fastapi import FastAPI, Request, Form, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import psycopg2
from contextlib import contextmanager
import os
import hashlib
import secrets
import re
from datetime import datetime

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# CORS (NO TRAILING SPACES)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://akandeaka.github.io", "http://localhost:8000", "https://aisec.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
TRAINING_DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTXlHZrU20uniUkjr-5Pis1pfJSOYDUiFVcML6UqW2Lu176_opvZPQvTGOpQZnNx02HyFf-jRYw3O8o/pub?output=csv"
BIDDING_CONTRACTS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS-nWpM2oCQ5xmda7a3tlLiRmMC2VaAdG4IhoQsypuVvbYDgtDaWn_bYcClrc35XUoHRvvMEISXTvCw/pub?output=csv"
MODEL_PATH = "model.pkl"
DATABASE_URL = os.getenv("DATABASE_URL")

# SMS Configuration (Twilio - ADD YOUR CREDENTIALS IN RENDER ENV VARS)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "YOUR_TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "YOUR_TWILIO_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+1234567890")

sessions = {}

@contextmanager
def get_db():
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
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
        # Users table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            company_name TEXT NOT NULL,
            cac_number TEXT NOT NULL
        );
        """)
        
        # Bids table with approval workflow columns
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
        
        # Admins table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        );
        """)
        
        # Insert default admin (only once)
        cur.execute(
            "INSERT INTO admins (username, hashed_password) VALUES (%s, %s) ON CONFLICT (username) DO NOTHING",
            ("admin", hashlib.sha256("admin123".encode()).hexdigest())
        )

init_database()

# ======================
# SMS & EMAIL NOTIFICATIONS
# ======================
def send_sms_notification(to_phone: str, message: str):
    """Send SMS via Twilio (add credentials to Render env vars)"""
    try:
        if "YOUR_TWILIO" in TWILIO_ACCOUNT_SID:
            print(f"[SMS PLACEHOLDER] To {to_phone}: {message}")
            return True
            
        conn = http.client.HTTPSConnection("api.twilio.com")
        payload = f"From={TWILIO_PHONE_NUMBER}&To={to_phone}&Body={message}"
        headers = {
            'Content-Type': "application/x-www-form-urlencoded",
            'Authorization': f"Basic {base64.b64encode(f'{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}'.encode()).decode()}"
        }
        conn.request("POST", f"/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json", payload, headers)
        res = conn.getresponse()
        data = res.read()
        return res.status == 201
    except Exception as e:
        print(f"SMS failed: {str(e)}")
        return False

def send_admin_decision_email(email: str, company_name: str, contract_name: str, decision: str, comments: str, bid_amount: float):
    """Send professional email notification for admin decision"""
    try:
        EMAIL_HOST = "smtp.gmail.com"
        EMAIL_PORT = 587
        EMAIL_USER = "aisec2025.notifications@gmail.com"
        EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "YOUR_16_CHAR_APP_PASSWORD")
        
        decision_emoji = "✅ APPROVED" if decision == "approved" else "❌ REJECTED"
        status_color = "#10b981" if decision == "approved" else "#ef4444"
        subject = f"AISEC Bid Decision: {decision_emoji} - {contract_name}"
        
        body = f"""Dear {company_name},

We are pleased to inform you that your bid for "{contract_name}" has been {decision.upper()} by the AISEC administration team.

📋 BID DETAILS:
• Contract: {contract_name}
• Bid Amount: ₦{bid_amount:,.2f} Billion
• Decision: {decision_emoji}
• Admin Comments: {comments if comments else "No additional comments"}

{'🎉 Congratulations! Our team will contact you shortly to discuss next steps and contract finalization.' if decision == 'approved' else '💡 We encourage you to review our AI fair pricing guidelines and consider bidding on future contracts where your pricing aligns with market rates.'}

Thank you for participating in the AISEC bidding process. We value your interest in contributing to Nigeria's infrastructure development.

Best regards,
AISEC Administration Team
AI for Secure and Efficient Contracting
📧 admin@aisec.gov.ng | 🌐 https://aisec.gov.ng
"""
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)
        
        print(f"✓ Decision email sent to {email} for {decision} bid")
        return True
    except Exception as e:
        print(f"✗ Email failed: {str(e)}")
        return False

# ======================
# ADMIN DASHBOARD (PROFESSIONAL VERSION)
# ======================
@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    try:
        get_admin_user(request)
        
        with get_db() as (cur, _):
            # Get ALL bids with admin_status
            cur.execute("""
                SELECT id, contract_id, company_name, cac_number, email, phone,
                       bid_amount, equipment_list, workforce, status, timestamp,
                       admin_status, comments
                FROM bids 
                ORDER BY timestamp DESC
            """)
            all_bids = cur.fetchall()
        
        # Separate bids into sections
        pending_bids = []
        approved_bids = []
        rejected_bids = []
        
        for bid in all_bids:
            bid_dict = {
                'id': bid[0],
                'contract_id': bid[1],
                'company_name': bid[2] or 'N/A',
                'cac_number': bid[3] or 'N/A',
                'email': bid[4],
                'phone': bid[5],
                'bid_amount': bid[6],
                'equipment_list': bid[7],
                'workforce': bid[8],
                'status': bid[9],
                'timestamp': bid[10],
                'admin_status': bid[11] or 'pending',
                'comments': bid[12] or ''
            }
            
            # Get contract name from dataset
            try:
                contract_row = df_bidding.iloc[bid[1]]
                bid_dict['contract_name'] = contract_row.get('project_name', f'Contract {bid[1]}')
            except:
                bid_dict['contract_name'] = f'Contract ID {bid[1]}'
            
            # Categorize by admin_status
            if bid_dict['admin_status'] == 'approved':
                approved_bids.append(bid_dict)
            elif bid_dict['admin_status'] == 'rejected':
                rejected_bids.append(bid_dict)
            else:
                pending_bids.append(bid_dict)
        
        # Calculate statistics
        total_bids = len(all_bids)
        pending_count = len(pending_bids)
        approved_count = len(approved_bids)
        rejected_count = len(rejected_bids)
        total_value = sum(b['bid_amount'] for b in all_bids)
        ai_approved = sum(1 for b in all_bids if 'Approved' in b['status'])
        ai_flagged = total_bids - ai_approved
        
        # AI Insights Data (for charts)
        fraud_risk_percentage = round((rejected_count / total_bids * 100) if total_bids > 0 else 0, 1)
        avg_bid_amount = round(total_value / total_bids, 2) if total_bids > 0 else 0
        
        # Build professional dashboard HTML
        dashboard_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AISEC Admin Dashboard | Bid Management</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                :root {{
                    --primary: #2563eb; --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
                    --pending: #3b82f6; --approved: #10b981; --rejected: #ef4444;
                    --dark: #0f172a; --light: #f8fafc; --gray: #64748b;
                }}
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                    color: var(--dark); 
                    line-height: 1.6;
                }}
                .header {{
                    background: linear-gradient(120deg, var(--dark), #1e3a8a);
                    color: white;
                    padding: 1.5rem 3rem;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                    position: sticky;
                    top: 0;
                    z-index: 1000;
                }}
                .logo {{ font-size: 1.8rem; font-weight: 800; display: flex; align-items: center; gap: 12px; }}
                .logo i {{ font-size: 2rem; }}
                .container {{ max-width: 1800px; margin: 2rem auto; padding: 0 2rem; }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 2rem;
                }}
                .stat-card {{
                    background: white;
                    border-radius: 20px;
                    padding: 1.8rem;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
                    text-align: center;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    border-top: 5px solid var(--primary);
                    position: relative;
                    overflow: hidden;
                }}
                .stat-card:hover {{ transform: translateY(-5px); box-shadow: 0 15px 35px rgba(0,0,0,0.12); }}
                .stat-card::before {{
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
                    transform: scale(0);
                    transition: transform 0.5s ease;
                    z-index: 0;
                }}
                .stat-card:hover::before {{ transform: scale(1); }}
                .stat-value {{
                    font-size: 2.8rem;
                    font-weight: 800;
                    margin: 0.5rem 0;
                    background: linear-gradient(135deg, var(--primary), #0ea5e9);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    position: relative;
                    z-index: 1;
                }}
                .stat-label {{ 
                    color: var(--gray); 
                    font-size: 1.1rem; 
                    font-weight: 600;
                    position: relative;
                    z-index: 1;
                }}
                .pending-card {{ border-top-color: var(--pending); }}
                .pending-card .stat-value {{ background: linear-gradient(135deg, var(--pending), #38bdf8); }}
                .approved-card {{ border-top-color: var(--approved); }}
                .approved-card .stat-value {{ background: linear-gradient(135deg, var(--approved), #34d399); }}
                .rejected-card {{ border-top-color: var(--rejected); }}
                .rejected-card .stat-value {{ background: linear-gradient(135deg, var(--rejected), #f87171); }}
                .insights-card {{ border-top-color: #8b5cf6; }}
                .insights-card .stat-value {{ background: linear-gradient(135deg, #8b5cf6, #a78bfa); }}
                
                .section-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin: 2.5rem 0 1.5rem;
                    padding-bottom: 0.8rem;
                    border-bottom: 3px solid var(--primary);
                }}
                .section-title {{ 
                    font-size: 1.8rem; 
                    font-weight: 800; 
                    color: var(--dark);
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                .section-title i {{ font-size: 1.9rem; }}
                .bid-section {{
                    background: white;
                    border-radius: 24px;
                    box-shadow: 0 15px 50px rgba(0,0,0,0.1);
                    margin-bottom: 2.5rem;
                    overflow: hidden;
                }}
                .section-badge {{
                    background: var(--pending);
                    color: white;
                    padding: 0.4rem 1.1rem;
                    border-radius: 50px;
                    font-weight: 700;
                    font-size: 1.1rem;
                }}
                .approved-section .section-badge {{ background: var(--approved); }}
                .rejected-section .section-badge {{ background: var(--rejected); }}
                
                table {{ width: 100%; border-collapse: collapse; }}
                th {{
                    background: linear-gradient(135deg, var(--dark), #1e3a8a);
                    color: white;
                    padding: 1.4rem 1.2rem;
                    text-align: left;
                    font-weight: 700;
                    font-size: 0.95rem;
                    position: sticky;
                    top: 70px;
                    z-index: 90;
                }}
                td {{
                    padding: 1.3rem 1.2rem;
                    border-bottom: 1px solid #f1f5f9;
                    font-size: 0.95rem;
                    color: #1e293b;
                }}
                tr:hover {{ background: #f8fafc; }}
                .pending-row {{ border-left: 5px solid var(--pending); }}
                .approved-row {{ border-left: 5px solid var(--approved); background: #f0fdf4; }}
                .rejected-row {{ border-left: 5px solid var(--rejected); background: #fff1f2; }}
                
                .action-cell {{
                    display: flex;
                    gap: 10px;
                    flex-wrap: wrap;
                }}
                .btn {{
                    padding: 0.65rem 1.4rem;
                    border-radius: 12px;
                    font-weight: 600;
                    font-size: 0.9rem;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    border: none;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    gap: 6px;
                }}
                .btn-approve {{ 
                    background: linear-gradient(135deg, var(--approved), #0da27e); 
                    color: white;
                }}
                .btn-approve:hover {{ transform: translateY(-1px); box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4); }}
                .btn-reject {{ 
                    background: linear-gradient(135deg, var(--rejected), #c82333); 
                    color: white;
                }}
                .btn-reject:hover {{ transform: translateY(-1px); box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4); }}
                .btn-view {{ 
                    background: linear-gradient(135deg, var(--primary), #1d4ed8); 
                    color: white;
                }}
                .btn-view:hover {{ transform: translateY(-1px); box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4); }}
                .btn:disabled {{ opacity: 0.6; cursor: not-allowed; transform: none; }}
                
                .comment-cell {{ 
                    font-style: italic; 
                    color: var(--gray); 
                    max-width: 250px; 
                    line-height: 1.5;
                }}
                .timestamp {{ 
                    color: var(--gray); 
                    font-family: 'Segoe UI', monospace; 
                    font-size: 0.85rem;
                }}
                .amount-highlight {{ 
                    font-weight: 800; 
                    font-size: 1.1rem;
                    background: linear-gradient(135deg, var(--primary), #0ea5e9);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }}
                .status-badge {{
                    padding: 0.35rem 0.9rem;
                    border-radius: 50px;
                    font-weight: 600;
                    font-size: 0.85rem;
                }}
                .status-approved {{ background: #dcfce7; color: #15803d; }}
                .status-rejected {{ background: #fee2e2; color: #b91c1c; }}
                .status-pending {{ background: #dbeafe; color: #1e40af; }}
                
                .insights-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                    gap: 2rem;
                    margin-top: 2rem;
                }}
                .chart-container {{
                    background: white;
                    border-radius: 24px;
                    padding: 1.8rem;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
                }}
                .chart-title {{
                    font-size: 1.4rem;
                    font-weight: 700;
                    margin-bottom: 1.5rem;
                    color: var(--dark);
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                .chart-title i {{ font-size: 1.5rem; }}
                canvas {{ max-width: 100%; }}
                
                .empty-state {{
                    text-align: center;
                    padding: 4rem 2rem;
                    color: var(--gray);
                }}
                .empty-state i {{ 
                    font-size: 5rem; 
                    margin-bottom: 1.5rem; 
                    opacity: 0.3;
                    display: block;
                }}
                .empty-state p {{ font-size: 1.2rem; margin-top: 1rem; max-width: 600px; margin-left: auto; margin-right: auto; }}
                
                .modal {{
                    display: none;
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0,0,0,0.6);
                    z-index: 2000;
                    justify-content: center;
                    align-items: center;
                }}
                .modal-content {{
                    background: white;
                    border-radius: 24px;
                    width: 90%;
                    max-width: 600px;
                    padding: 2.5rem;
                    box-shadow: 0 25px 80px rgba(0,0,0,0.3);
                    animation: modalSlide 0.4s ease-out;
                }}
                @keyframes modalSlide {{
                    from {{ opacity: 0; transform: translateY(-50px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
                .modal-title {{
                    font-size: 1.8rem;
                    font-weight: 800;
                    margin-bottom: 1.5rem;
                    color: var(--dark);
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }}
                .modal-title i {{ font-size: 2rem; }}
                textarea {{
                    width: 100%;
                    padding: 1.2rem;
                    border: 2px solid #e2e8f0;
                    border-radius: 16px;
                    font-family: inherit;
                    font-size: 1rem;
                    margin: 1rem 0;
                    min-height: 120px;
                    resize: vertical;
                    transition: border-color 0.3s;
                }}
                textarea:focus {{
                    outline: none;
                    border-color: var(--primary);
                    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
                }}
                .modal-actions {{
                    display: flex;
                    justify-content: flex-end;
                    gap: 15px;
                    margin-top: 1.5rem;
                }}
                .btn-modal {{
                    padding: 0.9rem 2rem;
                    border-radius: 14px;
                    font-weight: 700;
                    font-size: 1.05rem;
                    cursor: pointer;
                    border: none;
                    min-width: 120px;
                }}
                .btn-confirm {{ 
                    background: linear-gradient(135deg, var(--approved), #0da27e); 
                    color: white;
                }}
                .btn-cancel {{ 
                    background: #f1f5f9; 
                    color: var(--dark);
                }}
                .btn-cancel:hover {{ background: #e2e8f0; }}
                
                .logout-btn {{
                    background: linear-gradient(135deg, var(--rejected), #b91c1c);
                    color: white;
                    padding: 0.85rem 1.8rem;
                    border-radius: 16px;
                    text-decoration: none;
                    font-weight: 700;
                    font-size: 1.05rem;
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.35);
                    transition: all 0.3s ease;
                }}
                .logout-btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(239, 68, 68, 0.45);
                }}
                
                @media (max-width: 900px) {{
                    .stats-grid {{ grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }}
                    .insights-container {{ grid-template-columns: 1fr; }}
                    th, td {{ padding: 1rem 0.8rem; font-size: 0.85rem; }}
                    .section-header {{ flex-direction: column; align-items: flex-start; gap: 1rem; }}
                    .action-cell {{ flex-direction: column; }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="logo">
                    <span>🛡️</span> AISEC ADMIN DASHBOARD
                </div>
                <a href="/admin/logout" class="logout-btn">
                    <span>🚪</span> Logout
                </a>
            </div>
            
            <div class="container">
                <h1 style="font-size: 2.5rem; font-weight: 800; margin: 1.5rem 0 2rem; color: var(--dark);">
                    📊 Bid Management & AI Analytics Hub
                </h1>
                
                <!-- STATISTICS CARDS -->
                <div class="stats-grid">
                    <div class="stat-card pending-card">
                        <div style="font-size: 2.2rem; margin-bottom: 0.5rem">⏳</div>
                        <div class="stat-value">{pending_count}</div>
                        <div class="stat-label">PENDING BIDS</div>
                        <div style="margin-top: 0.8rem; font-size: 0.95rem; color: var(--gray);">
                            Require admin review
                        </div>
                    </div>
                    <div class="stat-card approved-card">
                        <div style="font-size: 2.2rem; margin-bottom: 0.5rem">✅</div>
                        <div class="stat-value">{approved_count}</div>
                        <div class="stat-label">APPROVED BIDS</div>
                        <div style="margin-top: 0.8rem; font-size: 0.95rem; color: var(--gray);">
                            Ready for contract finalization
                        </div>
                    </div>
                    <div class="stat-card rejected-card">
                        <div style="font-size: 2.2rem; margin-bottom: 0.5rem">❌</div>
                        <div class="stat-value">{rejected_count}</div>
                        <div class="stat-label">REJECTED BIDS</div>
                        <div style="margin-top: 0.8rem; font-size: 0.95rem; color: var(--gray);">
                            Require bidder clarification
                        </div>
                    </div>
                    <div class="stat-card insights-card">
                        <div style="font-size: 2.2rem; margin-bottom: 0.5rem">💡</div>
                        <div class="stat-value">₦{total_value:,.1f}B</div>
                        <div class="stat-label">TOTAL BID VALUE</div>
                        <div style="margin-top: 0.8rem; font-size: 0.95rem; color: var(--gray);">
                            Avg: ₦{avg_bid_amount:,.2f}B per bid
                        </div>
                    </div>
                </div>
                
                <!-- AI INSIGHTS SECTION -->
                <div class="section-header">
                    <div class="section-title">
                        <span>🧠</span> AI-Powered Fraud Detection Insights
                    </div>
                </div>
                <div class="insights-container">
                    <div class="chart-container">
                        <div class="chart-title">
                            <span>📊</span> Bid Status Distribution
                        </div>
                        <canvas id="statusChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <div class="chart-title">
                            <span>⚠️</span> Fraud Risk Analysis
                        </div>
                        <div style="text-align: center; padding: 2rem 1rem;">
                            <div style="font-size: 4.5rem; font-weight: 800; background: linear-gradient(135deg, var(--warning), #ea580c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem;">
                                {fraud_risk_percentage}%
                            </div>
                            <div style="font-size: 1.4rem; font-weight: 700; color: var(--dark); margin-bottom: 0.8rem;">
                                Fraud Risk Level
                            </div>
                            <div style="color: var(--gray); line-height: 1.6;">
                                Based on AI analysis of {total_bids} bids. {rejected_count} bids rejected due to pricing anomalies exceeding fair market range.
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- PENDING BIDS SECTION -->
                <div class="section-header">
                    <div class="section-title">
                        <span>⏳</span> PENDING BIDS FOR REVIEW
                    </div>
                    <div class="section-badge">Requires Action</div>
                </div>
                <div class="bid-section pending-section">
                    <table>
                        <thead>
                            <tr>
                                <th>Bid ID</th>
                                <th>Contract</th>
                                <th>Company / CAC</th>
                                <th>Contact</th>
                                <th>Bid Amount (₦B)</th>
                                <th>AI Assessment</th>
                                <th>Submitted</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        # Add pending bids table rows
        if not pending_bids:
            dashboard_html += """
                            <tr>
                                <td colspan="8" class="empty-state">
                                    <i>📭</i>
                                    <h3 style="font-size: 1.8rem; margin: 1.5rem 0; color: var(--dark);">No Pending Bids</h3>
                                    <p>All bids have been reviewed. Great job keeping up with submissions!</p>
                                </td>
                            </tr>
            """
        else:
            for bid in pending_bids:
                ai_status = "✅ FAIR" if "Approved" in bid['status'] else "⚠️ INFLATED"
                ai_class = "status-approved" if "Approved" in bid['status'] else "status-rejected"
                dashboard_html += f"""
                            <tr class="pending-row">
                                <td><strong>#{bid['id']}</strong></td>
                                <td><strong>{bid['contract_name']}</strong></td>
                                <td>
                                    <div style="font-weight: 600; color: var(--dark);">{bid['company_name']}</div>
                                    <div style="color: var(--gray); font-size: 0.9rem; margin-top: 4px;">CAC: {bid['cac_number']}</div>
                                </td>
                                <td>
                                    <div>{bid['email']}</div>
                                    <div style="color: var(--gray); font-size: 0.9rem; margin-top: 2px;">{bid['phone']}</div>
                                </td>
                                <td><span class="amount-highlight">₦{bid['bid_amount']:,.2f}</span></td>
                                <td><span class="status-badge {ai_class}">{ai_status}</span></td>
                                <td class="timestamp">{bid['timestamp']}</td>
                                <td class="action-cell">
                                    <button class="btn btn-approve" onclick="openApproveModal({bid['id']}, '{bid['company_name']}', '{bid['contract_name']}')">
                                        <span>✅</span> Approve
                                    </button>
                                    <button class="btn btn-reject" onclick="openRejectModal({bid['id']}, '{bid['company_name']}', '{bid['contract_name']}')">
                                        <span>❌</span> Reject
                                    </button>
                                    <button class="btn btn-view" onclick="viewBidDetails({bid['id']})">
                                        <span>👁️</span> View
                                    </button>
                                </td>
                            </tr>
                """
        
        # Add Approved Bids Section
        dashboard_html += """
                        </tbody>
                    </table>
                </div>
                
                <!-- APPROVED BIDS SECTION -->
                <div class="section-header">
                    <div class="section-title">
                        <span>✅</span> APPROVED BIDS
                    </div>
                    <div class="section-badge">Contract Ready</div>
                </div>
                <div class="bid-section approved-section">
                    <table>
                        <thead>
                            <tr>
                                <th>Bid ID</th>
                                <th>Contract</th>
                                <th>Company / CAC</th>
                                <th>Contact</th>
                                <th>Bid Amount (₦B)</th>
                                <th>AI Assessment</th>
                                <th>Approved On</th>
                                <th>Admin Comments</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        if not approved_bids:
            dashboard_html += """
                            <tr>
                                <td colspan="8" class="empty-state">
                                    <i>✅</i>
                                    <h3 style="font-size: 1.8rem; margin: 1.5rem 0; color: var(--dark);">No Approved Bids Yet</h3>
                                    <p>Approve pending bids to move them here. Approved bids are ready for contract finalization.</p>
                                </td>
                            </tr>
            """
        else:
            for bid in approved_bids:
                ai_status = "✅ FAIR" if "Approved" in bid['status'] else "⚠️ INFLATED"
                ai_class = "status-approved" if "Approved" in bid['status'] else "status-rejected"
                dashboard_html += f"""
                            <tr class="approved-row">
                                <td><strong>#{bid['id']}</strong></td>
                                <td><strong>{bid['contract_name']}</strong></td>
                                <td>
                                    <div style="font-weight: 600; color: var(--dark);">{bid['company_name']}</div>
                                    <div style="color: var(--gray); font-size: 0.9rem; margin-top: 4px;">CAC: {bid['cac_number']}</div>
                                </td>
                                <td>
                                    <div>{bid['email']}</div>
                                    <div style="color: var(--gray); font-size: 0.9rem; margin-top: 2px;">{bid['phone']}</div>
                                </td>
                                <td><span class="amount-highlight">₦{bid['bid_amount']:,.2f}</span></td>
                                <td><span class="status-badge {ai_class}">{ai_status}</span></td>
                                <td class="timestamp">{bid['timestamp']}</td>
                                <td class="comment-cell">{bid['comments'] or 'No comments'}</td>
                            </tr>
                """
        
        # Add Rejected Bids Section
        dashboard_html += """
                        </tbody>
                    </table>
                </div>
                
                <!-- REJECTED BIDS SECTION -->
                <div class="section-header">
                    <div class="section-title">
                        <span>❌</span> REJECTED BIDS
                    </div>
                    <div class="section-badge">Requires Clarification</div>
                </div>
                <div class="bid-section rejected-section">
                    <table>
                        <thead>
                            <tr>
                                <th>Bid ID</th>
                                <th>Contract</th>
                                <th>Company / CAC</th>
                                <th>Contact</th>
                                <th>Bid Amount (₦B)</th>
                                <th>AI Assessment</th>
                                <th>Rejected On</th>
                                <th>Admin Comments</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        if not rejected_bids:
            dashboard_html += """
                            <tr>
                                <td colspan="8" class="empty-state">
                                    <i>❌</i>
                                    <h3 style="font-size: 1.8rem; margin: 1.5rem 0; color: var(--dark);">No Rejected Bids Yet</h3>
                                    <p>Reject bids that require clarification or exceed fair pricing thresholds.</p>
                                </td>
                            </tr>
            """
        else:
            for bid in rejected_bids:
                ai_status = "✅ FAIR" if "Approved" in bid['status'] else "⚠️ INFLATED"
                ai_class = "status-approved" if "Approved" in bid['status'] else "status-rejected"
                dashboard_html += f"""
                            <tr class="rejected-row">
                                <td><strong>#{bid['id']}</strong></td>
                                <td><strong>{bid['contract_name']}</strong></td>
                                <td>
                                    <div style="font-weight: 600; color: var(--dark);">{bid['company_name']}</div>
                                    <div style="color: var(--gray); font-size: 0.9rem; margin-top: 4px;">CAC: {bid['cac_number']}</div>
                                </td>
                                <td>
                                    <div>{bid['email']}</div>
                                    <div style="color: var(--gray); font-size: 0.9rem; margin-top: 2px;">{bid['phone']}</div>
                                </td>
                                <td><span class="amount-highlight">₦{bid['bid_amount']:,.2f}</span></td>
                                <td><span class="status-badge {ai_class}">{ai_status}</span></td>
                                <td class="timestamp">{bid['timestamp']}</td>
                                <td class="comment-cell">{bid['comments'] or 'No comments'}</td>
                            </tr>
                """
        
        # Close tables and add modals/scripts
        dashboard_html += """
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- APPROVE MODAL -->
            <div id="approveModal" class="modal">
                <div class="modal-content">
                    <div class="modal-title">
                        <span>✅</span> Approve Bid
                    </div>
                    <p id="approveBidInfo" style="margin-bottom: 1.5rem; color: var(--gray);"></p>
                    <textarea id="approveComments" placeholder="Optional: Add comments for the bidder (e.g., next steps, requirements)"></textarea>
                    <div class="modal-actions">
                        <button class="btn-modal btn-cancel" onclick="closeModal('approveModal')">Cancel</button>
                        <button class="btn-modal btn-confirm" onclick="submitApprove()">Confirm Approval</button>
                    </div>
                </div>
            </div>
            
            <!-- REJECT MODAL -->
            <div id="rejectModal" class="modal">
                <div class="modal-content">
                    <div class="modal-title">
                        <span>❌</span> Reject Bid
                    </div>
                    <p id="rejectBidInfo" style="margin-bottom: 1.5rem; color: var(--gray);"></p>
                    <textarea id="rejectComments" placeholder="Required: Explain why this bid was rejected (e.g., pricing exceeds fair range, missing documentation)" required></textarea>
                    <div class="modal-actions">
                        <button class="btn-modal btn-cancel" onclick="closeModal('rejectModal')">Cancel</button>
                        <button class="btn-modal btn-confirm" style="background: linear-gradient(135deg, var(--rejected), #c82333);" onclick="submitReject()">Confirm Rejection</button>
                    </div>
                </div>
            </div>
            
            <script>
                let currentBidId = null;
                
                function openApproveModal(bidId, company, contract) {
                    currentBidId = bidId;
                    document.getElementById('approveBidInfo').textContent = `Approving bid #${bidId} from ${company} for "${contract}"`;
                    document.getElementById('approveComments').value = '';
                    document.getElementById('approveModal').style.display = 'flex';
                }
                
                function openRejectModal(bidId, company, contract) {
                    currentBidId = bidId;
                    document.getElementById('rejectBidInfo').textContent = `Rejecting bid #${bidId} from ${company} for "${contract}"`;
                    document.getElementById('rejectComments').value = '';
                    document.getElementById('rejectModal').style.display = 'flex';
                }
                
                function closeModal(modalId) {
                    document.getElementById(modalId).style.display = 'none';
                }
                
                function submitApprove() {
                    const comments = document.getElementById('approveComments').value;
                    window.location.href = `/admin/bids/${currentBidId}/approve?comments=${encodeURIComponent(comments)}`;
                }
                
                function submitReject() {
                    const comments = document.getElementById('rejectComments').value;
                    if (!comments.trim()) {
                        alert('Please provide rejection reason');
                        return;
                    }
                    window.location.href = `/admin/bids/${currentBidId}/reject?comments=${encodeURIComponent(comments)}`;
                }
                
                // Initialize Charts
                document.addEventListener('DOMContentLoaded', function() {{
                    const ctx1 = document.getElementById('statusChart').getContext('2d');
                    new Chart(ctx1, {{
                        type: 'doughnut',
                        data: {{
                            labels: ['Pending', 'Approved', 'Rejected'],
                            datasets: [{{
                                 [{pending_count}, {approved_count}, {rejected_count}],
                                backgroundColor: [
                                    'rgba(59, 130, 246, 0.85)',
                                    'rgba(16, 185, 129, 0.85)',
                                    'rgba(239, 68, 68, 0.85)'
                                ],
                                borderColor: [
                                    'rgba(59, 130, 246, 1)',
                                    'rgba(16, 185, 129, 1)',
                                    'rgba(239, 68, 68, 1)'
                                ],
                                borderWidth: 2
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{
                                    position: 'bottom',
                                    labels: {{
                                        padding: 20,
                                        font: {{
                                            size: 13
                                        }}
                                    }}
                                }},
                                tooltip: {{
                                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                                    titleFont: {{
                                        size: 14
                                    }},
                                    bodyFont: {{
                                        size: 13
                                    }}
                                }}
                            }}
                        }}
                    }});
                }});
            </script>
            
            <div style="text-align: center; padding: 2.5rem; margin-top: 2rem; background: white; border-radius: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.08);">
                <p style="color: var(--gray); font-size: 1.1rem; max-width: 800px; margin: 0 auto;">
                    <strong style="color: var(--primary);">🛡️ AISEC</strong> - AI for Secure and Efficient Contracting • Real-time fraud detection since 2026
                </p>
                <p style="margin-top: 1rem; font-weight: 600; color: var(--primary);">
                    System Status: <span style="color: var(--approved);">✅ All Systems Operational</span>
                </p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(dashboard_html)
    
    except HTTPException:
        return RedirectResponse(url="/admin/login", status_code=303)
    except Exception as e:
        print(f"✗✗✗ ADMIN DASHBOARD ERROR: {str(e)}")
        return HTMLResponse(f"""
        <div style='max-width:700px;margin:50px auto;background:#fef2f2;border:3px solid #ef4444;border-radius:20px;padding:40px;text-align:center'>
            <div style='font-size:64px;margin-bottom:20px'>⚠️</div>
            <h1 style='color:#991b1b;margin-bottom:15px'>Admin Dashboard Error</h1>
            <p style='color:#991b1b;font-size:18px;margin-bottom:25px'>Unable to load bid data</p>
            <div style='background:#fee2e2;padding:20px;border-radius:12px;margin:20px 0;font-family:monospace;color:#b91c1c;text-align:left;overflow:auto;max-height:200px'>{str(e)}</div>
            <a href='/admin/login' style='display:inline-block;margin-top:20px;padding:14px 35px;background:#1e40af;color:white;text-decoration:none;border-radius:10px;font-weight:600;font-size:16px'>⇦ Return to Login</a>
        </div>
        """, status_code=500)

# ======================
# ADMIN APPROVAL/REJECTION ENDPOINTS (WITH NOTIFICATIONS)
# ======================
@app.post("/admin/bids/{bid_id}/approve")
async def admin_approve_bid(bid_id: int, comments: str = ""):
    try:
        with get_db() as (cur, conn):
            # Get bid details BEFORE update
            cur.execute("""
                SELECT company_name, email, phone, contract_id, bid_amount 
                FROM bids WHERE id = %s
            """, (bid_id,))
            bid_details = cur.fetchone()
            
            if not bid_details:
                raise HTTPException(status_code=404, detail="Bid not found")
            
            # Update bid status
            cur.execute("""
                UPDATE bids 
                SET admin_status = 'approved', comments = %s 
                WHERE id = %s
            """, (comments or "Approved by admin", bid_id))
            
            # Get contract name
            contract_name = "Unknown Contract"
            try:
                contract_row = df_bidding.iloc[bid_details[3]]
                contract_name = contract_row.get('project_name', contract_name)
            except:
                pass
            
            # SEND NOTIFICATIONS (in background thread for production)
            company_name = bid_details[0]
            email = bid_details[1]
            phone = bid_details[2]
            bid_amount = bid_details[4]
            
            # Send email notification
            email_sent = send_admin_decision_email(
                email, 
                company_name, 
                contract_name, 
                "approved", 
                comments, 
                bid_amount
            )
            
            # Send SMS notification (Twilio)
            sms_sent = send_sms_notification(
                phone,
                f"AISEC: Your bid for '{contract_name}' has been APPROVED! Amount: ₦{bid_amount:,.2f}B. {comments if comments else 'Check email for details.'} - AISEC Admin"
            )
            
            print(f"✓ Bid #{bid_id} APPROVED | Email: {'Sent' if email_sent else 'Failed'} | SMS: {'Sent' if sms_sent else 'Failed'}")
        
        return RedirectResponse(url="/admin/dashboard", status_code=303)
    
    except Exception as e:
        print(f"[ERROR admin_approve] {e}")
        return RedirectResponse(url="/admin/dashboard?error=approval_failed", status_code=303)

@app.post("/admin/bids/{bid_id}/reject")
async def admin_reject_bid(bid_id: int, comments: str = Form(...)):
    try:
        if not comments.strip():
            return RedirectResponse(url=f"/admin/dashboard?error=comments_required", status_code=303)
        
        with get_db() as (cur, conn):
            # Get bid details BEFORE update
            cur.execute("""
                SELECT company_name, email, phone, contract_id, bid_amount 
                FROM bids WHERE id = %s
            """, (bid_id,))
            bid_details = cur.fetchone()
            
            if not bid_details:
                raise HTTPException(status_code=404, detail="Bid not found")
            
            # Update bid status
            cur.execute("""
                UPDATE bids 
                SET admin_status = 'rejected', comments = %s 
                WHERE id = %s
            """, (comments, bid_id))
            
            # Get contract name
            contract_name = "Unknown Contract"
            try:
                contract_row = df_bidding.iloc[bid_details[3]]
                contract_name = contract_row.get('project_name', contract_name)
            except:
                pass
            
            # SEND NOTIFICATIONS
            company_name = bid_details[0]
            email = bid_details[1]
            phone = bid_details[2]
            bid_amount = bid_details[4]
            
            # Send email notification
            email_sent = send_admin_decision_email(
                email, 
                company_name, 
                contract_name, 
                "rejected", 
                comments, 
                bid_amount
            )
            
            # Send SMS notification
            sms_sent = send_sms_notification(
                phone,
                f"AISEC: Your bid for '{contract_name}' was REJECTED. Reason: {comments[:100]} - Contact admin for details. - AISEC Admin"
            )
            
            print(f"✓ Bid #{bid_id} REJECTED | Email: {'Sent' if email_sent else 'Failed'} | SMS: {'Sent' if sms_sent else 'Failed'}")
        
        return RedirectResponse(url="/admin/dashboard", status_code=303)
    
    except Exception as e:
        print(f"[ERROR admin_reject] {e}")
        return RedirectResponse(url="/admin/dashboard?error=rejection_failed", status_code=303)

# ======================
# HELPER FUNCTIONS (ADD TO EXISTING CODE)
# ======================
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

def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    sessions[token] = user_id
    return token

