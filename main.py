import os
from fastapi import FastAPI, Request, Form, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import hashlib
import secrets

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





# JWT Configuration
import secrets
from datetime import datetime, timedelta

# Simple session storage (in production, use Redis or database)
sessions = {}

def create_session_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    sessions[token] = {
        "user_id": user_id,
        "expires_at": datetime.utcnow() + timedelta(minutes=30)
    }
    return token

def get_current_user(token: str = None):
    if not token or token not in sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    session = sessions[token]
    if datetime.utcnow() > session["expires_at"]:
        del sessions[token]
        raise HTTPException(status_code=401, detail="Session expired")
    
    cursor.execute("SELECT id, email FROM users WHERE id = ?", (session["user_id"],))
    user = cursor.fetchone()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return {"id": user[0], "email": user[1]}


# Add CORS middleware
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

# URLs for both datasets (NO trailing spaces)
TRAINING_DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTXlHZrU20uniUkjr-5Pis1pfJSOYDUiFVcML6UqW2Lu176_opvZPQvTGOpQZnNx02HyFf-jRYw3O8o/pub?output=csv"
BIDDING_CONTRACTS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS-nWpM2oCQ5xmda7a3tlLiRmMC2VaAdG4IhoQsypuVvbYDgtDaWn_bYcClrc35XUoHRvvMEISXTvCw/pub?output=csv"

MODEL_PATH = "model.pkl"

def ensure_model_and_data():
    """Train model using training data if not exists"""
    if not os.path.exists(MODEL_PATH):
        print("Training model...")
        subprocess.run([sys.executable, "train_model.py"], check=True)

ensure_model_and_data()

# Load both datasets
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
    hashed_password TEXT NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS bids (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id INTEGER,
    user_id INTEGER NOT NULL,
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
conn.commit()

def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    sessions[token] = user_id
    return token

def get_current_user(request: Request):
    token = request.cookies.get("session_token")
    if not token or token not in sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return sessions[token]

@app.post("/login", response_class=HTMLResponse)
async def login_user(response: Response, email: str = Form(...), password: str = Form(...)):
    try:
        hashed = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute("SELECT id FROM users WHERE email = ? AND hashed_password = ?", (email, hashed))
        user = cursor.fetchone()
        if user:
            # Create session and set cookie
            session_token = create_session(user[0])
            resp = RedirectResponse(url="/contracts", status_code=303)
            resp.set_cookie(key="session_token", value=session_token, httponly=True, max_age=3600)
            return resp
        else:
            return "<h2 style='color:red;'>❌ Invalid credentials</h2><p><a href='/login'>Try again</a></p>"
    except Exception as e:
        return f"<h2 style='color:red;'>❌ Login error: {str(e)}</h2><p><a href='/login'>Try again</a></p>"

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

def adjust_for_inflation(base_price, inflation_rate=0.12, years=2):
    return base_price * ((1 + inflation_rate) ** years)

def create_access_token( dict):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"exp": expire, **data}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = None):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        cursor.execute("SELECT id, email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return {"id": user[0], "email": user[1]}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return RedirectResponse(url="/contracts")

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
async def register_user(email: str = Form(...), password: str = Form(...)):
    if not email or "@" not in email:
        return "<h2 style='color:red;'>❌ Invalid email format</h2><p><a href='/register'>Go back</a></p>"
    
    if len(password) < 6:
        return "<h2 style='color:red;'>❌ Password too short (min 6 characters)</h2><p><a href='/register'>Go back</a></p>"
    
    try:
        hashed = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute("INSERT INTO users (email, hashed_password) VALUES (?, ?)", (email, hashed))
        conn.commit()
        return "<h2 style='color:green;'>✅ Registration successful!</h2><p><a href='/login'>Login here</a></p>"
    except sqlite3.IntegrityError:
        return "<h2 style='color:red;'>❌ Email already registered</h2><p><a href='/login'>Login here</a></p>"
    except Exception as e:
        return f"<h2 style='color:red;'>❌ Error: {str(e)}</h2><p><a href='/register'>Try again</a></p>"

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login_user(email: str = Form(...), password: str = Form(...)):
    try:
        hashed = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute("SELECT id FROM users WHERE email = ? AND hashed_password = ?", (email, hashed))
        user = cursor.fetchone()
        if user:
            return f"<h2 style='color:green;'>✅ Login successful!</h2><p>User ID: {user[0]}</p><p><a href='/contracts'>View contracts</a></p>"
        else:
            return "<h2 style='color:red;'>❌ Invalid credentials</h2><p><a href='/login'>Try again</a> or <a href='/register'>register</a></p>"
    except Exception as e:
        return f"<h2 style='color:red;'>❌ Login error: {str(e)}</h2><p><a href='/login'>Try again</a></p>"


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("access_token")
    return response

@app.get("/contracts", response_class=HTMLResponse)
def contracts(request: Request):
    # Check if user is authenticated
    token = request.cookies.get("access_token")
    try:
        user = get_current_user(token)
        return templates.TemplateResponse("contracts_fragment.html", {
            "request": request, 
            "contracts": df_bidding.to_dict(orient="records"),
            "user_id": user["id"]
        })
    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)

@app.get("/contracts/{contract_id}", response_class=HTMLResponse)
def contract_detail(request: Request, contract_id: int):
    token = request.cookies.get("access_token")
    try:
        user = get_current_user(token)
        row = df_bidding.iloc[contract_id]
        return templates.TemplateResponse("contract_detail.html", {
            "request": request, 
            "contract": row.to_dict(),
            "user_id": user["id"]
        })
    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)

@app.post("/contracts/{contract_id}/submit_bid", response_class=HTMLResponse)
async def submit_bid(
    request: Request,
    contract_id: int,
    email: str = Form(...),
    phone: str = Form(...),
    bid_amount: float = Form(...),
    equipment_list: str = Form(...),
    workforce: str = Form(...),
):
    token = request.cookies.get("access_token")
    try:
        user = get_current_user(token)
        user_id = user["id"]
        
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
        INSERT INTO bids (contract_id, user_id, email, phone, bid_amount, equipment_list, workforce, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (contract_id, user_id, email, phone, bid_amount, equipment_list, workforce, status_msg))
        conn.commit()

        return f"<h2>Bid Result</h2><p>Status: {status_msg}</p><br><a href='/contracts' class='btn btn-primary'>Back to Contracts</a>"
        
    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)






