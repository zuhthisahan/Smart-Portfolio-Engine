from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBRegressor
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus
import pandas as pd
import os
from deltalake import DeltaTable



# 1. Initialize the API
app = FastAPI(title="Smart Portfolio Decision Engine")

# 2. Load the Machine Learning Model (Happens once when the server starts)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR /"src" / "models" / "time_series" / "xgb_spend_forecaster_v1.json"

GOLD_TABLE_PATH = BASE_DIR / "data" / "gold" / "time_series_features"
GOLD_PROFILE_TABLE_PATH = BASE_DIR / "data" / "gold" / "user_portfolio_features"

forecaster = XGBRegressor()

# We use a try-except block so the API doesn't crash if the file is missing locally
try:
    forecaster.load_model(MODEL_PATH)
    print("✅ ML Model loaded successfully into memory.")
except Exception as e:
    print(f"⚠️ Warning: Could not load model at {MODEL_PATH}. Error: {e}")

# 3. Define the Data Schemas (What the Frontend will send us)
class UserFinancialProfile(BaseModel):
    client_id: str
    monthly_income: float
    risk_score: int
    investment_horizon: int
    # The Time-Series Lags (Can be fetched from DB for existing users, or typed in by new users)
    current_month_spend: float
    spend_1_month_ago: float
    spend_2_month_ago: float

# 4. The PuLP Optimizer Function
ASSETS = {
    'Savings': {'return': 0.02, 'risk': 1},
    'Bonds':   {'return': 0.04, 'risk': 3},
    'ETFs':    {'return': 0.08, 'risk': 6},
    'Crypto':  {'return': 0.15, 'risk': 10}
}

def optimize_portfolio(budget: float, risk_score: float, horizon: int) -> dict:
    if budget <= 0:
        return {"Message": "No disposable income available for investment."}

    prob = LpProblem("Smart_Portfolio", LpMaximize)
    allocations = {asset: LpVariable(f"Alloc_{asset}", lowBound=0) for asset in ASSETS}
    
    # Objective & Constraints
    prob += lpSum([allocations[a] * ASSETS[a]['return'] for a in ASSETS])
    prob += lpSum([allocations[a] for a in ASSETS]) == budget
    prob += lpSum([allocations[a] * ASSETS[a]['risk'] for a in ASSETS]) <= (budget * risk_score)
    
    if horizon < 5:
        prob += allocations['Crypto'] == 0

    prob.solve()
    if LpStatus[prob.status] == 'Optimal':
        return {asset: round(allocations[asset].varValue, 2) for asset in ASSETS}
    return {"Error": "Optimization Failed"}

try:
    # Read the Delta table directly into a Pandas DataFrame
    dt = DeltaTable(GOLD_TABLE_PATH)
    # Sort by time so the most recent month is at the bottom
    db_df = dt.to_pandas().sort_values(by=['client_id', 'txn_month'])
    print(f" User Database loaded. Found {db_df['client_id'].nunique()} unique users.")  
except Exception as e:
    print(f" Warning: Could not load Gold Table at {GOLD_TABLE_PATH}. Error: {e}")
    # Fallback empty dataframe so the API doesn't crash
    db_df = pd.DataFrame(columns=['client_id', 'txn_month', 'current_month_spend', 'spend_1_month_ago', 'spend_2_months_ago'])

try:
    profile_dt = DeltaTable(GOLD_PROFILE_TABLE_PATH)
    profile_df = profile_dt.to_pandas()
    print(f" User Profile Database loaded. Found {len(profile_df)} profile rows.")
except Exception as e:
    print(f" Warning: Could not load Gold Profile Table at {GOLD_PROFILE_TABLE_PATH}. Error: {e}")
    profile_df = pd.DataFrame(columns=['client_id', 'id', 'monthly_income', 'risk_tolerance_score', 'investment_horizon'])



# 5. The Magic Endpoint (Combines ML + Optimizer)
@app.post("/smart-advisor")
async def generate_smart_plan(profile: UserFinancialProfile):
    """
    Takes user history, predicts next month's spend, calculates budget, and optimizes portfolio.
    """
    try:
        # Step 2: Format data for XGBoost
        ml_features = pd.DataFrame([{
            'current_month_spend': profile.current_month_spend,
            'spend_1_month_ago': profile.spend_1_month_ago,
            'spend_2_months_ago': profile.spend_2_month_ago,
        }])

        # Step 3: AI Predicts Next Month's Expense
        predicted_expense = float(forecaster.predict(ml_features)[0])
        
        # Step 4: Calculate actual disposable budget
        calculated_budget = profile.monthly_income - predicted_expense

        # Step 5: Run the Optimizer
        portfolio_recommendation = optimize_portfolio(
            budget=calculated_budget, 
            risk_score=profile.risk_score, 
            horizon=profile.investment_horizon
        )

        # Step 6: Return the complete package to the Frontend
        return {
            "client_id": profile.client_id,
            "financial_forecast": {
                "monthly_income": profile.monthly_income,
                "predicted_expense": round(predicted_expense, 2),
                "safe_investable_budget": round(calculated_budget, 2)
            },
            "recommended_action": portfolio_recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 6. Health Check Endpoint
@app.get("/")
def health_check():
    return {"status": "API is live and ready."}

@app.get("/users")
async def get_all_users():
    """Returns a list of all unique client IDs."""
    try:
        unique_users = db_df['client_id'].unique().tolist()
        return {"users": unique_users}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Database not initialized.")

@app.get("/users/{client_id}")
async def get_user_profile(client_id: str):
    """Fetches the most recent financial snapshot for a specific user to auto-fill the UI."""
    user_history = db_df[db_df['client_id'].astype(str) == str(client_id)]
    
    if user_history.empty:
        raise HTTPException(status_code=404, detail="User not found.")

    profile_key = 'client_id' if 'client_id' in profile_df.columns else 'id' if 'id' in profile_df.columns else None
    if profile_key is None:
        raise HTTPException(status_code=500, detail="Profile data is missing a user identifier column.")

    user_profile = profile_df[profile_df[profile_key].astype(str) == client_id]
    if user_profile.empty:
        raise HTTPException(status_code=404, detail="User profile not found.")
    
    # Grab the very last row (the most recent month)
    latest_data = user_history.iloc[-1]
    latest_profile = user_profile.iloc[-1]
    
    return {
        "client_id": client_id,
        "monthly_income": float(latest_profile['monthly_income']),
        "risk_score": int(latest_profile['risk_tolerance_score']),
        "investment_horizon": int(latest_profile['investment_horizon']),
        "current_month_spend": float(latest_data['current_month_spend']),
        "spend_1_month_ago": float(latest_data['spend_1_month_ago']),
        "spend_2_months_ago": float(latest_data['spend_2_months_ago']),
        
    }

@app.get("/users/{client_id}/history")
async def get_user_history(client_id: str):
    """Returns the raw monthly spend rows for a historical user."""
    user_history = db_df[db_df['client_id'].astype(str) == str(client_id)].copy()

    if user_history.empty:
        raise HTTPException(status_code=404, detail="User not found.")

    user_history['txn_month'] = pd.to_datetime(user_history['txn_month'], errors='coerce')
    user_history = user_history.dropna(subset=['txn_month']).sort_values('txn_month')

    if user_history.empty:
        raise HTTPException(status_code=404, detail="User history is missing valid transaction months.")

    return [
        {
            "client_id": str(row['client_id']),
            "txn_month": row['txn_month'].strftime("%Y-%m"),
            "current_month_spend": float(row['current_month_spend']),
            "spend_1_month_ago": None if pd.isna(row.get('spend_1_month_ago')) else float(row['spend_1_month_ago']),
            "spend_2_months_ago": None if pd.isna(row.get('spend_2_months_ago')) else float(row['spend_2_months_ago']),
            "spend_3_months_ago": None if pd.isna(row.get('spend_3_months_ago')) else float(row['spend_3_months_ago']) if 'spend_3_months_ago' in user_history.columns else None,
            "avg_spend_last_3_months": None if pd.isna(row.get('avg_spend_last_3_months')) else float(row['avg_spend_last_3_months']) if 'avg_spend_last_3_months' in user_history.columns else None,
        }
        for _, row in user_history.iterrows()
    ]

@app.get("/users/{client_id}/history")
async def get_user_history(client_id: str):
    """Fetches the entire time-series history for a user to plot on the frontend."""
    user_history = db_df[db_df['client_id'] == client_id]
    
    if user_history.empty:
        raise HTTPException(status_code=404, detail="User not found.")
    
    # Return the necessary columns for the chart
    chart_data = user_history[['client_id', 'txn_month', 'current_month_spend']].to_dict(orient='records')
    return chart_data