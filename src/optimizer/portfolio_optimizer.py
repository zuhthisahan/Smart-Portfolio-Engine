import pandas as pd
import os
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus
from config.spark_session import init_spark_session
import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment("Smart_Portfolio_Optimization")

# 1. Define the Assortment (The Financial Assets)
ASSETS = {
    'Savings': {'return': 0.02, 'risk': 1},
    'Bonds':   {'return': 0.04, 'risk': 3},
    'ETFs':    {'return': 0.08, 'risk': 6},
    'Crypto':  {'return': 0.15, 'risk': 10}
}

def optimize_portfolio(budget: float, risk_score: float, horizon: int) -> dict:
    """
    Core Optimization Engine: Allocates a budget across assets to maximize return,
    strictly bounded by the user's risk tolerance and time horizon constraints.
    """
    # If the user has no money to invest, return 0s immediately
    if budget <= 0:
        return {asset: 0.0 for asset in ASSETS}

    # Initialize the Linear Programming Problem
    prob = LpProblem("Smart_Portfolio_Optimization", LpMaximize)

    # Define the Decision Variables (Continuous numbers, Minimum $0)
    allocations = {
        asset: LpVariable(f"Alloc_{asset}", lowBound=0, cat='Continuous') 
        for asset in ASSETS
    }

    # --- OBJECTIVE FUNCTION ---
    # Maximize total expected return
    prob += lpSum([allocations[a] * ASSETS[a]['return'] for a in ASSETS]), "Total_Expected_Return"

    # --- CONSTRAINTS ---
    # 1. Budget Constraint: Total allocated must equal the disposable income
    prob += lpSum([allocations[a] for a in ASSETS]) == budget, "Budget_Constraint"

    # 2. Risk Constraint: The weighted average risk of the portfolio must be <= user's risk score
    # (Sum of (Allocation * Asset Risk)) <= (Budget * User Risk Score)
    prob += lpSum([allocations[a] * ASSETS[a]['risk'] for a in ASSETS]) <= (budget * risk_score), "Risk_Constraint"

    # 3. Time Horizon Guardrail: If horizon is < 5 years, NO Crypto allowed
    if horizon < 5:
        prob += allocations['Crypto'] == 0, "Horizon_Crypto_Block"

    # Solve the mathematical equation
    prob.solve()

    # Extract the results safely
    if LpStatus[prob.status] == 'Optimal':
        return {asset: round(allocations[asset].varValue, 2) for asset in ASSETS}
    else:
        return {"Error": "Optimization Failed to find a feasible solution"}

def run_optimization_engine():
    """Reads the Gold table, runs the optimizer for a sample of users, and displays the results."""
    print("Booting up the Smart Portfolio Decision Engine...")
    
    # Initialize Spark to read the Delta table
    spark = init_spark_session()

    # Read the Gold Features we engineered previously
    # gold_path = "../../data/gold/user_portfolio_features"
    GOLD_PATH = "/opt/airflow/data/gold/user_portfolio_features"
    gold_df = spark.read.format("delta").load(GOLD_PATH)
    
    # Convert a sample of 5 users to Pandas for the PuLP engine
    users_pd = gold_df.limit(5).toPandas()

    for index, row in users_pd.iterrows():
        # Start mlflow for each user optimization run
        with mlflow.start_run(run_name = f"User_{row['client_id']}"):
            budget = row['monthly_disposable_income']
            risk = row['risk_tolerance_score']
            horizon = row['investment_horizon']

            # Log the input params
            mlflow.log_param('budget', budget)
            mlflow.log_param("risk_score", risk)
            mlflow.log_param("horizon", horizon)
            
            # Runt the optimizer
            reccomendation = optimize_portfolio(budget, risk, horizon)
            
            # Log the output metrics
            if "Error" not in reccomendation:
                # Calculate expected annual return and risk for the recommended portfolio
                total_return =  sum(reccomendation[a]  * ASSETS[a]['return'] for a in ASSETS)
                total_risk = sum(reccomendation[a] * ASSETS[a]['risk'] for a in ASSETS)

                # Log the calculated metrics
                mlflow.log_metric("expected_annual_return", round(total_return), 2)
                mlflow.log_metric("expected_annual_risk", round(total_risk), 2)

                # Log individual asset allocations as metrics
                
                # Log the specific allocations as individual parameters
                for asset, amt in reccomendation.items():
                    mlflow.log_param(f"alloc_{asset}", amt)
                
                print(f"Optimized User {row['client_id']} - logged to MLflow")
                
    spark.stop()
    
if __name__ == "__main__":
    run_optimization_engine()