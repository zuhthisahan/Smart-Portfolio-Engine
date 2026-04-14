import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_BASE_URL = "http://localhost:8000"
NEW_CUSTOM_USER = "New Custom User"
st.set_page_config(page_title="Smart Portfolio Engine", layout="wide")


st.title("Smart Portfolio Decision Engine")
st.markdown("AI-powered expense forecasting and mathematical portfolio optimization.")


@st.cache_data(ttl=60)
def get_users():
    try:
        response = requests.get(f"{API_BASE_URL}/users")
        if response.status_code == 200:
            return response.json().get("users", [])
    except Exception:
        st.error("Cannot connect to backend API. Is FastAPI running?")
    return []


@st.cache_data(ttl=60)
def get_user_history(client_id):
    try:
        response = requests.get(f"{API_BASE_URL}/users/{client_id}/history")
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


user_list = get_users()

# --- 2. Sidebar / User Selection ---
with st.sidebar:
    st.header("User Selection")
    if user_list:
        selected_client = st.selectbox("Select a Client Profile", [NEW_CUSTOM_USER] + user_list)
    else:
        selected_client = NEW_CUSTOM_USER
        st.warning("No database users found. Operating in Guest Mode.")


# --- 3. Data Hydration Logic ---
if selected_client != NEW_CUSTOM_USER:
    try:
        user_data = requests.get(f"{API_BASE_URL}/users/{selected_client}").json()
        st.info(f"Loaded historical data for **{selected_client}**")
    except Exception:
        user_data = {}
        st.warning("Failed to fetch user data.")
else:
    user_data = {
        "monthly_income": 5000.0,
        "risk_score": 5,
        "investment_horizon": 15,
        "current_month_spend": 2000.0,
        "spend_1_month_ago": 2100.0,
        "spend_2_months_ago": 1900.0,
    }
    st.info("Guest Mode: Enter your financial details below to generate a custom plan.")


# --- 4. The Main UI Form (Always Visible) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Profile")
    income = st.number_input("Monthly Income ($)", value=float(user_data.get("monthly_income", 5000)))
    risk = st.slider("Risk Tolerance (1-10)", min_value=1, max_value=10, value=int(user_data.get("risk_score", 5)))
    horizon = st.number_input("Investment Horizon (Years)", value=int(user_data.get("investment_horizon", 10)))

with col2:
    st.subheader("Recent Spend History")
    spend_1 = st.number_input("Spend 1 Month Ago ($)", value=float(user_data.get("current_month_spend", 0)))
    spend_2 = st.number_input("Spend 2 Months Ago ($)", value=float(user_data.get("spend_1_month_ago", 0)))
    spend_3 = st.number_input("Spend 3 Months Ago ($)", value=float(user_data.get("spend_2_months_ago", 0)))


# --- 5. The Action Button ---
st.divider()
if st.button("Generate Smart Portfolio Plan", use_container_width=True):
    with st.spinner("AI is forecasting budget and optimizing assortment..."):
        payload_client_id = "Guest" if selected_client == NEW_CUSTOM_USER else selected_client

        payload = {
            "client_id": payload_client_id,
            "monthly_income": income,
            "risk_score": risk,
            "investment_horizon": horizon,
            "current_month_spend": spend_1,
            "spend_1_month_ago": spend_2,
            "spend_2_month_ago": spend_3,
        }

        response = requests.post(f"{API_BASE_URL}/smart-advisor", json=payload)

        if response.status_code == 200:
            result = response.json()
            forecast = result["financial_forecast"]
            allocations = result["recommended_action"]

            st.success("Plan Generated Successfully!")

            m1, m2, m3 = st.columns(3)
            m1.metric("Monthly Income", f"${forecast['monthly_income']:,.2f}")
            m2.metric("Predicted Expense (AI)", f"${forecast['predicted_expense']:,.2f}", delta="- Expense", delta_color="inverse")
            m3.metric("Investable Budget", f"${forecast['safe_investable_budget']:,.2f}", delta="Safe to Invest")

            st.subheader("Optimal Asset Allocation (PuLP Optimizer)")

            if "Error" not in allocations and "Message" not in allocations:
                df_chart = pd.DataFrame({
                    "Asset": list(allocations.keys()),
                    "Amount ($)": list(allocations.values()),
                })
                df_chart = df_chart[df_chart["Amount ($)"] > 0]

                custom_colors = ["#10B981", "#3B82F6", "#8B5CF6", "#06B6D4"]
                fig = px.pie(df_chart, values="Amount ($)", names="Asset", hole=0.4, color_discrete_sequence=custom_colors)
                st.plotly_chart(fig, use_container_width=True)

                if payload_client_id != "Guest":
                    monthly_history = get_user_history(selected_client)
                    
                    if monthly_history and len(monthly_history) > 0:
                        history_df = pd.DataFrame(monthly_history)
                        
                        # Safely convert to datetime without aggressive dropping
                        history_df["txn_month"] = pd.to_datetime(history_df["txn_month"], format='mixed', errors="ignore")
                        history_df = history_df.sort_values("txn_month")
                        
                        # Calculate the 3-Month Rolling Average
                        # Since we already filtered for a specific user, we don't strictly need groupby,
                        # but it's safe to leave it in case multiple clients are in the dataframe.
                        if "client_id" in history_df.columns:
                            history_df["smoothed_target_spend"] = (
                                history_df
                                .groupby("client_id")["current_month_spend"]
                                .transform(lambda x: x.rolling(window=3, min_periods=1).mean().round(2))
                            )
                        else:
                            history_df["smoothed_target_spend"] = history_df["current_month_spend"].rolling(window=3, min_periods=1).mean().round(2)

                        # Build the Plotly figure with ONLY the smoothed line
                        spend_history_fig = go.Figure()
                        
                        spend_history_fig.add_trace(go.Scatter(
                            x=history_df["txn_month"],
                            y=history_df["smoothed_target_spend"],
                            mode="lines+markers",
                            name="Smoothed Target Spend",
                            line=dict(color="#EF4444", width=3),
                        ))
                        
                        spend_history_fig.update_layout(
                            title=f"Smoothed Spend Trend (3-Month Avg) for {selected_client}",
                            xaxis_title="Month",
                            yaxis_title="Amount Spent ($)",
                            hovermode="x unified",
                        )
                        
                        st.subheader("Historical Spend Trend")
                        st.plotly_chart(spend_history_fig, use_container_width=True)
                    else:
                        st.info(f"No historical spend series was returned for {selected_client}. Make sure your backend has a `/users/{{client_id}}/history` endpoint!")
            else:
                st.warning(allocations.get("Message", allocations.get("Error")))
        else:
            st.error(f"API Error: {response.text}")
            
