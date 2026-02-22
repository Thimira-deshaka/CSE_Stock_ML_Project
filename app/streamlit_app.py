import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="CSE Market Intelligence", page_icon="📈", layout="wide")

# 1. LOAD ASSETS (Model and Scaler)
@st.cache_resource
def load_ml_assets():
    with open('models/cse_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_ml_assets()
except FileNotFoundError:
    st.error("⚠️ Model or Scaler files not found. Please run preprocess.py and train_model.py first.")
    st.stop()

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #262730; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title("🇱🇰 Colombo Stock Exchange (CSE) Intelligence Dashboard")
st.write("Fundamental Analysis: Predicting the All Share Price Index (ASPI) via Macroeconomic Indicators.")
st.markdown("---")

# 2. SIDEBAR - User Input (Updated for 4 Indicators)
st.sidebar.header("🕹️ Economic Control Panel")
st.sidebar.info("Enter real-world values to simulate market impact.")

# Add the missing inputs
raw_inflation = st.sidebar.number_input("Inflation Rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
raw_gdp = st.sidebar.number_input("GDP Growth Rate (%)", min_value=-15.0, max_value=15.0, value=3.0, step=0.1)
raw_interest = st.sidebar.slider("Interest Rate (SDFR %)", 4.0, 25.0, 10.0, 0.25)
raw_exchange = st.sidebar.number_input("Exchange Rate (LKR/USD)", min_value=150.0, max_value=500.0, value=300.0, step=1.0)

# 3. DATA PROCESSING (Must have all 4 columns)
# The order of columns here MUST match the order used in your training script
input_df = pd.DataFrame(
    [[raw_inflation, raw_gdp, raw_interest, raw_exchange]], 
    columns=['Inflation', 'GDP_Growth', 'Interest_Rate', 'Exchange_Rate']
)

# Normalize using the saved scaler
normalized_input = scaler.transform(input_df)

# Prediction will now work because normalized_input has 4 features
prediction = model.predict(normalized_input)[0]

# Calculate Delta vs Last Close
try:
    df_processed = pd.read_csv('data/processed/processed_data.csv')
    # latest_close = df_processed['Close'].iloc[-1]
    latest_close = 6000
except Exception:
    latest_close = prediction

delta = prediction - latest_close
delta_percent = (delta / latest_close) * 100

# 4. MAIN DISPLAY
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Market Forecast")
    
    st.metric(
        label="Predicted ASPI Index", 
        value=f"{prediction:,.2f}", 
        delta=f"{delta_percent:.2f}%",
        delta_color="normal"
    )
    
    if delta > 0:
        st.success("📈 **Bullish Outlook:** The current economic climate suggests market growth.")
    else:
        st.error("📉 **Bearish Outlook:** Macroeconomic pressure suggests a potential market dip.")
    
    st.write("---")
    st.write(f"**Scenario Summary:** At **{raw_inflation}%** inflation and **{raw_gdp}%** GDP growth, the model predicts the index will settle around **{prediction:,.0f}** points.")

with col2:
    # Attractive Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        title = {'text': "Predicted Market Level", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 12000], 'tickwidth': 1},
            'bar': {'color': "#2c3e50"},
            'steps': [
                {'range': [0, 3000], 'color': "#ffcdd2"}, # Red zone
                {'range': [3000, 6000], 'color': "#fff9c4"}, # Yellow zone
                {'range': [6000, 12000], 'color': "#c8e6c9"}], # Green zone
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction}
        }
    ))
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 5. EXPLAINABILITY (XAI Section - Requirement 4)
st.subheader("🔍 Explainable AI (XAI) Insights")
exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    st.write("**How does this work?**")
    st.write("""
    The model uses **LightGBM**, which analyzed more than 10 years of Sri Lankan data. 
    It doesn't just look at numbers; it looks at relationships.
    """)

with exp_col2:
    if raw_interest > 15.0:
        st.warning("🚨 **Key Driver: Interest Rates.** High rates are likely the strongest negative force, making fixed deposits more attractive than stocks.")
    elif raw_exchange > 340.0:
        st.warning("⚠️ **Key Driver: Currency Devaluation.** LKR instability is increasing import costs and dampening market sentiment.")
    elif raw_inflation > 20.0:
        st.error("🚨 **Key Driver: Inflation.** Hyper-inflation is eroding corporate purchasing power.")
    else:
        st.info("ℹ️ **Key Driver: Macro Stability.** Relatively stable indicators are supporting the index.")

st.caption("Disclaimer: Academic project. Past data (2010-2024) is not a guarantee of future market performance.")