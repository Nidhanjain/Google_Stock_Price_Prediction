import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from google import genai
from predict import predict_today

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Google Stock Price Prediction System",
    layout="centered"
)

# --- NEW: INITIALIZE GOOGLE GENAI CLIENT ---
# This pulls your API key from Streamlit's Secret Manager
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

st.title("📈 Stock Prediction System (Next Day)")
st.markdown(
    """
    This system is based on stock prices from 2016 to 2021 and predicts:
    - **Direction (BUY / SELL / NO TRADE)**
    - **Probability of Move**
    - **Expected % move**
    - **Predicted next-day price**
    - **Confidence price range**
    
    ⚠️ All inputs are **DECIMAL values** (example: `0.01 = 1%`)
    """
)

st.markdown("---")

# ----------------------------
# Inputs (ENGINEERED FEATURES)
# ----------------------------
st.subheader("Enter Today's Engineered Indicators")

price_sma5 = st.number_input(
    "Price vs SMA-5 (decimal, 0.01 = +1%)",
    value=0.01
)

price_sma20 = st.number_input(
    "Price vs SMA-20 (decimal, 0.02 = +2%)",
    value=0.02
)

trend = st.selectbox(
    "Trend (SMA-5 > SMA-20)",
    options=[0, 1],
    format_func=lambda x: "Uptrend" if x == 1 else "Downtrend"
)

ret_5 = st.number_input(
    "5-day Return (decimal)",
    value=0.01
)

ret_10 = st.number_input(
    "10-day Return (decimal)",
    value=0.02
)

vol_5 = st.number_input(
    "5-day Volatility (std of returns)",
    value=0.015
)

vol_change = st.number_input(
    "Volume Change (decimal)",
    value=0.05
)

hl_range = st.number_input(
    "High-Low Range ((H−L)/Close)",
    value=0.01
)

oc_range = st.number_input(
    "Open-Close Range ((C−O)/O)",
    value=0.005
)

today_close = st.number_input(
    "Today's Close Price",
    value=100.0,
    min_value=0.01
)

# ----------------------------
# Build feature dict
# ----------------------------
today_features = {
    "price_sma5": price_sma5,
    "price_sma20": price_sma20,
    "trend": trend,
    "ret_5": ret_5,
    "ret_10": ret_10,
    "vol_5": vol_5,
    "vol_change": vol_change,
    "hl_range": hl_range,
    "oc_range": oc_range
}

st.markdown("---")

# ----------------------------
# Prediction Logic
# ----------------------------
# Create 3 columns for quick stats
col1, col2, col3 = st.columns(3)
col1.metric("Ticker", "GOOGL")
col2.metric("Current Price", f"${today_close}")
col3.metric("Training Era", "2016-2021")
if st.button("🔮 Predict"):
    result = predict_today(today_features, today_close)
    if result is None:
        st.error("Prediction failed. Please check your inputs.")
        st.stop()

    # Display the results with color coding
    if result["decision"] == "BUY":
        st.success(f"📈 BUY SIGNAL (Prob: {result['prob_up']:.2f})")
        st.write(f"**Predicted Target Price:** ${result['predicted_price']:.2f}")
        st.write(f"**Confidence Range:** ${result['lower_price']:.2f} - ${result['upper_price']:.2f}")
        
    elif result["decision"] == "SELL":
        # For SELL, we show the probability of the price going DOWN (1 - prob_up)
        st.error(f"📉 SELL SIGNAL (Prob: {1 - result['prob_up']:.2f})")
        st.write(f"**Predicted Target Price:** ${result['predicted_price']:.2f}")
        st.write(f"**Confidence Range:** ${result['lower_price']:.2f} - ${result['upper_price']:.2f}")
        
    else:
        st.warning(f"⚖️ NO TRADE (Neutral Zone - Prob: {result['prob_up']:.2f})")
        st.info("The model is not confident enough in either direction to issue a signal.")

    st.markdown("---")
    
    # ----------------------------
    # Visualization Section
    # ----------------------------
    st.subheader("Price Trend Visualization")
    
    # Create a simulated recent history leading up to today_close
    history_size = 20
    noise = np.random.normal(0, 1, history_size)
    history = np.linspace(today_close * 0.95, today_close, history_size) + noise
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(history_size), history, label="Recent History", color="#1f77b4", linewidth=2)
    
    # Plot Prediction
    prediction_index = history_size
    ax.scatter(prediction_index, result['predicted_price'], color='orange', s=100, label="Predicted Price", zorder=5)
    
    # Add Confidence Interval Bar
    ax.vlines(prediction_index, result['lower_price'], result['upper_price'], color='orange', alpha=0.3, linewidth=5, label="Confidence Band")
    
    # Formatting
    ax.set_title(f"Next Day Forecast for ${today_close}", color="white")
    ax.set_ylabel("Price ($)", color="white")
    ax.set_facecolor("#262730")
    fig.patch.set_facecolor("#262730")
    ax.tick_params(colors='white')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)

    # --- NEW: GOOGLE GEMINI 2.0 FLASH INTEGRATION ---
    # --- HIGHLIGHT: GEMINI 2.0 FLASH AI AGENT INTEGRATION ---
    st.markdown("---")
    st.header("🤖 Google Gemini: Explainable AI (XAI) Analyst")
    
    with st.spinner("Agentic Reasoning in progress..."):
        # We give Gemini a specific 'Persona' and 'Task' to make it a highlight
        system_instruction = "You are a Senior Quantitative Analyst at Google Cloud Finance. Your job is to explain ML model outputs to retail investors using clear, professional, and actionable language."
        
        prompt = f"""
        --- ML MODEL OUTPUT ---
        Prediction: {result['decision']}
        Confidence: {result['prob_up']:.2f}
        Predicted Target: ${result['predicted_price']:.2f}
        
        --- TECHNICAL DATA ---
        - Price vs 20-day Moving Average: {price_sma20*100}%
        - Volume Change: {vol_change*100}%
        - Volatility: {vol_5*100}%
        - Trend Direction: {'Uptrend' if trend == 1 else 'Downtrend'}
        
        --- ANALYSIS REQUEST ---
        1. Explain WHY the model chose a '{result['decision']}' signal based on these specific technical indicators.
        2. Identify the #1 Risk Factor for this trade today.
        3. Suggest a 'What-If' scenario: If Volume Change spikes to 20%, how would it change the sentiment?
        """
        
        # Using the newest Client-side System Instructions (Highlight of Gemini 2.0)
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt,
            config={
                "system_instruction": system_instruction,
                "temperature": 0.7  # Balanced between creative and factual
            }
        )
        
        # Displaying the response in a more 'Professional' container
        with st.expander("🔍 Deep Dive: How the AI made this decision", expanded=True):
            st.markdown(response.text)
            st.info("💡 **GDG Presentation Tip:** This section uses 'Explainable AI' (XAI) to solve the black-box problem of traditional Machine Learning.")

st.markdown("---")
st.caption("Model uses time-series trained ML + Google Gemini 2.0 for Explainable AI.")

