import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from google import genai
from google.genai import errors
from predict import predict_today

# ----------------------------
# 1. PAGE CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="Google Stock Price Prediction System | XAI",
    page_icon="📈",
    layout="centered"
)

# ----------------------------
# 2. GLOBAL STYLING & DISCLAIMER
# ----------------------------
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #4CAF50; color: white; }
    </style>
    """, unsafe_allow_html=True)

# GDG Presentation Instruction Disclaimer
st.info("""
**🚀 GDG System Architecture & Instructions:**
1. **ML Layer:** Traditional Scikit-Learn model predicts price direction based on 2016-2021 historical patterns.
2. **XAI Layer:** Google Gemini 2.0 Flash interprets the "Black Box" metrics into human-readable logic.
3. **Resilience:** If the API is busy, the system uses **Exponential Backoff** (Wait 2s, 4s, 8s) and **Dual-Key Failover**.
4. **Optimization:** Results are cached to minimize API costs and maximize speed.
""")

# ----------------------------
# 3. GEMINI CLIENT INITIALIZATION
# ----------------------------
def get_client(use_backup=False):
    """Initializes client with Primary or Backup key from st.secrets."""
    try:
        if use_backup:
            key = st.secrets.get("GEMINI_API_KEY_BACKUP")
            if not key: # Fallback to primary if backup not set
                key = st.secrets["GEMINI_API_KEY"]
        else:
            key = st.secrets["GEMINI_API_KEY"]
        return genai.Client(api_key=key)
    except Exception:
        st.error("API Key not found in Secrets. Please check your Dashboard.")
        return None

# Global client initialization
client = get_client()

# ----------------------------
# 4. ROBUST AI ANALYST (DUAL-KEY + BACKOFF + JITTER)
# ----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_xai_analysis(decision, prob, sma20, vol, vol_chg, trend_val):
    """
    Expert AI Analyst with Failover and Jittered Exponential Backoff.
    """
    global client
    system_instr = "You are a Senior Quantitative Analyst at Google Finance."
    prompt = f"""
    ML Prediction for GOOGL: {decision} ({prob:.2%}).
    Technical Context: SMA20 {sma20*100}%, Volatility {vol*100}%, Volume Change {vol_chg*100}%.
    Trend: {'Upward' if trend_val == 1 else 'Downward'}.
    Explain why these technicals led to this specific {decision} signal.
    """

    max_retries = 3
    retry_count = 0
    wait_time = 2  

    while retry_count < max_retries:
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=prompt,
                config={"system_instruction": system_instr, "temperature": 0.7}
            )
            return response.text
        except errors.ClientError as e:
            if "429" in str(e):
                retry_count += 1
                
                # --- AUTO-SWITCH KEY LOGIC ---
                if retry_count == 1:
                    st.toast("Primary API busy, switching to backup...", icon="🔄")
                    client = get_client(use_backup=True)
                
                # --- JITTERED BACKOFF ---
                jitter = random.uniform(1.0, 3.0)
                sleep_duration = wait_time + jitter
                st.warning(f"Rate limit hit. Retrying in {sleep_duration:.1f}s...")
                time.sleep(sleep_duration)
                
                wait_time *= 2 # Exponentially increase wait
                
                if retry_count == max_retries:
                    return "⚠️ AI Analyst is currently busy. Please wait 60 seconds."
            elif "404" in str(e):
                return "❌ Model Version Error: Ensure SDK supports gemini-2.0-flash."
            else:
                return f"❌ API Error: {str(e)}"
    
    return "AI Analyst timed out."

# ----------------------------
# 5. UI HEADER & DESCRIPTION
# ----------------------------
st.title("📈 Google Stock Prediction System")
st.markdown("### Agentic Machine Learning & Explainable AI")

with st.expander("📖 Project Methodology"):
    st.write("""
        This system utilizes a **Scikit-Learn** time-series model trained on historical data.
        The prediction is then fed into **Google Gemini 2.0 Flash** to perform 'Explainable AI' (XAI), 
        bridging the gap between raw data and human reasoning.
    """)

st.markdown("---")

# ----------------------------
# 6. INPUT PARAMETERS
# ----------------------------
st.subheader("🛠️ Technical Indicators (Today's Values)")
l_col, r_col = st.columns(2)

with l_col:
    price_sma5 = st.number_input("Price vs SMA-5", value=0.01)
    price_sma20 = st.number_input("Price vs SMA-20", value=0.02)
    trend = st.selectbox("Market Trend", [0, 1], format_func=lambda x: "Uptrend" if x == 1 else "Downtrend")
    ret_5 = st.number_input("5-Day Return", value=0.01)
    ret_10 = st.number_input("10-Day Return", value=0.02)

with r_col:
    vol_5 = st.number_input("5-Day Volatility", value=0.015, format="%.3f")
    vol_change = st.number_input("Volume Change", value=0.05)
    hl_range = st.number_input("High-Low Range", value=0.01)
    oc_range = st.number_input("Open-Close Range", value=0.005)
    today_close = st.number_input("Today's Close Price ($)", value=150.00)

today_features = {
    "price_sma5": price_sma5, "price_sma20": price_sma20, "trend": trend,
    "ret_5": ret_5, "ret_10": ret_10, "vol_5": vol_5,
    "vol_change": vol_change, "hl_range": hl_range, "oc_range": oc_range
}

# ----------------------------
# 7. PREDICTION AND VISUALIZATION
# ----------------------------
st.markdown("---")
if st.button("🔮 Run Predictive Engine"):
    result = predict_today(today_features, today_close)
    
    if result:
        # Display Decision Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Ticker", "GOOGL")
        with c2:
            st.metric("Signal", result['decision'])
        with c3:
            st.metric("Confidence", f"{result['prob_up']:.1%}")

        if result["decision"] == "BUY":
            st.success(f"📈 **BUY SIGNAL ISSUED** at ${today_close}")
        elif result["decision"] == "SELL":
            st.error(f"📉 **SELL SIGNAL ISSUED** at ${today_close}")
        else:
            st.warning("⚖️ **NEUTRAL - NO TRADE RECOMMENDED**")

        st.write(f"**Predicted Target:** `${result['predicted_price']:.2f}` (Range: `${result['lower_price']:.2f}` - `${result['upper_price']:.2f}`)")

        # --- Charting ---
        st.subheader("📊 Price Action Forecast")
        h_size = 20
        h_data = np.linspace(today_close*0.97, today_close, h_size) + np.random.normal(0, 0.4, h_size)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(h_size), h_data, color="#3b82f6", label="Historical")
        ax.scatter(h_size, result['predicted_price'], color="#fbbf24", s=100, label="Target")
        ax.vlines(h_size, result['lower_price'], result['upper_price'], color="#fbbf24", alpha=0.3, linewidth=6)
        
        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#111827")
        ax.tick_params(colors='white')
        ax.legend()
        st.pyplot(fig)

        # ----------------------------
        # 8. THE GEMINI AI REPORT (XAI)
        # ----------------------------
        st.markdown("---")
        st.subheader("🤖 Gemini Explainable AI (XAI) Analysis")
        
        with st.spinner("Wait... Gemini is calculating with backoff..."):
            report = get_xai_analysis(
                result['decision'], 
                result['prob_up'], 
                price_sma20, 
                vol_5, 
                vol_change, 
                trend
            )
            
            with st.chat_message("assistant"):
                st.markdown(report)
            
            st.caption("AI Insight powered by Gemini 2.0 Flash. Rate-limit logic: Jittered Backoff Enabled.")

# ----------------------------
# 9. FOOTER
# ----------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6b7280;'>Developed for GDG Hackathon 2026 | Google Gemini API Tier: Free</p>", unsafe_allow_html=True)

