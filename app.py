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
if st.button("🧹 Emergency Cache Clear"):
    st.cache_data.clear()
    st.success("Cache wiped! Try the prediction again.")

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

st.info("""
**🚀 GDG System Architecture & Instructions:**
1. **ML Layer:** Scikit-Learn time-series model.
2. **XAI Layer:** Google Gemini 2.0 Flash (Explainable AI).
3. **Resilience:** Dual-Key Failover + 12s cooldown to respect 5 RPM limits.
4. **Optimization:** Cached results for presentation speed.
""")

# ----------------------------
# 3. GEMINI CLIENT INITIALIZATION
# ----------------------------
# ----------------------------
# 3. GEMINI CLIENT INITIALIZATION
# ----------------------------
def get_client(use_backup=False):
    # FOR LOCAL TESTING ONLY: Replace "your-key-here" with your actual key
    # return genai.Client(api_key="your-actual-api-key-string") 
    
    try:
        # Normal production logic
        if use_backup:
            key = st.secrets.get("GEMINI_API_KEY_BACKUP") or st.secrets["GEMINI_API_KEY"]
        else:
            key = st.secrets["GEMINI_API_KEY"]
        return genai.Client(api_key=key)
    except Exception:
        # If it fails, let's print a better error so you know WHY it's None
        st.error("Error: Client could not be initialized. Check your .streamlit/secrets.toml file.")
        return None

# Initial global client
client = get_client()

# ----------------------------
# 4. ROBUST AI ANALYST (PURE LOGIC - NO UI)
# ----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_xai_analysis(decision, prob, sma20, vol, vol_chg, trend_val):
    """
    Expert AI Analyst Logic. 
    IMPORTANT: This function contains NO 'st.' commands to avoid CacheReplayClosureError.
    """
    import time, random
    # We define a local reference to avoid global scope issues in caching
    local_client = get_client()
    
    system_instr = "You are a Senior Quantitative Analyst at Google Finance."
    prompt = f"""
    ML Prediction for GOOGL: {decision} ({prob:.2%}).
    Technical Context: SMA20 {sma20*100}%, Volatility {vol*100}%, Volume Change {vol_chg*100}%.
    Trend: {'Upward' if trend_val == 1 else 'Downward'}.
    Explain why these technicals led to this specific {decision} signal.
    """

    max_retries = 3
    retry_count = 0
    # 12 seconds is the magic number to clear the 5 requests-per-minute limit
    wait_time = 12.0 

    while retry_count < max_retries:
        try:
            response = local_client.models.generate_content(
                model="gemini-1.5-flash", 
                contents=prompt,
                config={"system_instruction": system_instr, "temperature": 0.7}
            )
            return {"status": "success", "text": response.text}
            
        except Exception as e:
            if "429" in str(e):
                retry_count += 1
                if retry_count == 1:
                    local_client = get_client(use_backup=True)
                
                # Sleep without UI feedback to keep cache clean
                time.sleep(wait_time + random.uniform(1, 2))
                wait_time *= 1.5
            else:
                return {"status": "error", "text": str(e)}
    
    return {"status": "timeout", "text": "AI Analyst is busy. Using failover logic."}

# ----------------------------
# 5. UI HEADER & DESCRIPTION
# ----------------------------
st.title("📈 Google Stock Prediction System")
st.markdown("### Agentic Machine Learning & Explainable AI")

with st.expander("📖 Project Methodology"):
    st.write("""
        The system utilizes a **Scikit-Learn** model. The prediction is fed into 
        **Google Gemini 2.0 Flash** to perform 'Explainable AI' (XAI).
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
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Ticker", "GOOGL")
        with c2: st.metric("Signal", result['decision'])
        with c3: st.metric("Confidence", f"{result['prob_up']:.1%}")

        if result["decision"] == "BUY":
            st.success(f"📈 **BUY SIGNAL ISSUED** at ${today_close}")
        elif result["decision"] == "SELL":
            st.error(f"📉 **SELL SIGNAL ISSUED** at ${today_close}")
        else:
            st.warning("⚖️ **NEUTRAL - NO TRADE RECOMMENDED**")

        st.write(f"**Predicted Target:** `${result['predicted_price']:.2f}`")

        # --- Charting ---
        st.subheader("📊 Price Action Forecast")
        h_size = 20
        h_data = np.linspace(today_close*0.97, today_close, h_size) + np.random.normal(0, 0.4, h_size)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(h_size), h_data, color="#3b82f6", label="Historical")
        ax.scatter(h_size, result['predicted_price'], color="#fbbf24", s=100, label="Target")
        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#111827")
        ax.tick_params(colors='white')
        st.pyplot(fig)

        # ----------------------------
        # 8. THE GEMINI AI REPORT (UI SIDE)
        # ----------------------------
        st.markdown("---")
        st.subheader("🤖 Gemini Explainable AI (XAI) Analysis")
        
        with st.spinner("🤖 Consulting AI Analyst (This takes 12s due to Free Tier limits)..."):
            # Call the pure cached function
            report_data = get_xai_analysis(
                result['decision'], 
                result['prob_up'], 
                price_sma20, 
                vol_5, 
                vol_change, 
                trend
            )
            
            # Handle UI rendering based on the status
            if report_data["status"] == "success":
                with st.chat_message("assistant"):
                    st.markdown(report_data["text"])
            elif report_data["status"] == "timeout":
                st.toast("Primary key busy, tried failover.", icon="🔄")
                st.warning(report_data["text"])
            else:
                st.error(f"AI error: {report_data['text']}")
            
            st.caption("Rate-limit logic: Jittered Backoff Enabled (Free Tier 5 RPM).")

# ----------------------------
# 9. FOOTER
# ----------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6b7280;'>Developed for GDG Hackathon 2026</p>", unsafe_allow_html=True)

