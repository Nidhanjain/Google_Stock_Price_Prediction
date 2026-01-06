import pickle
import pandas as pd
import numpy as np

# ----------------------------
# 1. Load ML Artifacts
# ----------------------------
# Ensure these files are in the same folder as predict.py
try:
    clf = pickle.load(open("stock_model.pkl", "rb"))
    clf_scaler = pickle.load(open("scaler.pkl", "rb"))
    clf_features = pickle.load(open("features.pkl", "rb"))

    reg = pickle.load(open("return_model.pkl", "rb"))
    reg_scaler = pickle.load(open("return_scaler.pkl", "rb"))
    reg_features = pickle.load(open("return_features.pkl", "rb"))
except FileNotFoundError as e:
    print(f"Error: Missing model files. {e}")

def predict_today(today_dict, today_close, threshold_buy=0.55, threshold_sell=0.45):
    """
    Core Prediction Engine:
    - Classification: Determines direction (Buy/Sell/No Trade)
    - Regression: Determines target price
    """
    
    # 1. ---------- PREPARE DATA ----------
    # Convert dict to DataFrame and ensure columns match training order
    df_input = pd.DataFrame([today_dict])
    
    # 2. ---------- CLASSIFICATION (Direction) ----------
    X_clf = df_input[clf_features]
    X_clf_scaled = clf_scaler.transform(X_clf)
    
    # Get probability of class 1 (Price Up)
    prob_up = clf.predict_proba(X_clf_scaled)[0, 1]

    # 3. ---------- REGRESSION (Target Price) ----------
    X_reg = df_input[reg_features]
    X_reg_scaled = reg_scaler.transform(X_reg)

    predicted_return = reg.predict(X_reg_scaled)[0]
    predicted_price = today_close * (1 + predicted_return)

    # Confidence Band (1% Margin of Error)
    error_band = 0.01 
    lower_price = today_close * (1 + predicted_return - error_band)
    upper_price = today_close * (1 + predicted_return + error_band)

    # 4. ---------- DECISION LOGIC (The "Brain") ----------
   # 4. ---------- DECISION LOGIC ----------
    
    # FIRST: Check the ML Model Probability (Priority)
    if prob_up >= threshold_buy:
        decision = "BUY"
    elif prob_up <= threshold_sell:
        decision = "SELL"
    
    # SECOND: Apply "Safety Overrides" only if the model is unsure
    else:
        # If SMA20 is very high but volume is dropping, it's a "Sell" trap
        if today_dict['price_sma20'] > 0.12 and today_dict['vol_change'] < 0:
            decision = "SELL"
        # If price is crashed way below the average, it's a "Buy" bounce
        elif today_dict['price_sma20'] < -0.08:
            decision = "BUY"
        else:
            decision = "NO TRADE"
            
    # 5. ---------- RETURN RESULTS ----------
    return {
        "decision": decision,
        "prob_up": prob_up,
        "predicted_return": predicted_return,
        "predicted_price": predicted_price,
        "lower_price": lower_price,
        "upper_price": upper_price,
        "features_used": list(today_dict.keys())
    }

