import pickle
import pandas as pd

# ----------------------------
# Load CLASSIFICATION artifacts
# ----------------------------
clf = pickle.load(open("stock_model.pkl", "rb"))
clf_scaler = pickle.load(open("scaler.pkl", "rb"))
clf_features = pickle.load(open("features.pkl", "rb"))

# ----------------------------
# Load REGRESSION artifacts
# ----------------------------
reg = pickle.load(open("return_model.pkl", "rb"))
reg_scaler = pickle.load(open("return_scaler.pkl", "rb"))
reg_features = pickle.load(open("return_features.pkl", "rb"))


def predict_today(today_dict, today_close, threshold_buy=0.60, threshold_sell=0.45):
    """
    today_dict: dict of today's engineered features
    today_close: today's close price
    """

    # ---------- CLASSIFICATION ----------
    X_clf = pd.DataFrame([today_dict])[clf_features]
    X_clf_scaled = clf_scaler.transform(X_clf)

    # Safety check for model type
    try:
        prob_up = clf.predict_proba(X_clf_scaled)[0, 1]
    except AttributeError:
        # Fallback if clf is a Regression model by mistake
        prediction = clf.predict(X_clf_scaled)[0]
        prob_up = 1.0 if prediction > 0 else 0.0

    prob_up = clf.predict_proba(X_clf_scaled)[0, 1]

    # --- THE OVERRIDE LOGIC ---
    # 1. Force SELL if volume is dropping while price is overextended
    if today_dict['price_sma20'] > 0.10 and today_dict['vol_change'] < 0:
        decision = "SELL"
    
    # 2. Force SELL if it's a massive crash (avoiding the "Buy the Dip" bias)
    elif today_dict['price_sma20'] < -0.07:
        decision = "SELL"

    # 3. Otherwise, use the model's prediction
    elif prob_up >= threshold_buy:
        decision = "BUY"
    elif prob_up <= threshold_sell:
        decision = "SELL"
    else:
        return {
            "decision": "NO TRADE",
            "prob_up": prob_up
        }

    # ---------- REGRESSION ----------
    X_reg = pd.DataFrame([today_dict])[reg_features]
    X_reg_scaled = reg_scaler.transform(X_reg)

    predicted_return = reg.predict(X_reg_scaled)[0]
    predicted_price = today_close * (1 + predicted_return)

    # ---------- CONFIDENCE BAND ----------
    error_band = 0.01  # 1% MAE assumption
    lower_price = today_close * (1 + predicted_return - error_band)
    upper_price = today_close * (1 + predicted_return + error_band)
    

    return {
        "decision": decision,
        "prob_up": prob_up,
        "predicted_return": predicted_return,
        "predicted_price": predicted_price,
        "lower_price": lower_price,
        "upper_price": upper_price
    }

