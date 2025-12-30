# 📈 Google Stock Price Prediction System 

An interactive Machine Learning dashboard that predicts the next-day price movement of Google (GOOGL) stock using a dual-model approach (Classification + Regression).

## 🚀 Overview
This project was developed for the **GDG (Google Developer Groups)** community. It utilizes historical data (2016–2021) to analyze technical indicators and provide actionable trading signals.

### Key Features:
* **Directional Prediction:** Classification model determines if the stock is likely to go **UP**, **DOWN**, or stay **NEUTRAL**.
* **Price Forecasting:** Regression model predicts the specific target price for the next trading day.
* **Confidence Intervals:** Displays a 1% error band to provide a "Safe Zone" for price targets.
* **Interactive Dashboard:** Built with Streamlit, allowing users to input live technical indicators.
* **Visual Trend Analysis:** Real-time plotting of recent price history vs. predicted future price.

---

## 🧠 The Models
The system uses two distinct models saved as `.pkl` artifacts:
1.  **Classification Model (`stock_model.pkl`):** Trained to calculate the probability of an upward move.
2.  **Regression Model (`return_model.pkl`):** Trained to predict the percentage return based on 5-day and 10-day price momentum.

### ⚠️ Note on "Mean Reversion Bias"
During testing, we observed that the model exhibits a **Mean Reversion Bias**. 
* **Behavior:** The model often issues a "BUY" signal during sharp price drops.
* **Reason:** Because the training data (2016–2021) was a period of strong recovery for Google, the model learned that "dips" are almost always followed by immediate "bounces."
* **Recommendation:** This tool should be used alongside broader market sentiment (S&P 500) and not as a standalone financial advisor.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/google-stock-prediction.git](https://github.com/YOUR_USERNAME/google-stock-prediction.git)
   cd google-stock-prediction
