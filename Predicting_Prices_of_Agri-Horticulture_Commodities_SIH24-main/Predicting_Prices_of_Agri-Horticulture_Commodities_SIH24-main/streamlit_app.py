import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
import os

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AgriNext 🌾",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AgriNext – Crop Price Prediction")
st.caption("AI based agriculture price forecasting (Educational Project)")

# -------------------------------------------------
# SAFE BASE PATH (VERY IMPORTANT)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(
    BASE_DIR,
    "Predicting_Prices_of_Agri-Horticulture_Commodities_SIH24-main",
    "static"
)

# -------------------------------------------------
# CONSTANT DATA
# -------------------------------------------------
BASE_PRICE = {
    "Paddy": 1245.5, "Arhar": 3200, "Bajra": 1175, "Barley": 980,
    "Copra": 5100, "Cotton": 3600, "Sesamum": 4200, "Gram": 2800,
    "Groundnut": 3700, "Jowar": 1520, "Maize": 1175, "Masoor": 2800,
    "Moong": 3500, "Niger": 3500, "Ragi": 1500, "Rape": 2500,
    "Jute": 1675, "Safflower": 2500, "Soyabean": 2200,
    "Sugarcane": 2250, "Sunflower": 3700, "Urad": 4300, "Wheat": 1350
}

ANNUAL_RAINFALL = [29, 21, 37.5, 30.7, 52.6, 150, 299, 251.7, 179.2, 70.5, 39.8, 10.9]

# -------------------------------------------------
# MODEL CLASS
# -------------------------------------------------
class Commodity:
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        self.X = data.iloc[:, :-1].values
        self.Y = data.iloc[:, 3].values

        depth = random.randint(7, 15)
        self.model = DecisionTreeRegressor(max_depth=depth)
        self.model.fit(self.X, self.Y)

    def predict(self, month, year, rainfall):
        X = np.array([[month, year, rainfall]])
        return self.model.predict(X)[0]

# -------------------------------------------------
# LOAD MODEL (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_model(crop):
    file_path = os.path.join(DATASET_PATH, f"{crop}.csv")

    if not os.path.exists(file_path):
        st.error(f"❌ CSV file not found:\n{file_path}")
        st.stop()

    return Commodity(file_path)

# -------------------------------------------------
# UI CONTROLS
# -------------------------------------------------
crop_name = st.selectbox("🌱 Select Crop", sorted(BASE_PRICE.keys()))
month = st.selectbox("📅 Select Month", list(range(1, 13)))
year = st.selectbox("📆 Select Year", list(range(2024, 2031)))

rainfall = ANNUAL_RAINFALL[month - 1]

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("🔍 Predict Price"):
    model = load_model(crop_name)

    wpi = model.predict(month, year, rainfall)
    price = round((wpi * BASE_PRICE[crop_name]) / 100, 2)

    st.success(f"💰 Predicted Market Price for **{crop_name}**")
    st.metric("Price (₹ / Quintal)", f"₹ {price}")

    # -----------------------------
    # 6 MONTH FORECAST
    # -----------------------------
    st.subheader("📈 6 Month Forecast")

    forecast_prices = []
    months = []

    for i in range(1, 7):
        m = month + i if month + i <= 12 else month + i - 12
        y = year if month + i <= 12 else year + 1
        r = ANNUAL_RAINFALL[m - 1]

        pred = model.predict(m, y, r)
        final_price = round((pred * BASE_PRICE[crop_name]) / 100, 2)

        months.append(f"+{i}")
        forecast_prices.append(final_price)

    df = pd.DataFrame({
        "Month": months,
        "Price": forecast_prices
    }).set_index("Month")

    st.line_chart(df)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.caption("👨‍💻 Developed by AgriNext Team | Streamlit Version")
