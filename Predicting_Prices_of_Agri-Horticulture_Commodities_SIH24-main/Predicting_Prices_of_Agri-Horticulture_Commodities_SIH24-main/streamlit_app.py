import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeRegressor
import os

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="AgriNext 🌾", layout="wide")
st.title("🌾 AgriNext – Crop Price Prediction")

# -------------------------------------------------
# AUTO FIND STATIC FOLDER (NO PATH BUG)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = None
for root, dirs, files in os.walk(BASE_DIR):
    if "static" in dirs:
        STATIC_DIR = os.path.join(root, "static")
        break

if STATIC_DIR is None:
    st.error("❌ static folder not found in repository")
    st.stop()

# -------------------------------------------------
# LOAD AVAILABLE CSV FILES
# -------------------------------------------------
csv_files = [f for f in os.listdir(STATIC_DIR) if f.lower().endswith(".csv")]

if not csv_files:
    st.error("❌ No CSV files found inside static folder")
    st.stop()

# Crop names from CSV
CROPS = sorted([os.path.splitext(f)[0] for f in csv_files])

# -------------------------------------------------
# BASE PRICE (OPTIONAL DEFAULT)
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
        df = pd.read_csv(csv_path)
        self.X = df.iloc[:, :-1].values
        self.Y = df.iloc[:, 3].values

        self.model = DecisionTreeRegressor(
            max_depth=random.randint(7, 15)
        )
        self.model.fit(self.X, self.Y)

    def predict(self, m, y, r):
        return self.model.predict(np.array([[m, y, r]]))[0]

# -------------------------------------------------
# CACHE MODEL
# -------------------------------------------------
@st.cache_resource
def load_model(csv_file):
    return Commodity(csv_file)

# -------------------------------------------------
# UI
# -------------------------------------------------
crop = st.selectbox("🌱 Select Crop", CROPS)
month = st.selectbox("📅 Month", list(range(1, 13)))
year = st.selectbox("📆 Year", list(range(2024, 2031)))

rainfall = ANNUAL_RAINFALL[month - 1]

# -------------------------------------------------
# PREDICT
# -------------------------------------------------
if st.button("🔍 Predict Price"):
    csv_path = os.path.join(STATIC_DIR, f"{crop}.csv")

    model = load_model(csv_path)
    wpi = model.predict(month, year, rainfall)

    base = BASE_PRICE.get(crop.capitalize(), 2000)
    price = round((wpi * base) / 100, 2)

    st.success(f"💰 Predicted Price for **{crop}**")
    st.metric("₹ / Quintal", f"₹ {price}")

    # Forecast
    st.subheader("📈 6 Month Forecast")
    prices = []

    for i in range(1, 7):
        m = month + i if month + i <= 12 else month + i - 12
        y = year if month + i <= 12 else year + 1
        r = ANNUAL_RAINFALL[m - 1]
        p = model.predict(m, y, r)
        prices.append(round((p * base) / 100, 2))

    st.line_chart(pd.DataFrame(prices, columns=["Price"]))

st.caption("👨‍💻 AgriNext | Streamlit Stable Build ✅")
