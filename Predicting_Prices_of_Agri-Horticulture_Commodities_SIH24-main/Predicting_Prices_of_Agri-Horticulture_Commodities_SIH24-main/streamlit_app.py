import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import altair as alt
from sklearn.tree import DecisionTreeRegressor

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AgriNext 🌾",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AgriNext – Crop Price Prediction")
st.caption("AI based agriculture market forecasting (Educational Project)")

# -------------------------------------------------
# AUTO FIND STATIC FOLDER
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

csv_files = [f for f in os.listdir(STATIC_DIR) if f.lower().endswith(".csv")]

if not csv_files:
    st.error("❌ No CSV files found inside static folder")
    st.stop()

CROPS = sorted([os.path.splitext(f)[0] for f in csv_files])

# -------------------------------------------------
# BASE PRICE & RAINFALL
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
def load_model(csv_path):
    return Commodity(csv_path)

# -------------------------------------------------
# UI INPUTS
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    crop = st.selectbox("🌱 Select Crop", CROPS)

with col2:
    month = st.selectbox("📅 Month", list(range(1, 13)))

with col3:
    year = st.selectbox("📆 Year", list(range(2024, 2031)))

rainfall = ANNUAL_RAINFALL[month - 1]

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("🔍 Predict Price"):
    csv_path = os.path.join(STATIC_DIR, f"{crop}.csv")
    model = load_model(csv_path)

    wpi = model.predict(month, year, rainfall)
    base = BASE_PRICE.get(crop.capitalize(), 2000)
    price = round((wpi * base) / 100, 2)

    st.success(f"💰 Predicted Market Price for **{crop}**")
    st.metric("₹ / Quintal", f"₹ {price}")

    # -----------------------------
    # 6 MONTH FORECAST (NO ZOOM)
    # -----------------------------
    st.subheader("📈 6-Month Price Forecast")

    forecast = []
    months = []

    for i in range(1, 7):
        m = month + i if month + i <= 12 else month + i - 12
        y = year if month + i <= 12 else year + 1
        r = ANNUAL_RAINFALL[m - 1]
        p = model.predict(m, y, r)
        forecast.append(round((p * base) / 100, 2))
        months.append(f"+{i}")

    df = pd.DataFrame({
        "Month": months,
        "Price": forecast
    })

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Month:N", title="Month"),
            y=alt.Y("Price:Q", title="Price (₹)", scale=alt.Scale(zero=False)),
            tooltip=["Month", "Price"]
        )
        .properties(width=700, height=400)
        .interactive(False)   # 🔴 zoom disabled
    )

    st.altair_chart(chart, use_container_width=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <style>
    .agrifooter {
        text-align: center;
        padding: 18px;
        font-size: 14px;
        color: #888;
    }
    .agrifooter span {
        color: #2ecc71;
        font-weight: 600;
    }
    .agrifooter a {
        text-decoration: none;
        color: #1abc9c;
        font-weight: 500;
    }
    .agrifooter a:hover {
        text-decoration: underline;
    }
    </style>

    <div class="agrifooter">
        🌾 <span>AgriNext</span> – Smart Agriculture Intelligence Platform <br>
        👨‍💻 Developed by <b>Rahul Patil</b> <br>
        🔗 <a href="https://github.com/Rahul-9307" target="_blank">GitHub</a> |
        📘 Educational Project | Made with ❤️ for Farmers
    </div>
    """,
    unsafe_allow_html=True
)
