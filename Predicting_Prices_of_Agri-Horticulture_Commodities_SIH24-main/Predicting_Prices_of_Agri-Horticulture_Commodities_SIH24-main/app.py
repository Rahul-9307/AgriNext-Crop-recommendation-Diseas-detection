import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import crops

# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(
    page_title="AgriNext 🌾",
    layout="centered"
)

st.title("🌾 AgriNext – Crop Price Prediction")
st.caption("Smart Agriculture | SIH Project")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------
# DATA
# ----------------------------------------
commodity_dict = {
    "Arhar": "static/Arhar.csv",
    "Bajra": "static/Bajra.csv",
    "Barley": "static/Barley.csv",
    "Cotton": "static/Cotton.csv",
    "Wheat": "static/Wheat.csv",
}

annual_rainfall = [29, 21, 37.5, 30.7, 52.6, 150, 299, 251.7, 179.2, 70.5, 39.8, 10.9]

base_price = {
    "Arhar": 3200,
    "Bajra": 1175,
    "Barley": 980,
    "Cotton": 3600,
    "Wheat": 1350
}

# ----------------------------------------
# MODEL CLASS
# ----------------------------------------
class Commodity:
    def __init__(self, name, csv_file):
        path = os.path.join(BASE_DIR, csv_file)
        data = pd.read_csv(path)

        self.X = data.iloc[:, :-1].values
        self.Y = data.iloc[:, 3].values

        from sklearn.tree import DecisionTreeRegressor
        depth = random.randrange(7, 18)
        self.model = DecisionTreeRegressor(max_depth=depth)
        self.model.fit(self.X, self.Y)

        self.name = name

    def predict(self, month, year, rainfall):
        value = np.array([month, year, rainfall]).reshape(1, 3)
        return self.model.predict(value)[0]

# ----------------------------------------
# LOAD MODELS (CACHED)
# ----------------------------------------
@st.cache_resource
def load_models():
    models = {}
    for name, csv in commodity_dict.items():
        models[name] = Commodity(name, csv)
    return models

models = load_models()

# ----------------------------------------
# UI
# ----------------------------------------
crop_name = st.selectbox("🌱 Select Crop", list(models.keys()))

month = st.slider("📅 Month", 1, 12, datetime.now().month)
year = st.slider("📆 Year", 2024, 2030, datetime.now().year)
rainfall = st.slider("🌧️ Rainfall (mm)", 0.0, 300.0, annual_rainfall[month-1])

if st.button("🔮 Predict Price"):
    model = models[crop_name]
    wpi = model.predict(month, year, rainfall)

    price = round((wpi * base_price[crop_name]) / 100, 2)

    st.success(f"💰 Estimated Price: ₹ {price}")

    crop_data = crops.crop(crop_name.lower())
    st.image(crop_data[0], caption=crop_name)
    st.write("📍 Prime Location:", crop_data[1])
    st.write("🌾 Crop Type:", crop_data[2])
    st.write("🌍 Export:", crop_data[3])
