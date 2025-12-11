import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
from PIL import Image

warnings.filterwarnings("ignore")

# ------------------------------------------
# SAFE IMAGE LOADING
# ------------------------------------------
def load_image(filename):
    return Image.open(os.path.join(os.path.dirname(__file__), filename))

st.set_page_config(page_title="AgriNext Crop Recommendation", layout="wide")

# Top Banner Image
banner = load_image("crop.png")
st.image(banner, use_column_width=True)

# ------------------------------------------
# SAFE CSV LOADING
# ------------------------------------------
csv_path = os.path.join(os.path.dirname(__file__), "Crop_recommendation.csv")
df = pd.read_csv(csv_path)

# ------------------------------------------
# FEATURE SELECTION
# ------------------------------------------
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# ------------------------------------------
# TRAIN MODEL
# ------------------------------------------
model = RandomForestClassifier(n_estimators=50, random_state=10)
model.fit(X, y)

# ------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------
def predict_crop(n, p, k, temp, hum, ph, rain):
    data = np.array([[n, p, k, temp, hum, ph, rain]])
    return model.predict(data)[0]


# ------------------------------------------
# STREAMLIT FRONTEND
# ------------------------------------------
def main():

    # -------------------- HEADER BAR ----------------------
    st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:12px 22px; background:#1e88e5; border-radius:8px;
                    margin-bottom:20px; box-shadow: 0 3px 8px rgba(0,0,0,0.3);">

            <h1 style="color:white; margin:0; font-size:28px;">AgriNext</h1>

            <a href="https://rahul-9307.github.io/AgriNext/" target="_blank"
                style="padding:8px 18px; background:white; color:#1e88e5;
                border-radius:6px; text-decoration:none; font-weight:600;">
                Home
            </a>

        </div>
    """, unsafe_allow_html=True)

    # -------------------- FLOATING BUTTON ----------------------
    st.markdown("""
        <style>
        .floating-btn {
            position: fixed;
            bottom: 18px;
            right: 18px;
            background:#1e88e5;
            color:white;
            padding:12px 25px;
            border-radius:30px;
            font-size:16px;
            font-weight:bold;
            text-decoration:none;
            box-shadow:0 4px 10px rgba(0,0,0,0.4);
            z-index:9999;
        }
        .floating-btn:hover {
            background:#1669c4;
        }
        </style>

        <a class="floating-btn" href="https://rahul-9307.github.io/AgriNext/" target="_blank">AgriNext</a>
    """, unsafe_allow_html=True)

    # -------------------- TITLE ----------------------
    st.markdown("<h2 style='text-align:center; margin-top:10px;'>🌱 Smart Crop Recommendation System</h2>",
                unsafe_allow_html=True)

    # -------------------- SIDEBAR ----------------------
    st.sidebar.header("Enter Soil & Weather Values")

    nitrogen = st.sidebar.number_input("Nitrogen (N)", 0.0, 200.0, 0.0)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", 0.0, 200.0, 0.0)
    potassium = st.sidebar.number_input("Potassium (K)", 0.0, 200.0, 0.0)
    temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 60.0, 0.0)
    humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 0.0)
    ph_level = st.sidebar.number_input("Soil pH", 0.0, 14.0, 0.0)
    rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 0.0)

    # -------------------- PREDICT BUTTON ----------------------
    if st.sidebar.button("Predict Crop"):
        result = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph_level, rainfall)
        st.success(f"🌾 Recommended Crop: **{result}**")


# RUN APP
if __name__ == "__main__":
    main()
