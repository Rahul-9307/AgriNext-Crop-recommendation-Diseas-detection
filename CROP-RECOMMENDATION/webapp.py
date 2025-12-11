import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
from PIL import Image

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AgriNext Crop Recommendation", layout="wide")

# ------------------------------------------
# SAFE IMAGE LOADING
# ------------------------------------------
def load_image(filename):
    return Image.open(os.path.join(os.path.dirname(__file__), filename))

banner = load_image("crop.png")
st.image(banner, use_column_width=True)


# ------------------------------------------
# SAFE CSV LOADING
# ------------------------------------------
csv_path = os.path.join(os.path.dirname(__file__), "Crop_recommendation.csv")
df = pd.read_csv(csv_path)

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# ------------------------------------------
# TRAIN MODEL
# ------------------------------------------
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# ------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------
def predict_crop(n, p, k, temp, hum, ph, rain):
    data = np.array([[n, p, k, temp, hum, ph, rain]])
    return model.predict(data)[0]


# ------------------------------------------
# MAIN UI
# ------------------------------------------
def main():

    # ---------------- ANIMATED FIXED HEADER ----------------
    st.markdown("""
        <style>

        .top-header {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background: linear-gradient(to right, #1e88e5, #42a5f5);
            padding: 14px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 9999;
            border-bottom: 3px solid #0d47a1;
            animation: slideDown 0.9s ease-in-out;
        }

        @keyframes slideDown {
            from { transform: translateY(-80px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .app-title {
            color: white;
            font-size: 30px;
            font-weight: bold;
        }

        .home-btn {
            background: white;
            color: #1e88e5;
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 600;
            text-decoration: none;
            border: 2px solid white;
            transition: 0.3s;
        }

        .home-btn:hover {
            background: #e3f2fd;
            border-color: #bbdefb;
        }

        .top-space { margin-top: 100px; }

        </style>

        <div class="top-header">
            <div class="app-title">AgriNext</div>

            <a class="home-btn" href="https://rahul-9307.github.io/AgriNext/" target="_blank">
                Home
            </a>
        </div>

        <div class="top-space"></div>
    """, unsafe_allow_html=True)


    # ---------------- PAGE TITLE ----------------
    st.markdown("<h1 style='text-align:center;'>🌱 SMART CROP RECOMMENDATIONS</h1>",
                unsafe_allow_html=True)


    # ---------------- SIDEBAR INPUTS ----------------
    st.sidebar.title("Agri🌾Next")
    st.sidebar.header("Enter Crop Details")

    nitrogen = st.sidebar.number_input("Nitrogen (N)", 0.0, 140.0, 0.0)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", 0.0, 145.0, 0.0)
    potassium = st.sidebar.number_input("Potassium (K)", 0.0, 205.0, 0.0)
    temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 51.0, 0.0)
    humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 0.0)
    ph_value = st.sidebar.number_input("pH Level", 0.0, 14.0, 0.0)
    rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 0.0)


    # ---------------- PREDICT BUTTON ----------------
    if st.sidebar.button("Predict"):
        values = np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall])

        if (values == 0).all():
            st.error("⚠️ Please enter valid values before prediction.")
        else:
            crop = predict_crop(*values)

            # ---------- RESULT CARD ----------
            st.markdown(f"""
                <div style="
                    background:#e8f5e9;
                    padding:20px;
                    border-radius:12px;
                    border-left:8px solid #43a047;
                    box-shadow:0 3px 10px rgba(0,0,0,0.1);
                    margin-top:20px;">
                    
                    <h2 style="color:#2e7d32;">🌾 Recommended Crop: <b>{crop}</b></h2>
                    <p style="font-size:16px; color:#1b5e20;">
                        Based on the soil and weather conditions you entered,
                        <b>{crop}</b> is the most suitable crop.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # ---------- TIPS & TRICKS ----------
            st.markdown(f"""
                <div style="
                    background:#e3f2fd;
                    padding:20px;
                    border-radius:12px;
                    border-left:8px solid #1976d2;
                    margin-top:25px;
                    box-shadow:0 3px 10px rgba(0,0,0,0.1);">

                    <h3 style="color:#0d47a1;">✨ Tips & Tricks for {crop.title()}</h3>

                    <ul style="line-height:1.7; font-size:16px; color:#0d47a1;">
                        <li>Maintain proper irrigation cycles.</li>
                        <li>Check soil pH regularly and balance nutrients.</li>
                        <li>Use organic compost for better soil health.</li>
                        <li>Monitor weather (rainfall & humidity).</li>
                        <li>Follow fertilizer recommendations for {crop}.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)


# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    main()
