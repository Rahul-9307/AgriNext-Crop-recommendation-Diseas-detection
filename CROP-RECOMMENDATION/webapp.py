import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from PIL import Image
import datetime
import os

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="AgriNext", layout="wide")

# ------------------------------
# Load Image Safely
# ------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(BASE_DIR, "crop.png")
    img = Image.open(image_path)
    st.image(img, use_column_width=True)
except:
    pass

# ------------------------------
# Train Model (Cloud Safe)
# ------------------------------
@st.cache_resource
def train_model():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "Crop_recommendation.csv")

    df = pd.read_csv(csv_path)

    X = df[['N','P','K','temperature','humidity','ph','rainfall']]
    y = df['label']

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(Xtrain, Ytrain)

    accuracy = metrics.accuracy_score(Ytest, model.predict(Xtest))
    return model, accuracy

model, accuracy = train_model()

# ------------------------------
# Prediction Function
# ------------------------------
def predict_crop(n, p, k, t, h, ph, r):
    input_data = np.array([[n, p, k, t, h, ph, r]])
    return model.predict(input_data)[0]

# ------------------------------
# Farmer Plan Generator
# ------------------------------
def get_crop_plan(crop):

    if crop.lower() == "rice":
        return """
🌾 RICE FARMING PLAN

Soil:
- Clay or loamy soil
- pH 5.5 – 7.0

Fertilizer Plan:
- Basal: 50kg Urea + 50kg DAP
- 30 Days: 25kg Urea
- 60 Days: 25kg Urea

Water Management:
- Maintain 2–5 cm water level
- Avoid complete drying

Harvest:
- 100–120 days
- Expected Yield: 18–25 quintals per acre
"""

    elif crop.lower() == "maize":
        return """
🌽 MAIZE FARMING PLAN

Soil:
- Well-drained fertile soil

Fertilizer:
- Balanced NPK application

Water:
- Moderate irrigation

Harvest:
- 90–110 days
"""

    else:
        return f"""
🌱 {crop.upper()} FARMING PLAN

Soil:
- Well-drained fertile soil

Fertilizer:
- Use recommended NPK doses

Water:
- Maintain proper irrigation

Tip:
- Monitor pests regularly
- Check market rate before selling
"""

# ------------------------------
# Streamlit UI
# ------------------------------
def main():

    st.title("🌾 AgriNext - Smart Crop Recommendation")

    st.sidebar.title("Enter Crop Details")

    nitrogen = st.sidebar.number_input("Nitrogen (N)", 0.0, 140.0, 50.0)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", 0.0, 145.0, 40.0)
    potassium = st.sidebar.number_input("Potassium (K)", 0.0, 205.0, 40.0)
    temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    ph = st.sidebar.number_input("pH Level", 0.0, 14.0, 6.5)
    rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

    if st.sidebar.button("Predict Crop"):

        prediction = predict_crop(
            nitrogen, phosphorus, potassium,
            temperature, humidity, ph, rainfall
        )

        st.success(f"✅ Recommended Crop: {prediction}")
        st.info(f"📊 Model Accuracy: {round(accuracy*100,2)}%")
        st.balloons()

        plan = get_crop_plan(prediction)

        st.markdown("## 📋 Complete Farmer Action Plan")
        st.text(plan)

        report_text = f"""
AgriNext Smart Crop Report
Date: {datetime.date.today()}

Recommended Crop: {prediction}
Model Accuracy: {round(accuracy*100,2)}%

{plan}
"""

        st.download_button(
            label="📥 Download Full Report",
            data=report_text,
            file_name="AgriNext_Report.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
