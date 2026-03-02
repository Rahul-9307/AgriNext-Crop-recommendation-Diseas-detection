import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from PIL import Image
import datetime

# PDF Libraries
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="AgriNext", layout="wide")

# ------------------------------
# Load Image
# ------------------------------
try:
    img = Image.open("crop.png")
    st.image(img, use_column_width=True)
except:
    pass

# ------------------------------
# Train Model (Cached)
# ------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("Crop_recommendation.csv")
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
RICE FARMING PLAN

Soil:
- Clay or loamy soil
- pH 5.5 – 7.0

Fertilizer:
- Basal: 50kg Urea + 50kg DAP
- 30 Days: 25kg Urea
- 60 Days: 25kg Urea

Water:
- Maintain 2–5 cm water level

Harvest:
- 100–120 days
- Yield: 18–25 quintals per acre
"""

    else:
        return f"""
{crop.upper()} FARMING PLAN

Soil:
- Well drained fertile soil

Fertilizer:
- Balanced NPK usage

Water:
- Proper irrigation management

Harvest:
- Monitor crop maturity stage

Tip:
- Check local market rate before selling
"""

# ------------------------------
# PDF Generator
# ------------------------------
def generate_pdf(crop, plan_text):

    file_name = f"{crop}_report.pdf"
    doc = SimpleDocTemplate(file_name, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>AgriNext Smart Crop Report</b>", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"<b>Recommended Crop:</b> {crop}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"<b>Date:</b> {datetime.date.today()}", styles["Normal"]))
    elements.append(Spacer(1, 0.5 * inch))

    for line in plan_text.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)
    return file_name

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

        # Show Farmer Plan
        plan = get_crop_plan(prediction)

        st.markdown("## 📋 Complete Farmer Action Plan")
        st.text(plan)

        # Generate PDF
        pdf_file = generate_pdf(prediction, plan)

        with open(pdf_file, "rb") as file:
            st.download_button(
                label="📥 Download Full Report",
                data=file,
                file_name=pdf_file,
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()
