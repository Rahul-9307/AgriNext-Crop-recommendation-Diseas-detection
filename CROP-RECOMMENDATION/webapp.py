import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from PIL import Image
import datetime
import os
import io

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

st.set_page_config(page_title="AgriNext", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------
# Load Image
# ------------------------------
try:
    image_path = os.path.join(BASE_DIR, "crop.png")
    img = Image.open(image_path)
    st.image(img, use_column_width=True)
except:
    pass

# ------------------------------
# Train Model
# ------------------------------
@st.cache_resource
def train_model():
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
# Prediction
# ------------------------------
def predict_crop(n, p, k, t, h, ph, r):
    input_data = np.array([[n, p, k, t, h, ph, r]])
    return model.predict(input_data)[0]

# ------------------------------
# Farmer Plan (Bilingual)
# ------------------------------
def get_crop_plan(crop, language):

    if language == "English":
        return f"""
Crop: {crop}

Soil:
- Use fertile well-drained soil

Fertilizer:
- Apply balanced NPK

Water:
- Maintain proper irrigation

Tip:
- Monitor pests regularly
- Check market rate before selling
"""
    else:
        return f"""
पीक: {crop}

जमीन:
- सुपीक व निचरा होणारी जमीन वापरा

खत:
- संतुलित NPK खतांचा वापर करा

पाणी:
- योग्य सिंचन व्यवस्था ठेवा

सूचना:
- किड नियंत्रण नियमित करा
- बाजारभाव तपासून विक्री करा
"""

# ------------------------------
# PDF Generator (Cloud Safe)
# ------------------------------
def generate_pdf(crop, plan, accuracy, language):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    title = "AgriNext Smart Crop Report" if language == "English" else "AgriNext स्मार्ट पीक अहवाल"

    elements.append(Paragraph(f"<b>{title}</b>", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Date: {datetime.date.today()}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(f"Recommended Crop: {crop}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(f"Model Accuracy: {round(accuracy*100,2)}%", styles["Normal"]))
    elements.append(Spacer(1, 0.4 * inch))

    for line in plan.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ------------------------------
# UI
# ------------------------------
st.title("🌾 AgriNext - Smart Crop Recommendation")

language = st.selectbox("🌍 Select Language / भाषा निवडा", ["English", "Marathi"])

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

    if language == "English":
        st.success(f"✅ Recommended Crop: {prediction}")
        st.info(f"📊 Model Accuracy: {round(accuracy*100,2)}%")
    else:
        st.success(f"✅ शिफारस केलेले पीक: {prediction}")
        st.info(f"📊 मॉडेल अचूकता: {round(accuracy*100,2)}%")

    plan = get_crop_plan(prediction, language)

    st.markdown("## 📋 Farmer Action Plan")
    st.text(plan)

    pdf_buffer = generate_pdf(prediction, plan, accuracy, language)

    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_buffer,
        file_name="AgriNext_Report.pdf",
        mime="application/pdf"
    )
