import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
from PIL import Image

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Agri🌾Next Crop Recommendation", layout="wide")

# ---------------------------------------
# LOAD IMAGE
# ---------------------------------------
def load_image(filename):
    return Image.open(os.path.join(os.path.dirname(__file__), filename))

banner = load_image("crop.png")
st.image(banner, use_column_width=True)

# ---------------------------------------
# LOAD CSV
# ---------------------------------------
csv_path = os.path.join(os.path.dirname(__file__), "Crop_recommendation.csv")
df = pd.read_csv(csv_path)

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# ---------------------------------------
# TRAIN MODEL
# ---------------------------------------
model = RandomForestClassifier(n_estimators=60, random_state=42)
model.fit(X, y)

# ---------------------------------------
# PREDICT FUNCTION
# ---------------------------------------
def predict_crop(n, p, k, temp, hum, ph, rain):
    data = np.array([[n, p, k, temp, hum, ph, rain]])
    return model.predict(data)[0]

# ---------------------------------------
# MAIN UI
# ---------------------------------------
def main():

    # SIMPLE HEADER (NO CSS)
    st.title("AgriNext - Smart Crop Recommendation")

    # SIDEBAR
    st.sidebar.title("Agri🌾Next")
    st.sidebar.title("Enter Crop Details")

    nitrogen = st.sidebar.number_input("Nitrogen (N)", 0.0, 140.0, 0.0)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", 0.0, 145.0, 0.0)
    potassium = st.sidebar.number_input("Potassium (K)", 0.0, 205.0, 0.0)
    temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 51.0, 0.0)
    humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 0.0)
    ph_value = st.sidebar.number_input("pH Level", 0.0, 14.0, 0.0)
    rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 0.0)

    # PREDICT BUTTON
    if st.sidebar.button("Predict"):
        values = np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall])

        if (values == 0).all():
            st.error("Please fill valid values before prediction.")
        else:
            crop = predict_crop(*values)

            st.subheader("🌾 Recommended Crop")
            st.success(f"{crop}")

           # TIPS
            st.subheader("✨ Tips & Tricks")
            st.write(f"""
- Maintain soil moisture properly.  
- Apply recommended fertilizers for **{crop}**.  
- Monitor pH and rainfall conditions.  
- Use organic compost for better soil health.  
- Ensure proper sunlight and irrigation.  
""") 
# MARATHI TIPS
st.subheader("🌾 शेतकऱ्यांसाठी टिप्स (Marathi)")
st.write(f"""
- मातीतील आर्द्रता योग्य प्रमाणात ठेवावी.  
- **{crop}** पिकासाठी शिफारस केलेले खत वेळेवर वापरावे.  
- मातीचे pH आणि पावसाचे प्रमाण तपासत राहावे.  
- सेंद्रिय खते (कंपोस्ट) वापरल्यास उत्पादन वाढते.  
- योग्य सूर्यप्रकाश आणि पाणी व्यवस्थापन करणे महत्वाचे आहे.  
""")
# SUPPORT MESSAGE
st.subheader("🤝 Support")
st.write("""
**Support by AgriNext Team**  
For any help or guidance, feel free to reach out to us.  
""")


# RUN APP
if __name__ == "__main__":
    main()


