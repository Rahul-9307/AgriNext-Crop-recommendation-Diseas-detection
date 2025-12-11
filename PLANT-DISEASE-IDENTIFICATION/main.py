import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# -----------------------------
# BASE DIRECTORY
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# SAFE IMAGE LOADING
# -----------------------------
def load_local_image(filename):
    return Image.open(os.path.join(BASE_DIR, filename))

# -----------------------------
# MODEL LOADING (AUTO-HANDLE .keras / .h5)
# -----------------------------
def load_model_file():
    keras_path = os.path.join(BASE_DIR, "trained_plant_disease_model.keras")
    h5_path = os.path.join(BASE_DIR, "trained_plant_disease_model.h5")

    # Check .keras file
    if os.path.exists(keras_path):
        st.success("Loaded model: trained_plant_disease_model.keras")
        return tf.keras.models.load_model(keras_path, compile=False)

    # Check .h5 file
    elif os.path.exists(h5_path):
        st.success("Loaded model: trained_plant_disease_model.h5")
        return tf.keras.models.load_model(h5_path, compile=False)

    # Model missing
    else:
        st.error("❌ Model file not found! Please place model in same folder as main.py")
        st.stop()

model = load_model_file()

# -----------------------------
# MODEL PREDICTION
# -----------------------------
def model_prediction(test_image_path):
    image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.sidebar.title("AgriSens")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Load Diseases.png safely
try:
    img = load_local_image("Diseases.png")
    st.image(img)
except:
    st.warning("⚠️ Diseases.png file not found in project folder.")

# HOME PAGE
if app_mode == "HOME":
    st.markdown("<h1 style='text-align:center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)

# PREDICTION PAGE
elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")

    uploaded_file = st.file_uploader("Choose an Image:")

    if uploaded_file:
        st.image(uploaded_file, width=300)

        temp_file_path = os.path.join(BASE_DIR, "temp.jpg")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Predict"):
            st.snow()

            # LABELS
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

            # PREDICT
            result_index = model_prediction(temp_file_path)
            predicted = class_name[result_index]

            st.success(f"Model is predicting it's **{predicted}** 🌿")

            # -------------------------------------------------------
            # AUTO-FERTILIZER RECOMMENDATION CARD (PREMIUM DESIGN)
            # -------------------------------------------------------
            st.markdown("""
            <div style='padding:20px; border-radius:18px; background:#f5faff;
                        box-shadow:0 4px 12px rgba(0,0,0,0.1); font-family: Poppins;'>

                <h2 style='color:#2b6a4b; text-align:center;'>🌱 Auto-Fertilizer Recommendation</h2>

                <div style='margin-top:15px;'>
                    <h3 style='color:#d35400;'>🍅 Disease: Tomato Early Blight</h3>
                </div>

                <div style='background:white; padding:15px; border-radius:12px; margin-top:10px;'>
                    <h4 style='color:#2c3e50;'>🧪 Fungicide Treatment</h4>
                    <p><b>Mancozeb 75% WP</b><br>
                    Quantity: <b>2g per liter</b><br>
                    Frequency: <b>Every 7 days</b><br>
                    Duration: <b>2–3 cycles</b></p>
                </div>

                <div style='background:white; padding:15px; border-radius:12px; margin-top:10px;'>
                    <h4 style='color:#2c3e50;'>🌿 Nutrient Booster</h4>
                    <p><b>NPK 19:19:19</b><br>
                    Quantity: <b>5g per liter</b><br>
                    Timing: <b>After 3 days of fungicide spray</b></p>
                </div>

                <div style='background:white; padding:15px; border-radius:12px; margin-top:10px;'>
                    <h4 style='color:#2c3e50;'>🌱 Soil Reviver</h4>
                    <p><b>Trichoderma viride</b><br>
                    5 kg/acre mixed with FYM</p>
                </div>

                <div style='background:#e8f8f5; padding:15px; border-radius:12px; margin-top:10px;'>
                    <h4 style='color:#117864;'>🌤️ Weather Precautions</h4>
                    <ul>
                        <li>Humidity > 80% = High infection risk</li>
                        <li>Avoid spraying during rain/wind</li>
                        <li>Prefer early morning/evening</li>
                    </ul>
                </div>

                <div style='margin-top:20px; padding:15px; background:#fff3cd; border-radius:12px;'>
                    <h4 style='color:#856404;'>🗓 Next Spray Reminder</h4>
                    Spray again after <b>7 days</b>.
                </div>

            </div>
            """, unsafe_allow_html=True)
