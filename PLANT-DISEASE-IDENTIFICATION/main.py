import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.sidebar.title("AgriSens")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# --------------------------
# MODEL LOADING SAFE METHOD
# --------------------------
def load_model_safe():
    if os.path.exists("trained_plant_disease_model.keras"):
        return tf.keras.models.load_model("trained_plant_disease_model.keras", compile=False)
    elif os.path.exists("trained_plant_disease_model.h5"):
        return tf.keras.models.load_model("trained_plant_disease_model.h5", compile=False)
    else:
        st.error("❌ Model file missing! Please upload trained_plant_disease_model.keras or .h5")
        st.stop()

model = load_model_safe()

# --------------------------
# PREDICTION FUNCTION
# --------------------------
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# --------------------------
# HEADER IMAGE
# --------------------------
try:
    img = Image.open("Diseases.png")
    st.image(img)
except:
    st.warning("⚠️ Diseases.png not found")

# --------------------------
# HOME PAGE
# --------------------------
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)

# --------------------------
# DISEASE RECOGNITION PAGE
# --------------------------
elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")
    uploaded_file = st.file_uploader("Choose an Image:")

    if uploaded_file:
        st.image(uploaded_file, use_column_width=True)

        # Save temporarily
        temp_path = "temp.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")

            result_index = model_prediction(temp_path)

            # Labels
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

            disease = class_name[result_index]
            st.success(f"Model is predicting it's **{disease}** 🌿")

            # -----------------------
            # ADVISORY CARD (PREMIUM)
            # -----------------------
            st.markdown("""
            <div style='padding:20px; border-radius:18px; background:#f5faff;
                        box-shadow:0 4px 12px rgba(0,0,0,0.1); font-family:Poppins;'>

                <h2 style='color:#2b6a4b; text-align:center;'>🌱 Auto-Fertilizer Recommendation</h2>

                <div style='margin-top:15px;'>
                    <h3 style='color:#d35400;'>🍅 Disease: Tomato Early Blight</h3>
                </div>

                <div style='background:white; padding:15px; border-radius:12px; margin-top:10px;'>
                    <h4>🧪 Fungicide Treatment</h4>
                    <p><b>Mancozeb 75% WP</b><br>
                    • 2g per liter<br>
                    • Apply every 7 days<br>
                    • Continue 2–3 cycles</p>
                </div>

                <div style='background:white; padding:15px; border-radius:12px; margin-top:10px;'>
                    <h4>🌿 Nutrient Booster</h4>
                    <p><b>NPK 19:19:19</b><br>
                    • 5g per liter<br>
                    • Use 3 days after fungicide</p>
                </div>

                <div style='background:white; padding:15px; border-radius:12px; margin-top:10px;'>
                    <h4>🌱 Soil Reviver</h4>
                    <p><b>Trichoderma viride</b><br>
                    • 5 kg/acre<br>
                    • Mix with FYM</p>
                </div>

                <div style='background:#e8f8f5; padding:15px; border-radius:12px; margin-top:10px;'>
                    <h4>🌤️ Weather Precautions</h4>
                    <ul>
                        <li>Avoid spraying in rain/wind</li>
                        <li>Humidity > 80% = High risk</li>
                        <li>Spray early morning / evening</li>
                    </ul>
                </div>

                <div style='margin-top:20px; padding:15px; background:#fff3cd; border-radius:12px;'>
                    <h4>🗓 Next Spray Reminder</h4>
                    After <b>7 days</b>.
                </div>

            </div>
            """, unsafe_allow_html=True)

# --------------------------
# FOOTER
# --------------------------
st.markdown("<p style='text-align:center; margin-top:40px; color:gray;'>✨ Supported by AgriNext Team ✨</p>", unsafe_allow_html=True)
