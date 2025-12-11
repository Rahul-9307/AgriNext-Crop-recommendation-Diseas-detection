import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

st.set_page_config(page_title="AgriNext – Plant Disease Detection", layout="centered")

# -----------------------------------------------------------
# CUSTOM CSS (FIXED + CLEAN)
# -----------------------------------------------------------
st.markdown("""
<style>

body {
    background-color: #0f1117;
}

h1, h2, h3, h4 {
    text-align:center;
    font-family:'Poppins', sans-serif;
    color:white;
}

/* Upload Box */
.upload-box {
    border: 2px dashed #6A5ACD;
    padding: 20px;
    border-radius: 15px;
    text-align:center;
}

/* Gradient Button */
.gradient-btn {
    background: linear-gradient(90deg, #6A5ACD, #00B4D8);
    color: white !important;
    padding: 14px;
    border-radius: 12px;
    border:none;
    width:100%;
    font-size:18px;
    cursor:pointer;
}

/* Prediction Result Card */
.result-card {
    background: #1c1f25;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    margin-top:20px;
    color: white;
    text-align:center;
}

/* Footer */
.footer-card {
    background:#1c1c1c;
    padding:30px;
    border-radius:18px;
    margin-top:60px;
    color:white;
    font-family:'Poppins', sans-serif;
    box-shadow:0 4px 15px rgba(0,0,0,0.5);
}

.footer-title {
    text-align:center;
    font-size:28px;
    font-weight:700;
    color:#A259FF;
    margin-bottom:10px;
}

.footer-text {
    font-size:18px;
    line-height:1.6;
}

.footer-bullets {
    font-size:18px;
    margin-top:12px;
}

.team-label {
    font-size:20px;
    font-weight:600;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------
@st.cache_resource
def load_model():

    current_dir = os.path.dirname(__file__)
    keras_path = os.path.join(current_dir, "trained_plant_disease_model.keras")
    h5_path = os.path.join(current_dir, "trained_plant_disease_model.h5")

    if os.path.exists(keras_path):
        return tf.keras.models.load_model(keras_path)
    if os.path.exists(h5_path):
        return tf.keras.models.load_model(h5_path)
    return None

model = load_model()


# -----------------------------------------------------------
# PREDICTION FUNCTION
# -----------------------------------------------------------
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128,128))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    result = model.predict(arr)
    return np.argmax(result)


# -----------------------------------------------------------
# HEADER
# -----------------------------------------------------------
st.markdown("<h1 style='color:#A259FF;'>🌾 AgriNext – स्मार्ट रोग निदान</h1>", unsafe_allow_html=True)
st.write("")


# -----------------------------------------------------------
# IMAGE UPLOAD
# -----------------------------------------------------------
uploaded = st.file_uploader("📸 पानाचा फोटो अपलोड करा", type=["jpg", "jpeg", "png"])

if uploaded:

    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    st.image(uploaded, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    if st.button("🔍 रोग ओळखा", key="predict", help="Click to Predict"):
        
        # Loader
        st.markdown("<center><img src='https://i.gifer.com/ZZ5H.gif' width='120'></center>", unsafe_allow_html=True)

        if model is None:
            st.error("❌ मॉडेल फाइल मिळाली नाही!")

        else:
            idx = predict_image(temp_path)

            class_name = [
                'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
                'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
                'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
                'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
                'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
                'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot',
                'Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato___Tomato_mosaic_virus','Tomato___healthy'
            ]

            # RESULT CARD
            st.markdown(f"""
            <div class='result-card'>
                <h3>🌱 ओळखलेला रोग</h3>
                <h2 style='color:#32CD32;'>✔️ {class_name[idx]}</h2>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------------------------------
# FOOTER (FIXED, CLEAN VERSION)
# -----------------------------------------------------------
st.markdown("""
<div class='footer-card'>

    <div class='footer-title'>👥 AgriNext Team</div>

    <div class='footer-text'>
        AgriNext हे शेतकऱ्यांसाठी अत्याधुनिक तंत्रज्ञान वापरून विकसित केलेले बुद्धिमान प्लॅटफॉर्म आहे.
        आमचे ध्येय — <strong>“प्रत्येक शेतकऱ्याला स्मार्ट शेतीची सुविधा मिळावी.”</strong>
    </div>

    <div class='footer-bullets'>
        🔹 AI आधारित रोग निदान <br>
        🔹 पिक सल्ला <br>
        🔹 स्थानिक भाषेत मार्गदर्शन <br>
        🔹 शेत पातळीवरील निर्णय सहाय्य <br>
    </div>

    <div class='team-label'>टीम:</div>
    <div class='footer-text'>
        • Rahul Patil (Developer) <br>
        • AgriNext Research & Advisory Team
    </div>

</div>
""", unsafe_allow_html=True)
