import streamlit as st
import os
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Agri🌾Next", layout="wide")

# -----------------------------------------------------------
# RAW IMAGE LINKS
# -----------------------------------------------------------
HERO_IMAGE = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Diseases.png"

IMG_REALTIME = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Real-Time%20Results.png"
IMG_INSIGHTS = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Actionable%20Insights.png"
IMG_DETECTION = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Disease%20Detection.png"


# -----------------------------------------------------------
# GLOBAL UI CSS — Everything Center + Smaller Layout
# -----------------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: "Poppins", sans-serif;
}

.hero-box {
    width: 80%;
    margin-left: auto;
    margin-right: auto;
    border-radius: 18px;
    overflow: hidden;
    border: 2px solid #2ecc71;
    margin-top: 20px;
    box-shadow: 0px 0px 15px rgba(0,255,150,0.25);
}

.center-text { 
    text-align:center; 
}

.small-img img {
    width: 90% !important;
    margin: auto;
    display: block;
}

.selectbox-center > div {
    margin-left: auto;
    margin-right: auto;
    width: 40%;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# HERO IMAGE
# -----------------------------------------------------------
st.markdown("<div class='hero-box'>", unsafe_allow_html=True)
st.image(HERO_IMAGE, use_column_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------
# PAGE SELECTOR (Centered)
# -----------------------------------------------------------
st.markdown("<div class='selectbox-center'>", unsafe_allow_html=True)
page = st.selectbox("Select a Page", ["HOME", "DISEASE RECOGNITION"])
st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------
# CLASS LABELS — MUST MATCH TRAINING
# -----------------------------------------------------------
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper_bell___Bacterial_spot", "Pepper_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]


# -----------------------------------------------------------
# MODEL LOADER — SAFE LOAD
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    target_name = "trained_plant_disease_model.keras"
    found_path = None

    for root, dirs, files in os.walk(".", topdown=True):
        if target_name in files:
            found_path = os.path.join(root, target_name)
            break

    if found_path:
        st.success(f"✅ Model Loaded: {found_path}")
        return tf.keras.models.load_model(found_path)

    st.error("❌ Model NOT FOUND — Upload trained_plant_disease_model.keras")
    return None


model = load_model()


# -----------------------------------------------------------
# PREDICT FUNCTION
# -----------------------------------------------------------
def predict_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(128, 128))
    arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img) / 255.0, 0)

    pred = model.predict(arr)
    idx = np.argmax(pred)
    conf = np.max(pred)

    return idx, CLASS_NAMES[idx], float(conf)


# -----------------------------------------------------------
# HOME PAGE UI
# -----------------------------------------------------------
if page == "HOME":

    st.markdown("""
    <h1 class='center-text' style='color:#2ecc71; font-weight:800;'>Agri🌾Next: Smart Disease Detection</h1>
    <p class='center-text' style='color:#ccc; font-size:18px;'>
        AI-powered platform for accurate plant disease recognition.
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("<div class='small-img'>", unsafe_allow_html=True)
        st.image(IMG_REALTIME)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<p class='center-text'><b>Real-Time Results</b></p>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='small-img'>", unsafe_allow_html=True)
        st.image(IMG_INSIGHTS)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<p class='center-text'><b>Actionable Insights</b></p>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='small-img'>", unsafe_allow_html=True)
        st.image(IMG_DETECTION)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<p class='center-text'><b>Disease Detection</b></p>", unsafe_allow_html=True)


# -----------------------------------------------------------
# DISEASE RECOGNITION
# -----------------------------------------------------------
elif page == "DISEASE RECOGNITION":

    st.markdown("""
    <h1 class='center-text' style='color:#2ecc71;'>🌿 Disease Recognition</h1>
    <p class='center-text' style='color:#bbb;'>Upload a plant leaf image to detect disease.</p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("📸 Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.image(uploaded, use_column_width=True)

        temp_path = "uploaded_temp.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if st.button("🔍 Detect Disease"):

            if model is None:
                st.error("❌ Model not loaded!")
            else:
                st.info("🔮 Predict by AgriNext Team")
                idx, disease, conf = predict_image(temp_path)

                st.success(f"🌱 Predicted Disease: **{disease}**")
                st.info(f"📊 Confidence: **{conf*100:.2f}%**")


# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("""
<div style='background:#111; padding:15px; border-radius:10px; margin-top:40px; color:white; text-align:center;'>
Developed by <b>Team Agri🌾Next</b> | Powered by Streamlit
</div>
""", unsafe_allow_html=True)
