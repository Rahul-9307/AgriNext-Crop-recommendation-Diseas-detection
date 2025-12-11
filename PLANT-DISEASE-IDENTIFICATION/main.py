import streamlit as st
import os
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="AgriSens", layout="wide")

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
page = st.sidebar.selectbox("📌 Select a Page", ["🏠 Home", "🌿 Disease Recognition"])

# -----------------------------------------------------------
# HERO CSS + FEATURE CSS
# -----------------------------------------------------------
st.markdown("""
<style>
.hero-box {
    width: 100%;
    border-radius: 18px;
    overflow: hidden;
    border: 2px solid #2ecc71;
    margin-top: 10px;
    box-shadow: 0px 0px 15px rgba(0,255,150,0.25);
}
.center-text { text-align:center; }
.feature-card img {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# PAGE 1: HOME
# -----------------------------------------------------------
if page == "🏠 Home":

    # Hero Banner Image
    diseases_img = os.path.join(os.path.dirname(__file__), "Diseases.png")

    if os.path.exists(diseases_img):
        st.markdown("<div class='hero-box'>", unsafe_allow_html=True)
        st.image(diseases_img, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠ Diseases.png not found!")

    # Title + Subtitle
    st.markdown("""
    <h1 class='center-text' style='color:#2ecc71; font-weight:800; margin-top:20px;'>
        AgriSens: Smart Disease Detection
    </h1>

    <p class='center-text' style='color:#ccc; font-size:18px;'>
        Empowering farmers with AI-powered plant disease recognition.<br>
        Upload leaf images to detect diseases and receive actionable insights.
    </p>
    """, unsafe_allow_html=True)

    st.write("## Features")

    # Feature Card Images
    img1 = "img_realtime.png"
    img2 = "img_insights.png"
    img3 = "img_detection.png"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img1, use_column_width=True)
        st.markdown("<p class='center-text'><b>Real-Time Results</b></p>", unsafe_allow_html=True)
        st.write("Receive instant predictions.")

    with col2:
        st.image(img2, use_column_width=True)
        st.markdown("<p class='center-text'><b>Actionable Insights</b></p>", unsafe_allow_html=True)
        st.write("Get detailed insights and remedies.")

    with col3:
        st.image(img3, use_column_width=True)
        st.markdown("<p class='center-text'><b>Disease Detection</b></p>", unsafe_allow_html=True)
        st.write("Identify plant diseases with AI.")

    st.write("## How It Works")
    st.markdown("""
    1. Go to the **Disease Recognition** page.<br>
    2. Upload an image of the affected leaf.<br>
    3. Instantly receive disease detection and solution advice.
    """, unsafe_allow_html=True)


# -----------------------------------------------------------
# PAGE 2: DISEASE RECOGNITION
# -----------------------------------------------------------
else:

    st.markdown("""
    <h1 class='center-text' style='color:#2ecc71;'>🌿 Disease Recognition</h1>
    <p class='center-text' style='color:#bbb;'>Upload a plant leaf image to detect disease using AI.</p>
    """, unsafe_allow_html=True)

    # -------------------------
    # MODEL LOADER
    # -------------------------
    @st.cache_resource
    def load_model():
        model_name = "trained_plant_disease_model.keras"
        for root, dirs, files in os.walk(".", topdown=True):
            if model_name in files:
                return tf.keras.models.load_model(os.path.join(root, model_name))
        st.error("❌ Model not found!")
        return None

    model = load_model()

    # -------------------------
    # PREDICT FUNCTION
    # -------------------------
    def predict_image(path):
        img = tf.keras.preprocessing.image.load_img(path, target_size=(128, 128))
        arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), 0)
        pred = model.predict(arr)
        return np.argmax(pred)

    uploaded = st.file_uploader("📸 Upload leaf image", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.image(uploaded, use_column_width=True)

        temp_path = "uploaded_temp.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if st.button("🔍 Detect Disease"):
            loader = st.empty()
            loader.markdown("<center><img src='https://i.gifer.com/ZZ5H.gif' width='120'></center>", unsafe_allow_html=True)

            if model:
                idx = predict_image(temp_path)
                loader.empty()
                st.success(f"🌱 Predicted Class Index: **{idx}**")
            else:
                loader.empty()
                st.error("❌ Model not loaded!")

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("""
<div style='background:#111; padding:15px; border-radius:10px; margin-top:40px; color:white; text-align:center;'>
Developed by <b>Team AgriSens</b> | Powered by Streamlit
</div>
""", unsafe_allow_html=True)
