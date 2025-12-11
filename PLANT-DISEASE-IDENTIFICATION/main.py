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
# GITHUB RAW IMAGES (WORKS 100% ON STREAMLIT CLOUD)
# -----------------------------------------------------------
HERO_IMAGE = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Diseases.png"

IMG_REALTIME = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Real-Time%20Results.png"
IMG_INSIGHTS = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Actionable%20Insights.png"
IMG_DETECTION = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Disease%20Detection.png"

# -----------------------------------------------------------
# HERO CSS
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
.center-text {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# HERO IMAGE DISPLAY
# -----------------------------------------------------------
st.markdown("<div class='hero-box'>", unsafe_allow_html=True)
st.image(HERO_IMAGE, use_column_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.write("")

# -----------------------------------------------------------
# PAGE SELECTOR (CENTER DROPDOWN)
# -----------------------------------------------------------
cols = st.columns(3)
with cols[1]:
    page = st.selectbox("Select a Page", ["HOME", "DISEASE RECOGNITION"])

# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
if page == "HOME":

    st.markdown("""
    <h1 class='center-text' style='color:#2ecc71; font-weight:800; margin-top:20px;'>
        AgriSens: Smart Disease Detection
    </h1>
    <p class='center-text' style='color:#ccc; font-size:18px;'>
        Empowering farmers with AI-powered plant disease recognition.<br>
        Upload leaf images to detect diseases accurately and access actionable insights.
    </p>
    """, unsafe_allow_html=True)

    st.write("## Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(IMG_REALTIME, use_column_width=True)
        st.markdown("<p class='center-text'><b>Real-Time Results</b></p>", unsafe_allow_html=True)
        st.write("Instant predictions using powerful AI models.")

    with col2:
        st.image(IMG_INSIGHTS, use_column_width=True)
        st.markdown("<p class='center-text'><b>Actionable Insights</b></p>", unsafe_allow_html=True)
        st.write("Know disease details and recommended remedies.")

    with col3:
        st.image(IMG_DETECTION, use_column_width=True)
        st.markdown("<p class='center-text'><b>Disease Detection</b></p>", unsafe_allow_html=True)
        st.write("Detect plant diseases with a single image.")

    st.write("## How It Works")
    st.markdown("""
    1. Go to the **Disease Recognition** page.<br>
    2. Upload a leaf image.<br>
    3. Get instant detection and suggested actions.<br>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------
# DISEASE RECOGNITION PAGE
# -----------------------------------------------------------
elif page == "DISEASE RECOGNITION":

    st.markdown("""
    <h1 class='center-text' style='color:#2ecc71;'>🌿 Disease Recognition</h1>
    <p class='center-text' style='color:#bbb;'>Upload a plant leaf image to detect disease using AI.</p>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def load_model():
        model_path = "trained_plant_disease_model.keras"
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path)
        st.error("❌ Model file not found!")
        return None

    model = load_model()

    def predict_image(path):
        img = tf.keras.preprocessing.image.load_img(path, target_size=(128, 128))
        arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), 0)
        pred = model.predict(arr)
        return np.argmax(pred)

    uploaded = st.file_uploader("📸 Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded:

        st.image(uploaded, use_column_width=True)

        temp_path = "uploaded_temp.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if st.button("🔍 Detect Disease"):
            st.info("⏳ Processing...")

            if model:
                idx = predict_image(temp_path)
                st.success(f"🌱 Predicted Disease Class Index: **{idx}**")
            else:
                st.error("❌ Model not loaded!")

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("""
<div style='background:#111; padding:15px; border-radius:10px; margin-top:40px; color:white; text-align:center;'>
Developed by <b>Team AgriSens</b> | Powered by Streamlit
</div>
""", unsafe_allow_html=True)
