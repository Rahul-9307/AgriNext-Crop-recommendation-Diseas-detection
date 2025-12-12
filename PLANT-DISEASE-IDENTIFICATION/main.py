import streamlit as st
import os
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Agri🌾Next", layout="centered")

# -----------------------------------------------------------
# RAW IMAGE LINKS
# -----------------------------------------------------------
HERO_IMAGE = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Diseases.png"

IMG_REALTIME = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Real-Time%20Results.png"
IMG_INSIGHTS = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Actionable%20Insights.png"
IMG_DETECTION = "https://raw.githubusercontent.com/Rahul-9307/AgriNextCROP-RECOMMENDATION/main/PLANT-DISEASE-IDENTIFICATION/Disease%20Detection.png"

# -----------------------------------------------------------
# ULTRA PREMIUM CSS (Glassmorphism + Premium Green Theme)
# -----------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Main container width */
.block-container {
    max-width: 900px;
    padding-top: 10px;
}

/* HERO IMAGE */
.hero-wrapper {
    display: flex;
    justify-content: center;
}
.hero-img {
    margin-top: 45px;
    width: 100%;
    border-radius: 18px;
    border: 2px solid #19c37d;
    box-shadow: 0 0 18px rgba(25,195,125,0.3);
}

/* Premium green heading */
h1, h2, h3 {
    color: #19c37d !important;
    font-weight: 700;
}

/* Glass Card */
.card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 18px;
    border: 1px solid rgba(25,195,125,0.4);
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    transition: 0.25s ease;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 25px rgba(25,195,125,0.45);
}

/* Small centered text */
.center-text { text-align: center; color: #ccc; }

/* Buttons */
.stButton>button {
    background: #19c37d;
    color: white;
    border-radius: 10px;
    padding: 10px 22px;
    border: none;
    font-size: 16px;
    transition: 0.25s;
    font-weight: 600;
}
.stButton>button:hover {
    background: #17a86c;
    transform: scale(1.05);
}

/* Footer */
.app-footer {
    background:#0e0e0e;
    padding:12px;
    border-radius:10px;
    margin-top:25px;
    text-align:center;
    font-size:13px;
    color:white;
    border:1px solid rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# HERO IMAGE
# -----------------------------------------------------------
st.markdown(f"""
<div class='hero-wrapper'>
    <img src='{HERO_IMAGE}' class='hero-img'>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# PAGE SELECTOR
# -----------------------------------------------------------
cols = st.columns([1,2,1])
with cols[1]:
    page = st.selectbox("Select a Page", ["HOME", "DISEASE RECOGNITION"])

# -----------------------------------------------------------
# CLASS LABELS
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
# AUTO MODEL LOADER
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    target_name = "trained_plant_disease_model.keras"
    found_path = None

    for root, dirs, files in os.walk(".", topdown=True):
        if target_name in files:
            found_path = os.path.join(root, target_name)
            break

    st.markdown(
        "<p class='center-text' style='font-size:14px;'>🔍 Loading AI Model...</p>",
        unsafe_allow_html=True)

    if found_path:
        st.success(f"✅ Model Found: {found_path}")
        return tf.keras.models.load_model(found_path)

    st.error("❌ Model NOT FOUND! Upload trained_plant_disease_model.keras.")
    return None

model = load_model()

# -----------------------------------------------------------
# PREDICT FUNCTION
# -----------------------------------------------------------
def predict_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(128,128))
    arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img)/255.0, 0)
    pred = model.predict(arr)
    idx = np.argmax(pred)
    conf = np.max(pred)
    return idx, CLASS_NAMES[idx], float(conf)

# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
if page == "HOME":

    st.markdown("<h1 style='text-align:center;'>Agri🌾Next — Smart Disease Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='center-text' style='font-size:16px;'>AI-powered plant health monitoring platform</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(IMG_REALTIME, width=220)
        st.markdown("<p class='center-text'><b>Real-Time Results</b></p></div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(IMG_INSIGHTS, width=220)
        st.markdown("<p class='center-text'><b>Actionable Insights</b></p></div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(IMG_DETECTION, width=220)
        st.markdown("<p class='center-text'><b>Disease Detection</b></p></div>", unsafe_allow_html=True)

    st.markdown("""
    <h2 style='text-align:center; margin-top:35px;'>How It Works 🔍</h2>
    <div style='max-width:700px; margin:auto; font-size:17px; line-height:1.6; color:#ccc;'>
        <ol>
            <li>Go to the <b>Disease Recognition</b> page</li>
            <li>Upload your plant leaf image</li>
            <li>Get instant disease prediction with accuracy</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------
# DISEASE RECOGNITION PAGE
# -----------------------------------------------------------
elif page == "DISEASE RECOGNITION":
    st.markdown("<h2 class='center-text'>🌿 Disease Recognition</h2>", unsafe_allow_html=True)
    st.markdown("<p class='center-text'>Upload a plant leaf image to detect disease</p>", unsafe_allow_html=True)

    col_up1, col_up2, col_up3 = st.columns([1,2,1])
    with col_up2:
        uploaded = st.file_uploader("", type=["jpg","jpeg","png"])

    if uploaded:
        img_left, img_center, img_right = st.columns([1,2,1])
        with img_center:
            st.image(uploaded, width=420)

        temp_path = "uploaded_temp.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        btn_left, btn_center, btn_right = st.columns([1,1,1])
        with btn_center:
            detect = st.button("🔍 Detect Disease")

        if detect:
            if model is None:
                st.error("❌ Model not loaded!")
            else:
                st.info("🔮 Predict by AgriNext Team")
                idx, disease, conf = predict_image(temp_path)

                st.markdown(f"""
                <div class='card' style='max-width:450px; margin:auto; text-align:center;'>
                    <h3>🌱 Predicted: <b>{disease}</b></h3>
                    <p style='color:#ccc;'>Confidence: <b>{conf*100:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("<div class='app-footer'>Developed by <b>Team Agri🌾Next</b> | Powered by Streamlit</div>", unsafe_allow_html=True)
