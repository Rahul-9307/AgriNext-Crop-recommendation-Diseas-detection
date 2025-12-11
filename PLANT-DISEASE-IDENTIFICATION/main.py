import streamlit as st
import tensorflow as tf
import numpy as np
import os
import zipfile
from PIL import Image

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Agri🌾Next – Disease Detection", layout="wide")


# -----------------------------------------------------------
# UNZIP MODEL IF NOT EXTRACTED
# -----------------------------------------------------------
MODEL_NAME = "trained_plant_disease_model.keras"
ZIP_NAME = "model.zip"

def extract_model_if_needed():
    """Extract model.zip → trained_plant_disease_model.keras"""
    if os.path.exists(MODEL_NAME):
        return True

    if os.path.exists(ZIP_NAME):
        st.info("📦 Extracting model.zip ... Please wait...")
        try:
            with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
                zip_ref.extractall(".")
            st.success("✅ Model extracted successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error extracting ZIP: {e}")
            return False

    st.error("❌ Model ZIP not found! Please upload model.zip with trained_plant_disease_model.keras inside.")
    return False


# -----------------------------------------------------------
# LOAD MODEL FUNCTION
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the extracted Keras model safely."""
    if not extract_model_if_needed():
        return None

    if os.path.exists(MODEL_NAME):
        st.success("✅ Model Loaded Successfully!")
        return tf.keras.models.load_model(MODEL_NAME)

    st.error("❌ Model file missing even after extraction!")
    return None


model = load_model()


# -----------------------------------------------------------
# PREDICT FUNCTION
# -----------------------------------------------------------
def predict_image(image_file):
    temp_path = "temp_img.jpg"

    with open(temp_path, "wb") as f:
        f.write(image_file.getbuffer())

    img = tf.keras.preprocessing.image.load_img(temp_path, target_size=(128, 128))
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)
    index = np.argmax(pred)
    confidence = float(np.max(pred))

    return index, confidence


# -----------------------------------------------------------
# CLASS LABELS
# -----------------------------------------------------------
CLASS_NAMES = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry___Powdery_mildew","Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot","Corn___Common_rust",
    "Corn___Northern_Leaf_Blight","Corn___healthy",
    "Grape___Black_rot","Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
    "Pepper_bell___Bacterial_spot","Pepper_bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Raspberry___healthy","Soybean___healthy",
    "Squash___Powdery_mildew","Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites","Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("Agri🌾Next")
page = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])


# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
if page == "HOME":
    st.markdown("<h1 style='text-align:center;color:#2ecc71;'>Agri🌾Next – Smart Disease Detection</h1>", 
                unsafe_allow_html=True)

    st.write("## How It Works")
    st.write("""
    1️⃣ Upload a plant leaf image  
    2️⃣ Our AI model identifies the disease  
    3️⃣ You get instant results with confidence score  
    """)


# -----------------------------------------------------------
# DISEASE RECOGNITION PAGE
# -----------------------------------------------------------
elif page == "DISEASE RECOGNITION":

    st.header("🌿 Disease Recognition")

    uploaded = st.file_uploader("📸 Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.image(uploaded, use_column_width=True)

        if st.button("🔍 Predict Disease"):
            if model is None:
                st.error("❌ Model not loaded!")
            else:
                st.info("⏳ Analyzing the plant leaf...")

                idx, conf = predict_image(uploaded)
                disease = CLASS_NAMES[idx]

                st.success(f"🌱 Predicted Disease: **{disease}**")
                st.info(f"📊 Confidence: **{conf*100:.2f}%**")


# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("""
<br><hr>
<div style='text-align:center;color:gray;'>
Developed by <b>Agri🌾Next Team</b>
</div>
""", unsafe_allow_html=True)
