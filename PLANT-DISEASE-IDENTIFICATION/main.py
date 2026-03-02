import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AgriNext 🌾 Disease Detection", layout="wide")

# -------------------------------
# LOAD MODEL ONLY ONCE (IMPORTANT)
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

model = load_model()

# -------------------------------
# CLASS LABELS
# -------------------------------
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# -------------------------------
# IMAGE PREDICTION FUNCTION
# -------------------------------
def model_prediction(test_image):
    image = Image.open(test_image).convert("RGB")
    image = image.resize((128, 128))

    input_arr = np.array(image) / 255.0  # Normalization (VERY IMPORTANT)
    input_arr = np.expand_dims(input_arr, axis=0)

    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)
    confidence = np.max(predictions)

    return result_index, confidence

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🌾 AgriNext")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# -------------------------------
# HOME PAGE
# -------------------------------
if app_mode == "HOME":
    st.title("🌿 SMART PLANT DISEASE DETECTION SYSTEM")
    st.write("Upload a leaf image to detect plant disease using Deep Learning.")
    st.image("Diseases.png", use_column_width=True)

# -------------------------------
# DISEASE RECOGNITION PAGE
# -------------------------------
elif app_mode == "DISEASE RECOGNITION":

    st.header("🔍 Plant Disease Recognition")

    test_image = st.file_uploader("📤 Upload a Leaf Image", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("🚀 Predict Disease"):
            with st.spinner("Model is analyzing..."):
                result_index, confidence = model_prediction(test_image)

                predicted_label = class_name[result_index]

                st.success(f"🌱 Predicted Disease: **{predicted_label}**")
                st.info(f"📊 Confidence: **{round(confidence * 100, 2)}%**")

                if confidence < 0.50:
                    st.warning("⚠️ Low confidence prediction. Please upload a clearer leaf image.")
