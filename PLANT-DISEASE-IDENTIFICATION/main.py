import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Agri 🌿 Next", layout="centered")

st.title("🌿 AgriNext - Plant Disease Identification")
st.write("Upload a leaf image to detect plant disease.")

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    model_path = "trained_plant_disease_model.keras"

    if not os.path.exists(model_path):
        st.error("Model file not found! Make sure 'trained_plant_disease_model.keras' exists.")
        return None

    model = tf.keras.models.load_model(model_path)
    return model


model = load_model()

# ---------------- CLASS LABELS ----------------
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):
    target_size = (128, 128)

    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model is not None:
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"🌱 Predicted Disease: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Model not loaded properly.")                   main.py
