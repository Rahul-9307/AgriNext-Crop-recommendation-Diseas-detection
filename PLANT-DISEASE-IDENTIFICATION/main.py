import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# ------------------------------
# Load model (only once)
# ------------------------------
@st.experimental_singleton
def load_model():
    current_dir = os.path.dirname(__file__)

    model_path = os.path.join(current_dir, "trained_plant_disease_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(current_dir, "trained_plant_disease_model.h5")

    if not os.path.exists(model_path):
        return None

    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()


# ------------------------------
# Prediction function
# ------------------------------
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    result = model.predict(array)
    return np.argmax(result)


# ------------------------------
# Page Design
# ------------------------------
st.set_page_config(page_title="AgriSens Disease Prediction", layout="centered")

# Sidebar
st.sidebar.title("🌿 AgriSens")
page = st.sidebar.radio("Go to", ["Home", "Disease Recognition"])

# Header
st.markdown("<h1 style='text-align:center; font-size:40px; color:#2E8B57;'>🌾 SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)


# Home Page
if page == "Home":
    st.markdown("""
        <div style='text-align:center'>
            <p style='font-size:18px;'>
                Welcome to **AgriSens Crop Disease Detection App**!  
                Detect plant disease using AI — just upload an image.
            </p>
        </div>
    """, unsafe_allow_html=True)


# Disease Recognition Page
elif page == "Disease Recognition":

    st.markdown("<h2 style='text-align:center; color:#6A5ACD;'>📸 Upload Plant Leaf Image</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display image
        st.image(uploaded_file, caption="Selected Image", use_column_width=True)

        if st.button("🔍 Predict Disease"):
            if model is None:
                st.error("❌ Model not found! Please upload model file in repo.")
            else:
                # Save temp
                temp_path = os.path.join(os.getcwd(), "temp_image.jpg")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Predict
                result = predict_image(temp_path)

                classes = [
                    'Apple___Apple_scab', 'Apple___Black_rot',
                    'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                    'Corn_(maize)___healthy',
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)',
                    'Peach___Bacterial_spot', 'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                    'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                    'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot',
                    'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                ]

                st.markdown(f"<h3 style='text-align:center; color:#228B22;'>✅ Prediction: {classes[result]}</h3>", unsafe_allow_html=True)
