import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# -----------------------------
# SAFE IMAGE LOADING
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_local_image(filename):
    return Image.open(os.path.join(BASE_DIR, filename))

# -----------------------------
# MODEL PREDICTION FUNCTION
# -----------------------------
def model_prediction(test_image_path):
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, "trained_plant_disease_model.keras"))
    image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.sidebar.title("AgriSens")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Load Diseases.png safely
try:
    img = load_local_image("Diseases.png")
    st.image(img)
except:
    st.warning("⚠️ ‘Diseases.png’ file not found! Please check your project folder.")

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")

    uploaded_file = st.file_uploader("Choose an Image:")

    if uploaded_file:
        st.image(uploaded_file, width=300)

        # Save uploaded file temporarily
        temp_file_path = os.path.join(BASE_DIR, "temp.jpg")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")

            # Predict
            result_index = model_prediction(temp_file_path)

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

            st.success(f"Model is predicting it's **{class_name[result_index]}** 🌿")
