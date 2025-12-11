import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image


# -------------------- MODEL PREDICTION FUNCTION --------------------
def model_prediction(test_image_path):

    current_dir = os.path.dirname(__file__)

    # Try .keras model
    model_path = os.path.join(current_dir, "trained_plant_disease_model.keras")

    # If .keras not found → try .h5
    if not os.path.exists(model_path):
        model_path = os.path.join(current_dir, "trained_plant_disease_model.h5")

    # Still not found → Stop app
    if not os.path.exists(model_path):
        st.error("❌ Model file not found!\n\n"
                 "Please upload **trained_plant_disease_model.keras** or **trained_plant_disease_model.h5**\n"
                 "in the same folder as main.py.")
        st.stop()

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Preprocess image
    image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

    # Predict
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


# -------------------- SIDEBAR --------------------
st.sidebar.title("AgriSens")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])


# -------------------- LOAD MAIN PAGE IMAGE --------------------
current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, "Diseases.png")

if os.path.exists(image_path):
    img = Image.open(image_path)
    st.image(img, use_column_width=True)
else:
    st.warning("⚠️ Diseases.png not found. Add it in the main.py folder.")


# -------------------- HOME PAGE --------------------
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)


# -------------------- DISEASE RECOGNITION PAGE --------------------
elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:

        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.snow()

            # Save uploaded image temporarily
            temp_path = os.path.join(current_dir, "temp.jpg")
            with open(temp_path, "wb") as f:
                f.write(test_image.getbuffer())

            # Predict
            result_index = model_prediction(temp_path)

            # Class labels
            class_name = [
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

            st.success(f"Prediction: **{class_name[result_index]}**")
