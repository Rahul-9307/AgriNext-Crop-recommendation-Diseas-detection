import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="AgriNext 🌾 Disease Detection", layout="wide")

# ==============================
# LOAD MODEL (Load Only Once)
# ==============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    return model

model = load_model()

# ==============================
# Prediction Function
# ==============================
def model_prediction(test_image):
    image = Image.open(test_image)
    image = image.resize((128, 128))
    input_arr = np.array(image)
    
    # Normalize image (VERY IMPORTANT)
    input_arr = input_arr / 255.0
    
    # Convert to batch
    input_arr = np.expand_dims(input_arr, axis=0)
    
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# ==============================
# Sidebar
# ==============================
st.sidebar.title("🌾 AgriNext")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# ==============================
# Home Page
# ==============================
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART PLANT DISEASE DETECTION SYSTEM 🌱</h1>", unsafe_allow_html=True)
    st.write("Upload a plant leaf image and detect disease instantly using AI.")

# ==============================
# Disease Recognition Page
# ==============================
elif app_mode == "DISEASE RECOGNITION":
    st.header("🔍 DISEASE RECOGNITION")

    test_image = st.file_uploader("📸 Upload Leaf Image", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("🚀 Predict"):
            st.snow()
            result_index = model_prediction(test_image)

            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 
                'Corn_(maize)___healthy', 'Grape___Black_rot', 
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 
                'Peach___Bacterial_spot', 'Peach___healthy', 
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 
                'Potato___healthy', 'Raspberry___healthy', 
                'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
                'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 
                'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
                'Tomato___Tomato_mosaic_virus', 
                'Tomato___healthy'
            ]

            st.success(f"🌿 Model Prediction: {class_name[result_index]}")

    else:
        st.warning("⚠ Please upload an image first.")
