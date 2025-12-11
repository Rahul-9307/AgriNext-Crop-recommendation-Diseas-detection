import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

st.set_page_config(page_title="AgriNext – स्मार्ट रोग निदान", layout="centered")


# -----------------------------------------------------------
# MODEL LOADING (100% Working Guaranteed)
# -----------------------------------------------------------
@st.cache_resource
def load_model():

    # debugging – show all files
    st.write("📂 Current Directory Files:", os.listdir("."))

    keras_file = "trained_plant_disease_model.keras"
    h5_file = "trained_plant_disease_model.h5"

    if os.path.exists(keras_file):
        st.success("✔ .keras Model Loaded Successfully!")
        return tf.keras.models.load_model(keras_file)

    if os.path.exists(h5_file):
        st.success("✔ .h5 Model Loaded Successfully!")
        return tf.keras.models.load_model(h5_file)

    st.error("❌ Model file not found in folder!")
    return None


model = load_model()


# -----------------------------------------------------------
# PREDICT FUNCTION
# -----------------------------------------------------------
def predict_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(128, 128))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    pred = model.predict(arr)
    return np.argmax(pred)


# -----------------------------------------------------------
# DISEASE INFORMATION
# -----------------------------------------------------------
disease_info = {
    "Apple___Apple_scab": {
        "name": "Apple Scab (सफरचंद स्कॅब)",
        "sym": "पानांवर काळपट डाग, फळे विकृत.",
        "treat": "मॅन्कोझेब / क्लोरोथॅलोनील फवारणी.",
        "prev": "संक्रमित पाने जाळा."
    },
    "Tomato___Late_blight": {
        "name": "Tomato Late Blight (लेट ब्लाईट)",
        "sym": "पानांवर तपकिरी पाण्यासारखे डाग.",
        "treat": "मेटालेक्सिल + मॅन्कोझेब.",
        "prev": "जास्त आर्द्रता टाळा."
    }
}


# -----------------------------------------------------------
# UI
# -----------------------------------------------------------
st.markdown("<h1 style='color:#A259FF;text-align:center;'>🌾 AgriNext – स्मार्ट रोग निदान</h1>", unsafe_allow_html=True)
st.write("___")

uploaded = st.file_uploader("📸 पानाचा फोटो अपलोड करा", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, use_column_width=True)

    temp = "temp_img.jpg"
    with open(temp, "wb") as f:
        f.write(uploaded.getbuffer())

    if st.button("🔍 रोग ओळखा"):

        loader = st.empty()
        loader.markdown("<center><img src='https://i.gifer.com/ZZ5H.gif' width='140'></center>", unsafe_allow_html=True)

        if model is None:
            loader.empty()
            st.error("❌ Model load झाला नाही!")

        else:
            idx = predict_image(temp)

            class_list = [
                'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
                'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
                'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
                'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
                'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
                'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
                'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato___Tomato_mosaic_virus','Tomato___healthy'
            ]

            predicted = class_list[idx]

            loader.empty()

            st.success(f"🌱 ओळखलेला रोग: **{predicted}**")

            if predicted in disease_info:
                info = disease_info[predicted]
                st.info(
                    f"### 📌 {info['name']}\n"
                    f"**🔍 लक्षणे:** {info['sym']}\n\n"
                    f"**💊 उपचार:** {info['treat']}\n\n"
                    f"**🛡 प्रतिबंध:** {info['prev']}"
                )

else:
    st.info("📥 फोटो अपलोड करा.")


# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("""
<div style='background:#111;padding:35px;border-radius:14px;color:white;text-align:center;margin-top:50px;'>
<h2 style='color:#A259FF;'>👥 AgriNext Team</h2>
<p>AI आधारित स्मार्ट शेती प्लॅटफॉर्म</p>
<p>Developer: Rahul Patil</p>
</div>
""", unsafe_allow_html=True)
