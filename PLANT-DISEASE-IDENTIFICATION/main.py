import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="AgriNext – स्मार्ट रोग निदान", layout="centered")


# -----------------------------------------------------------
# AUTO-DETECT MODEL FILE (WORKS 100%)
# -----------------------------------------------------------
@st.cache_resource
def load_model():

    # Search model inside current folder & all subfolders
    model_name = "trained_plant_disease_model.keras"
    found_path = None

    for root, dirs, files in os.walk(".", topdown=True):
        if model_name in files:
            found_path = os.path.join(root, model_name)
            break

    # Debug (shows where model is found)
    st.write("🔍 Searching model in folders...")
    st.write("📂 Files:", os.listdir("."))

    if found_path:
        st.success(f"✔ Model Found at: {found_path}")
        return tf.keras.models.load_model(found_path)

    st.error("❌ Model file NOT found! Please keep trained_plant_disease_model.keras anywhere inside project.")
    return None


model = load_model()


# -----------------------------------------------------------
# PREDICT FUNCTION
# -----------------------------------------------------------
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    pred = model.predict(arr)
    return np.argmax(pred)


# -----------------------------------------------------------
# SAMPLE DISEASE INFO
# -----------------------------------------------------------
disease_info = {
    "Apple___Apple_scab": {
        "title": "Apple Scab (सफरचंद स्कॅब)",
        "symptoms": "पानांवर काळपट डाग, फळे विकृत.",
        "treat": "मॅन्कोझेब / क्लोरोथॅलोनील फवारणी.",
        "prevent": "संक्रमित पाने जाळा."
    },
    "Tomato___Late_blight": {
        "title": "Late Blight (लेट ब्लाईट)",
        "symptoms": "पानांवर तपकिरी पाण्यासारखे डाग.",
        "treat": "मेटालेक्सिल + मॅन्कोझेब.",
        "prevent": "आर्द्रता कमी ठेवा."
    }
}


# -----------------------------------------------------------
# UI HEADER
# -----------------------------------------------------------
st.markdown("<h1 style='color:#A259FF;text-align:center;'>🌾 AgriNext – स्मार्ट वनस्पती रोग निदान</h1>", unsafe_allow_html=True)
st.write("___")


# -----------------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------------
uploaded = st.file_uploader("📸 पानाचा फोटो अपलोड करा", type=["jpg", "jpeg", "png"])

if uploaded:

    st.image(uploaded, use_column_width=True)

    temp_image = "temp_uploaded_image.jpg"
    with open(temp_image, "wb") as f:
        f.write(uploaded.getbuffer())

    if st.button("🔍 रोग ओळखा"):

        loader = st.empty()
        loader.markdown("<center><img src='https://i.gifer.com/ZZ5H.gif' width='120'></center>",
                        unsafe_allow_html=True)

        if model is None:
            loader.empty()
            st.error("❌ Model load झाला नाही!")

        else:
            idx = predict_image(temp_image)

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

            # Disease Info
            if predicted in disease_info:
                info = disease_info[predicted]
                st.info(
                    f"### 📌 {info['title']}\n"
                    f"**🔍 लक्षणे:** {info['symptoms']}\n\n"
                    f"**💊 उपचार:** {info['treat']}\n\n"
                    f"**🛡 प्रतिबंध:** {info['prevent']}"
                )

else:
    st.info("📥 कृपया फोटो अपलोड करा.")


# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("""
<div style='background:#111;padding:35px;border-radius:12px;color:white;text-align:center;margin-top:50px;'>
<h2 style='color:#A259FF;'>👥 AgriNext Team</h2>
<p>AI आधारित स्मार्ट शेती प्लॅटफॉर्म</p>
<p>Developer: Rahul Patil</p>
</div>
""", unsafe_allow_html=True)
