import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Agri🌾Next – स्मार्ट रोग निदान", layout="centered")


# -----------------------------------------------------------
# HERO IMAGE CSS DESIGN
# -----------------------------------------------------------
st.markdown("""
<style>
.hero-box {
    width: 100%;
    border-radius: 18px;
    overflow: hidden;
    border: 2px solid #2ecc71;
    margin-top: 10px;
    box-shadow: 0px 0px 15px rgba(0,255,150,0.2);
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# SHOW HERO IMAGE
# -----------------------------------------------------------
if os.path.exists("Diseases.png"):
    st.markdown("<div class='hero-box'>", unsafe_allow_html=True)
    st.image("banner.jpg", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.warning("⚠ banner.jpg फाइल सापडली नाही! कृपया कोड फोल्डरमध्ये ठेवा.")


st.write("")  # spacing


# -----------------------------------------------------------
# FINAL CLEAN MODEL LOADER (AUTO-DETECT)
# -----------------------------------------------------------
@st.cache_resource
def load_model():

    model_name = "trained_plant_disease_model.keras"
    found_path = None

    for root, dirs, files in os.walk(".", topdown=True):
        if model_name in files:
            found_path = os.path.join(root, model_name)
            break

    if found_path:
        return tf.keras.models.load_model(found_path)

    st.error("❌ Model file NOT found! Add trained_plant_disease_model.keras inside project.")
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
# BASIC DISEASE INFO
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
        "treat": "मेटालेक्सिल + मॅन्कोझेब फवारणी.",
        "prevent": "जास्त आर्द्रता टाळा."
    }
}


# -----------------------------------------------------------
# TITLE UI
# -----------------------------------------------------------
st.markdown("""
<h1 style='color:#2ecc71;text-align:center; font-weight:700; margin-top:30px;'>
AgriSens: Smart Disease Detection
</h1>
<p style='text-align:center; color:#bbb; font-size:18px;'>
AI आधारित वनस्पती रोग ओळख प्रणाली
</p>
""", unsafe_allow_html=True)

st.write("___")


# -----------------------------------------------------------
# FILE UPLOADER
# -----------------------------------------------------------
uploaded = st.file_uploader("📸 पानाचा फोटो अपलोड करा", type=["jpg", "jpeg", "png"])

if uploaded:

    st.image(uploaded, use_column_width=True)

    temp_path = "uploaded_temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    if st.button("🔍 रोग ओळखा"):

        loader = st.empty()
        loader.markdown(
            "<center><img src='https://i.gifer.com/ZZ5H.gif' width='120'></center>",
            unsafe_allow_html=True
        )

        if model is None:
            loader.empty()
            st.error("❌ Model लोड झाला नाही!")

        else:
            idx = predict_image(temp_path)

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

            # -----------------------------------------------------------
            # RESULT CARD
            # -----------------------------------------------------------
            if predicted in disease_info:
                d = disease_info[predicted]

                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #6a11cb, #2575fc);
                    padding: 25px;
                    border-radius: 15px;
                    color: white;
                    margin-top: 20px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                ">
                    <h2 style="text-align:center; margin-bottom:10px;">🌿 {d['title']}</h2>

                    <p style="font-size:18px; line-height:1.6;">
                        <b>🔍 लक्षणे:</b> {d['symptoms']} <br><br>
                        <b>💊 उपचार:</b> {d['treat']} <br><br>
                        <b>🛡 प्रतिबंध:</b> {d['prevent']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

else:
    st.info("📥 कृपया फोटो अपलोड करा.")


# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("""
<div style='background:#111;padding:35px;border-radius:12px;color:white;text-align:center;margin-top:50px;'>
<h2 style='color:#2ecc71;'>👥 Support by Agri🌾Next Team</h2>
<p>AI आधारित स्मार्ट शेती प्लॅटफॉर्म</p>
<p>Developer: Agri🌾Next</p>
</div>
""", unsafe_allow_html=True)

