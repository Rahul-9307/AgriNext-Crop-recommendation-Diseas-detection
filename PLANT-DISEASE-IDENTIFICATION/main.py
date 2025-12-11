import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

st.set_page_config(page_title="AgriNext – स्मार्ट रोग निदान", layout="centered")

# -----------------------------------------------------------
# CSS
# -----------------------------------------------------------
st.markdown("""
<style>
h1, h2, h3 { text-align:center; font-family:'Poppins', sans-serif; }

.result-card {
    background:#fff;
    padding:25px;
    border-radius:18px;
    box-shadow:0 4px 15px rgba(0,0,0,0.2);
    margin-top:20px;
    text-align:center;
}

.info-box {
    background:#EEF1FF;
    padding:20px;
    border-radius:15px;
    margin-top:20px;
    border-left:6px solid #6A5ACD;
    font-size:17px;
}

.footer {
    width:100%;
    background:#111;
    padding:40px;
    margin-top:60px;
    border-radius:16px;
    color:white;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    current = os.path.dirname(__file__)
    p1 = os.path.join(current, "trained_plant_disease_model.keras")
    p2 = os.path.join(current, "trained_plant_disease_model.h5")

    if os.path.exists(p1): return tf.keras.models.load_model(p1)
    if os.path.exists(p2): return tf.keras.models.load_model(p2)
    return None

model = load_model()


# -----------------------------------------------------------
# DISEASE INFO
# -----------------------------------------------------------
disease_info = {
    "Apple___Apple_scab": {
        "name": "Apple Scab (सफरचंद स्कॅब)",
        "sym": "पानांवर काळपट डाग, फळे विकृत.",
        "treat": "मॅन्कोझेब / क्लोरोथॅलोनील फवारणी.",
        "prev": "संक्रमित पाने जाळा, हवेचा प्रवेश वाढवा."
    },

    "Tomato___Late_blight": {
        "name": "Tomato Late Blight (टोमॅटो लेट ब्लाईट)",
        "sym": "पानांवर तपकिरी पाण्यासारखे डाग.",
        "treat": "मेटालेक्सिल + मॅन्कोझेब.",
        "prev": "जास्त आर्द्रता टाळा, रोगग्रस्त झाडे काढून टाका."
    }
}


# -----------------------------------------------------------
# PREDICT FUNCTION
# -----------------------------------------------------------
def predict_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(128,128))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    pred = model.predict(arr)
    return np.argmax(pred)


# -----------------------------------------------------------
# UI
# -----------------------------------------------------------
st.markdown("<h1 style='color:#A259FF;'>🌾 AgriNext – स्मार्ट रोग निदान</h1>", unsafe_allow_html=True)
st.write("___")

uploaded = st.file_uploader("📸 पानाचा फोटो अपलोड करा", type=["jpg","jpeg","png"])

if uploaded:
    st.image(uploaded, use_column_width=True)

    # save temp
    temp = "temp_img.jpg"
    with open(temp, "wb") as f:
        f.write(uploaded.getbuffer())

    if st.button("🔍 रोग ओळखा"):

        # --------------------------
        # VISIBLE LOADER FIX
        # --------------------------
        loader = st.empty()
        loader.markdown("<center><img src='https://i.gifer.com/ZZ5H.gif' width='140'></center>", unsafe_allow_html=True)

        if model is None:
            loader.empty()
            st.error("❌ मॉडेल फाइल सापडली नाही!")
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

            # stop loader
            loader.empty()

            # Show result
            st.markdown(f"""
            <div class='result-card'>
                <h3>🌱 ओळखलेला रोग</h3>
                <h2 style='color:#2E8B57;'>{predicted}</h2>
            </div>
            """, unsafe_allow_html=True)

            # Disease Information
            if predicted in disease_info:
                info = disease_info[predicted]
                st.markdown(f"""
                <div class='info-box'>
                    <b>📌 रोगाचे नाव:</b> {info['name']} <br><br>
                    <b>🔍 लक्षणे:</b> {info['sym']} <br><br>
                    <b>💊 उपचार:</b> {info['treat']} <br><br>
                    <b>🛡 प्रतिबंध:</b> {info['prev']}
                </div>
                """, unsafe_allow_html=True)

else:
    st.info("📥 फोटो अपलोड करा.")


# -----------------------------------------------------------
# FOOTER FIXED (Always Visible)
# -----------------------------------------------------------
st.markdown("""
<div class='footer'>
    <h2 style='color:#A259FF;'>👥 AgriNext Team</h2>
    <p>AI आधारित स्मार्ट शेती प्लॅटफॉर्म — शेतकऱ्यांसाठी बनवलेले.</p>
    <p>Team: Rahul Patil & AgriNext Advisory Group</p>
</div>
""", unsafe_allow_html=True)
