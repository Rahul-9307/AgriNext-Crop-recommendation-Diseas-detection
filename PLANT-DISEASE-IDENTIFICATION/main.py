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
# CUSTOM CSS
# -----------------------------------------------------------
st.markdown("""
<style>

h1, h2, h3, h4 {
    text-align:center;
    font-family:'Poppins', sans-serif;
}

/* Gradient Button */
.gradient-btn {
    background: linear-gradient(90deg, #6A5ACD, #00B4D8);
    color: white;
    padding: 14px 26px;
    border-radius: 12px;
    text-align:center;
    font-size: 18px;
    width: 100%;
    border:none;
    margin-top: 10px;
}

/* Card */
.result-card {
    background: #ffffff;
    padding:25px;
    border-radius:18px;
    box-shadow:0 4px 15px rgba(0,0,0,0.2);
    text-align:center;
    margin-top:25px;
}

/* Info card */
.info-card {
    background:#F8F9FF;
    padding:20px;
    border-radius:18px;
    font-size:18px;
    line-height:1.6;
    margin-top:20px;
    border-left:6px solid #6A5ACD;
}

/* Upload Box */
.upload-box {
    border: 2px dashed #6A5ACD;
    padding: 25px;
    border-radius: 15px;
    text-align:center;
}

/* Footer */
.footer-card {
    background:#1a1a1a;
    padding:50px;
    border-radius:18px;
    margin-top:80px;
    color:white;
    font-family:'Poppins', sans-serif;
    width:100%;
}

.footer-title {
    text-align:center;
    font-size:32px;
    font-weight:700;
    color:#A259FF;
    margin-bottom:10px;
}

.footer-text {
    font-size:18px;
    line-height:1.8;
    text-align:center;
}

.footer-bullets {
    font-size:18px;
    margin-top:15px;
    text-align:center;
}

.team-label {
    font-size:22px;
    font-weight:600;
    margin-top:25px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)



# -----------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------
@st.cache_resource
def load_model():

    current_dir = os.path.dirname(__file__)
    keras_path = os.path.join(current_dir, "trained_plant_disease_model.keras")
    h5_path = os.path.join(current_dir, "trained_plant_disease_model.h5")

    if os.path.exists(keras_path):
        return tf.keras.models.load_model(keras_path)

    if os.path.exists(h5_path):
        return tf.keras.models.load_model(h5_path)

    return None


model = load_model()


# -----------------------------------------------------------
# DISEASE INFORMATION DICTIONARY
# -----------------------------------------------------------
disease_info = {
    "Apple___Apple_scab": {
        "name": "Apple Scab (सफरचंद स्कॅब रोग)",
        "symptoms": "पानांवर काळपट व गोल ठिपके, फळांवर विकृती.",
        "treatment": "मँकोझेब/क्लोरोथॅलोनील फवारणी.",
        "prevention": "बागेत हवेचा प्रवेश वाढवा, संक्रमित पाने जाळा."
    },

    "Tomato___Late_blight": {
        "name": "Tomato Late Blight (टोमॅटो लेट ब्लाईट)",
        "symptoms": "पानांवर तपकिरी पाण्यासारखे डाग, संपूर्ण झाड मरते.",
        "treatment": "मेटालेक्सिल + मँकोझेब फवारणी.",
        "prevention": "जास्त आर्द्रता टाळा, रोगग्रस्त झाडे हटवा."
    },

    "Potato___Early_blight": {
        "name": "Potato Early Blight (बटाटा अर्ली ब्लाईट)",
        "symptoms": "पानांवर वर्तुळाकार रिंगयुक्त डाग.",
        "treatment": "क्लोरोथॅलोनील / मँकोझेब स्प्रे.",
        "prevention": "योग्य अंतर ठेवून लागवड, रोगग्रस्त पाने काढून टाका."
    },

    # You can extend more diseases later...
}



# -----------------------------------------------------------
# PREDICT FUNCTION
# -----------------------------------------------------------
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128,128))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    result = model.predict(arr)
    return np.argmax(result)



# -----------------------------------------------------------
# HEADER
# -----------------------------------------------------------
st.markdown("<h1 style='color:#A259FF; font-weight:700;'>🌾 AgriNext – स्मार्ट वनस्पती रोग निदान</h1>", unsafe_allow_html=True)
st.write("___")



# -----------------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------------
st.markdown("<h3>📸 कृपया पानाचा फोटो अपलोड करा</h3>", unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])


if uploaded:

    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    st.image(uploaded, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    temp_path = "temp_input.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    if st.button("🔍 रोग ओळखा", help="Predict Disease"):

        # Center loader properly
        st.markdown(
            "<center><img src='https://i.gifer.com/ZZ5H.gif' width='120'></center>",
            unsafe_allow_html=True
        )

        if model is None:
            st.error("❌ मॉडेल फाइल मिळाली नाही!")

        else:
            idx = predict_image(temp_path)

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

            predicted = class_name[idx]

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown("<h3>🌱 ओळखलेला रोग</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:#2E8B57;'>✅ {predicted}</h2>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Show disease info if available
            if predicted in disease_info:
                info = disease_info[predicted]

                st.markdown(f"""
                <div class='info-card'>
                    <b>📌 रोगाचे नाव:</b> {info['name']} <br><br>
                    <b>🔍 लक्षणे:</b> {info['symptoms']} <br><br>
                    <b>💊 उपचार:</b> {info['treatment']} <br><br>
                    <b>🛡️ प्रतिबंध:</b> {info['prevention']}
                </div>
                """, unsafe_allow_html=True)

else:
    st.info("📥 फोटो अपलोड करा.")



# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("""
<div class='footer-card'>
    <div class='footer-title'>👥 AgriNext Team</div>

    <div class='footer-text'>
        AgriNext — शेतकऱ्यांसाठी विकसित पुढील पिढीचे AI प्लॅटफॉर्म  
        आमचे ध्येय: <strong>“प्रत्येक शेतकऱ्याला स्मार्ट शेतीची सुविधा देणे.”</strong>
    </div>

    <div class='footer-bullets'>
        🔹 AI आधारित रोग निदान <br>
        🔹 पिक सल्ला<br>
        🔹 स्थानिक भाषेत मार्गदर्शन<br>
        🔹 स्मार्ट निर्णय सहाय्य
    </div>

    <div class='team-label'>टीम:</div>
    <div class='footer-text'>
        • Rahul Patil (Developer) <br>
        • AgriNext Research & Advisory Team
    </div>
</div>
""", unsafe_allow_html=True)
