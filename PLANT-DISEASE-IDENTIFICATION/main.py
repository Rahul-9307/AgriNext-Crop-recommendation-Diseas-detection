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
h1, h2, h3, h4 { text-align:center; font-family:'Poppins', sans-serif; }

.gradient-btn {
    background: linear-gradient(90deg, #6A5ACD, #00B4D8);
    color: white;
    padding: 12px 26px;
    border-radius: 12px;
    text-align:center;
    font-size: 18px;
    width: 100%;
    border:none;
}

.result-card {
    background: #ffffff;
    padding:20px;
    border-radius:15px;
    box-shadow:0 4px 12px rgba(0,0,0,0.15);
    text-align:center;
    margin-top:20px;
}

.upload-box {
    border: 2px dashed #6A5ACD;
    padding: 25px;
    border-radius: 15px;
}

.footer-box {
    background:#f2f2f2;
    padding:20px;
    border-radius:12px;
    margin-top:40px;
    font-size:15px;
}
</style>
""", unsafe_allow_html=True)



# -----------------------------------------------------------
# MODEL LOAD FUNCTION (SAFE + NEW API)
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
# PREDICTION FUNCTION
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
st.markdown("<h1>🌾 AgriNext – स्मार्ट वनस्पती रोग निदान</h1>", unsafe_allow_html=True)
st.write("___")


# -----------------------------------------------------------
# FILE UPLOAD SECTION
# -----------------------------------------------------------
st.markdown("<h3>📸 कृपया पानाचा फोटो अपलोड करा</h3>", unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["jpg", "png", "jpeg"])


if uploaded:

    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    st.image(uploaded, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # save temporarily
    temp_path = "temp_input.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # Predict Button
    if st.button("🔍 रोग ओळखा", key="predictbtn"):
        
        # Loader animation
        loader_gif = "https://i.gifer.com/ZZ5H.gif"
        st.markdown(f"<center><img src='{loader_gif}' width='140'></center>", unsafe_allow_html=True)

        if model is None:
            st.error("❌ मॉडेल फाइल मिळाली नाही! कृपया GitHub मध्ये फाइल योग्यरित्या अपलोड करा.")

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

            # Result Card
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown("<h3>🌱 ओळखलेला रोग</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:#2E8B57;'>✅ {class_name[idx]}</h2>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("📥 फोटो अपलोड करा.")


# -----------------------------------------------------------
# FOOTER – AGRINEXT TEAM
# -----------------------------------------------------------
st.markdown("""
<div class='footer-box'>
<h3>👥 AgriNext Team</h3>
<p>
AgriNext हे शेतकऱ्यांसाठी अत्याधुनिक तंत्रज्ञान वापरून तयार केलेले प्लॅटफॉर्म आहे.  
आमचे ध्येय –  
<strong>“प्रत्येक शेतकऱ्याला स्मार्ट शेतीची सुविधा देणे.”</strong><br><br>

🔸 AI आधारित रोग निदान  
🔸 पिक सल्ला  
🔸 स्थानिक भाषेत मार्गदर्शन  
🔸 शेत पातळीवरील निर्णय सहाय्य  

<br><b>टीम:</b><br>
• Rahul Patil (Developer)<br>
• AgriNext Research & Advisory Team
</p>
</div>
""", unsafe_allow_html=True)
