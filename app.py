import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# 💾 Model info dictionary
MODELS = {
    "Model 1: Custom CNN": {
        "path": "pneumonia_cnn_model.h5",
        "accuracy": 0.8158
    },
    "Model 2: MobileNet": {
        "path": "pneumonia_classifier_moblevnet.h5",
        "accuracy": 0.888
    }
}

# 🎛 Sidebar
st.sidebar.title("⚙️ Configuration")
selected_model_name = st.sidebar.selectbox("Select Model", list(MODELS.keys()))

# 🚀 Load the selected model
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

model_info = MODELS[selected_model_name]
model = load_model(model_info["path"])

# 🧠 Header
st.markdown(
    """
    <div style='text-align: center'>
        <h1 style='color: #0B5345;'>🩺 Pneumonia Detection App</h1>
        <p style='font-size: 18px;'>Upload a chest X-ray and let the model detect signs of pneumonia</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(f"🧪 Using: **{selected_model_name}** | 🎯 Accuracy: **{model_info['accuracy']*100:.2f}%**")

# 📤 Upload Section
uploaded_file = st.file_uploader("📸 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')

    # 🖼 Show the image
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="📷 Uploaded X-ray", width=250)

    # 🔧 Preprocess
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    with col2:
        st.markdown("### 🧪 Run Diagnosis")
        if st.button("🔍 Predict"):
            prediction = model.predict(img)

            if prediction.shape[1] == 1:  # Binary classifier
                confidence = float(prediction[0][0])
                class_idx = int(np.round(confidence))
            else:
                confidence = float(np.max(prediction))
                class_idx = int(np.argmax(prediction))

            labels = ["Normal", "Pneumonia"]
            result_label = labels[class_idx]
            confidence_percent = confidence * 100 if class_idx == 1 else (1 - confidence) * 100

            st.success(f"🧬 Prediction: **{result_label}**")
            st.progress(int(confidence_percent))
            st.write(f"🧾 Confidence: **{confidence_percent:.2f}%**")

else:
    st.warning("Please upload a chest X-ray image to proceed.")
