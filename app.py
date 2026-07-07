import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Horse vs Kangaroo", page_icon="🦒")

model = tf.keras.models.load_model("Horse_Kangaroo_classifier.keras")

class_names = ["Horse", "Kangaroo"]

st.title("🐎 Horse vs 🦘 Kangaroo Classifier")
st.write("Upload an image to predict whether it is a Horse or Kangaroo.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    image = image.resize((150, 150))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction[0])

    st.subheader("Prediction Result")
    st.success(f"Animal : {class_names[np.argmax(score)]}")
    st.info(f"Confidence : {100*np.max(score):.2f}%")