"""
Fashion MNIST CNN Model Streamlit Arayüzü

Amaç:
    Eğitilmiş CNN modelini kullanarak kullanıcıların yüklediği kıyafet görüntülerini
    sınıflandıran bir web arayüzü oluşturmak

Veri: internetten indirelim

Plan program:
    1. modeli yüklemek
    2. dosya yükleme arayüzü
    3. görüntü ön işleme
    4. tahmin
    5. sonuçların görselleştirilmesi

Kurulumlar:
    pip install streamlit pillow
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# =========================
# SINIF İSİMLERİ
# =========================
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# model yükle
@st.cache_resource
def load_model():
    return keras.models.load_model("fashion_mnist_cnn_model.keras")

model = load_model() # cnn modeli bunun içerisinde

# ui
st.title("Fashion MNIST CNN Tahmin Arayüzü")
st.write("Bir kıyafet görüntüsü yükleyin ve model tahmin etsin")

uploaded_file = st.file_uploader(
    "Görüntü Yükle",
    type = ["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption = "Yüklenen Görüntü", width = 200)

    # preprocessing
    img = image.resize((28, 28))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis = -1) 
    img = np.expand_dims(img, axis = 0) # (28, 28) -> (1, 28, 28, 1)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Tahmin: {class_names[class_index]}")
    st.write(f"Güven: {confidence:.2f}")