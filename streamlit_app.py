import streamlit as st
import numpy as np
import tensorflow as tf
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

# Model mimarisini kodla tanımla (Versiyon uyumsuzluklarını önlemek için en güvenli yöntem)
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    return model

# model yükle (Sadece ağırlıkları yükler)
@st.cache_resource
def load_model():
    model = create_model()
    model.load_weights("fashion_mnist_cnn_weights.weights.h5")
    return model

model = load_model()  # cnn modeli bunun içerisinde

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