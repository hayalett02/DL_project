"""
Fashion MNIST görüntü sınıflandırma ANN ve CNN Karşılaştırması

Amaç:
    1. ANN ve CNN ile görüntü sınıflandırma
    2. Karşılaştır ve en iyi modeli seç

Veri seti:
    1. 28x28 -> 70.000 görüntü, gray scale, 10 sınıf

Plan program:
    1. veri setini yükleme
    2. veri ön işleme
    3. ANN modeli oluşturma ve eğitme
    4. CNN modeli oluşturma ve eğitme
    5. Sonuçları karşılaştırma
    6. Grafiksel olarak değerlendir ve CNN modelini kaydet    
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. veri seti yükleme
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# normalize
x_train = x_train / 255
x_test = x_test / 255

x_train_cnn = x_train[..., np.newaxis] # (70000, 28, 28) -> (70000, 28, 28 , 1)
x_test_cnn = x_test[..., np.newaxis]

# ann modeli
ann_model = keras.Sequential([
    layers.Flatten(input_shape = (28, 28)), # (70000, 28, 28) -> (70000, 28 x 28)
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

ann_model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

history_ann = ann_model.fit(
    x_train, y_train, epochs = 3, validation_split = 0.1, verbose = 1
)

ann_test_loss, ann_test_acc = ann_model.evaluate(x_test, y_test)
print("ann_test_acc: ",ann_test_acc)
"""
Epoch 1/3
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8245 - loss: 0.4853 - val_accuracy: 0.8647 - val_loss: 0.3643

Epoch 2/3
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8674 - loss: 0.3623 - val_accuracy: 0.8647 - val_loss: 0.3603

Epoch 3/3
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.8806 - loss: 0.3243 - val_accuracy: 0.8652 - val_loss: 0.3624

313/313 ━━━━━━━━━━━━━━━━━━━━ 0s 821us/step - accuracy: 0.8622 - loss: 0.3800
ann_test_acc:  0.8622000217437744
"""

# 4. CNN model eğitimi
cnn_model = keras.Sequential(
    [
        layers.Conv2D(32, (3,3), activation="relu", input_shape = (28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation = "softmax")
    ]
)
cnn_model.compile(
    optimizer= "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

history_cnn = cnn_model.fit(
    x_train_cnn, y_train,
    epochs = 3,
    validation_split = 0.1,
    verbose = 1
)

cnn_test_loss, cnn_test_acc = cnn_model.evaluate(x_test_cnn, y_test)
print("cnn_test_acc: ",cnn_test_acc)

"""
Epoch 1/3
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.8352 - loss: 0.4555 - val_accuracy: 0.8685 - val_loss: 0.3682
Epoch 2/3
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.8878 - loss: 0.3062 - val_accuracy: 0.8927 - val_loss: 0.2890
Epoch 3/3
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.9033 - loss: 0.2629 - val_accuracy: 0.8942 - val_loss: 0.2782
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8903 - loss: 0.2943 
cnn_test_acc:  0.8902999758720398
"""

cnn_model.save("fashion_mnist_cnn_model.keras")
#cnn_model.save("fashion_mnist_cnn_model.h5")
