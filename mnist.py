import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# Cargar el dataset MNIST
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los valores de los píxeles (de 0-255 a 0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Definir el modelo de red neuronal
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Convertir la imagen en un vector
    keras.layers.Dense(128, activation='relu'),  # Capa oculta con 128 neuronas
    keras.layers.Dropout(0.2),  # Regularización para evitar sobreajuste
    keras.layers.Dense(10, activation='softmax')  # Capa de salida con 10 clases
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

start = time.time()

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

finish = time.time()
print(f"\nTiempo de entrenamiento: {finish - start:.2f} segundos")

# Evaluar el modelo con datos de prueba
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)


# Mostrar precisión del modelo
print(f"\nPrecisión en test: {test_acc:.4f}")

# # Graficar la evolución del entrenamiento
# plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
# plt.plot(history.history['val_accuracy'], label='Precisión validación')
# plt.xlabel('Épocas')
# plt.ylabel('Precisión')
# plt.legend()
# plt.show()
# plt.close()

# # Generar predicciones en el conjunto de prueba
# y_pred = np.argmax(model.predict(x_test), axis=1)

# # Crear la matriz de confusión
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Visualizar la matriz de confusión
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
# plt.xlabel("Predicción")
# plt.ylabel("Etiqueta Real")
# plt.title("Matriz de Confusión")
# plt.show()


