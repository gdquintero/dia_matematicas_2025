import tensorflow as tf
import time
from keras import layers, models


# Verifica que está usando GPU
print("GPU disponible:", tf.config.list_physical_devices('GPU'))

# Cargar CIFAR-100
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizar

# Modelo CNN profundo
model = models.Sequential([
    layers.Conv2D(64, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(100, activation='softmax')  # 100 clases
])

# Compilar
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar (esto exige mucho la GPU)
start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.1)
end = time.time()

print(f"Tiempo de entrenamiento: {(end - start)/60:.2f} minutos")

# Evaluar
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Precisión en test: {test_acc:.3f}")
