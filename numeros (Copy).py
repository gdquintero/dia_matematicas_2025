import os
import cv2
import numpy as np
import argparse

def procesar_imagenes(carpeta_entrada, carpeta_salida):
    """Procesa imágenes de números escritos a mano y las guarda en formato .npy"""
    
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    for archivo in os.listdir(carpeta_entrada):
        if archivo.endswith(('.png', '.jpg', '.jpeg')):
            ruta_imagen = os.path.join(carpeta_entrada, archivo)
            imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)  # Cargar en escala de grises
            
            # Verificar si la imagen se cargó correctamente
            if imagen is None:
                print(f"No se pudo cargar la imagen: {archivo}")
                continue
            
            # Redimensionar a 28x28 píxeles
            imagen = cv2.resize(imagen, (28, 28))
            
            # Invertir colores (si el fondo es blanco y los números son negros)
            if np.mean(imagen) > 127:
                imagen = 255 - imagen
            
            # Normalizar valores a rango [0,1]
            imagen = imagen.astype('float32') / 255.0
            
            # Agregar dimensión de batch y canal de color
            imagen = np.expand_dims(imagen, axis=0)  # Agregar batch
            imagen = np.expand_dims(imagen, axis=-1)  # Agregar canal
            
            # Guardar la imagen procesada
            nombre_salida = os.path.splitext(archivo)[0] + ".npy"
            ruta_salida = os.path.join(carpeta_salida, nombre_salida)
            np.save(ruta_salida, imagen)
            print(f"Imagen procesada y guardada: {ruta_salida}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar imágenes de números escritos a mano para modelos de MNIST")
    parser.add_argument("--input", type=str, required=True, help="Carpeta de entrada con imágenes")
    parser.add_argument("--output", type=str, required=True, help="Carpeta de salida para imágenes procesadas")
    args = parser.parse_args()
    
    procesar_imagenes(args.input, args.output)
