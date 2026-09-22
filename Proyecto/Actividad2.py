import cv2
import numpy as np
import matplotlib.pyplot as plt


# TALLER DE LABORATORIO 1: TRANSFORMACIÓN DE ESPACIOS

print("--- TALLER DE LABORATORIO 1 ---")

# 1. Crear píxel BGR completamente amarillo (Azul=0, Verde=255, Rojo=255)
pixel_bgr = np.array([0, 255, 255], dtype=np.float32)

# 2. Pesos ponderados para conversión BGR a Grises: Y = 0.114*B + 0.587*G + 0.299*R
pesos = np.array([0.114, 0.587, 0.299])

# Producto punto entre el píxel y los pesos
valor_gris = np.dot(pixel_bgr, pesos)

# 3. Imprimir el resultado
print(f"El valor equivalente en escala de grises para el amarillo puro es: {valor_gris:.2f}\n")

# 4. Probar conversión con una imagen real guardando el resultado
ruta_imagen = 'Proyecto/leon.jpg'  # Puedes usar 'Proyecto/documento.jpg' o 'Proyecto/leon.jpg'
imagen_real = cv2.imread(ruta_imagen)

if imagen_real is not None:
    # Conversión mediante OpenCV
    img_gris = cv2.cvtColor(imagen_real, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('Proyecto/resultado_grises.jpg', img_gris)
    print("Imagen convertida a escala de grises guardada en: 'Proyecto/resultado_grises.jpg'")
else:
    print(f"No se pudo cargar la imagen de prueba en '{ruta_imagen}'.")


# TALLER DE LABORATORIO 2: ANÁLISIS ESTADÍSTICO

print("--- TALLER DE LABORATORIO 2 ---")

if imagen_real is not None:
    
    canal_b, canal_g, canal_r = cv2.split(imagen_real)

    
    hist_b = cv2.calcHist([canal_b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([canal_g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([canal_r], [0], None, [256], [0, 256])

   
    plt.figure(figsize=(10, 6))
    plt.plot(hist_b, color='blue', label='Canal Azul (B)')
    plt.plot(hist_g, color='green', label='Canal Verde (G)')
    plt.plot(hist_r, color='red', label='Canal Rojo (R)')
    
    plt.title("Distribución de Intensidades por Canal (RGB/BGR)")
    plt.xlabel("Valor del Pixel (0-255)")
    plt.ylabel("Frecuencia (Cantidad de píxeles)")
    plt.legend()
    plt.grid(True)

  
    ruta_grafico = 'Proyecto/histograma_canales.png'
    plt.savefig(ruta_grafico)
    plt.close()

    print(f"Gráfico del histograma guardado en: '{ruta_grafico}'")
    
  
    promedio_b = np.mean(canal_b)
    promedio_g = np.mean(canal_g)
    promedio_r = np.mean(canal_r)
    
    canales = {'Azul': promedio_b, 'Verde': promedio_g, 'Rojo': promedio_r}
    dominante = max(canales, key=canales.get)
    
    print(f"Conclusión analítica:")
    print(f"- Promedio de intensidades: Azul={promedio_b:.2f}, Verde={promedio_g:.2f}, Rojo={promedio_r:.2f}")
    print(f"- El color dominante en la iluminación general es el **{dominante}**.")