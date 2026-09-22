import cv2
import numpy as np


ruta_imagen = 'Proyecto/documento.jpg'
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)


if imagen is None:
    print(f"Error: No se pudo encontrar o cargar la imagen en '{ruta_imagen}'.")
else:
    
    _, imagen_binaria = cv2.threshold(imagen, 110, 255, cv2.THRESH_BINARY)

 
    kernel = np.ones((3, 3), np.uint8)

  
    apertura = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)

    
    cierre = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel)

  
    cv2.imwrite('Proyecto/resultado_binario.jpg', imagen_binaria)
    cv2.imwrite('Proyecto/resultado_apertura.jpg', apertura)
    cv2.imwrite('Proyecto/resultado_cierre.jpg', cierre)

    print("Proceso completado con éxito. Se han guardado los resultados:")
    print(" - Proyecto/resultado_binario.jpg")
    print(" - Proyecto/resultado_apertura.jpg")
    print(" - Proyecto/resultado_cierre.jpg")