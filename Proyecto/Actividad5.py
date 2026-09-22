import cv2
import numpy as np

# 1. Cargar imagen en escala de grises 
imagen = cv2.imread('Proyecto/documento.jpg', cv2.IMREAD_GRAYSCALE)

if imagen is None:
    print("Error: No se pudo cargar la imagen 'Proyecto/documento.jpg'.")
else:
    # 2. Detección de Bordes con Sobel 
    sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)

    # 3. Detección de Bordes con Canny 
    bordes_canny = cv2.Canny(imagen, 50, 150)

    # 4. Experimentación con umbrales de Canny 
    canny_bajo = cv2.Canny(imagen, 10, 50)    # Umbrales bajos (captura más ruido)
    canny_alto = cv2.Canny(imagen, 200, 250)  # Umbrales altos (solo bordes muy fuertes)

    # Guardar resultados para visualizar 
    cv2.imwrite('Proyecto/sobel_x.jpg', sobel_x_abs)
    cv2.imwrite('Proyecto/sobel_y.jpg', sobel_y_abs)
    cv2.imwrite('Proyecto/canny_normal.jpg', bordes_canny)
    cv2.imwrite('Proyecto/canny_bajo.jpg', canny_bajo)
    cv2.imwrite('Proyecto/canny_alto.jpg', canny_alto)

    print("Proceso completado. Se han guardado los resultados en 'Proyecto/':")
    print(" - sobel_x.jpg (Bordes verticales)")
    print(" - sobel_y.jpg (Bordes horizontales)")
    print(" - canny_normal.jpg (Umbrales 50-150)")
    print(" - canny_bajo.jpg (Umbrales 10-50)")
    print(" - canny_alto.jpg (Umbrales 200-250)")