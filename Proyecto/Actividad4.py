import cv2
import numpy as np

# 1. Cargar la imagen original de entrada
ruta_imagen = 'Proyecto/documento.jpg'
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print(f"Error: No se pudo encontrar o cargar la imagen en '{ruta_imagen}'.")
else:
    #  Generación de Ruido de Sal y Pimienta 
    imagen_ruidosa = imagen.copy()
    prob_ruido = 0.05  # 5% de píxeles afectados
    
    # Agregar Ruido de Sal 
    num_sal = np.ceil(prob_ruido * imagen.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_sal)) for i in imagen.shape[:2]]
    imagen_ruidosa[tuple(coords)] = 255

    # Agregar Ruido de Pimienta 
    num_pimienta = np.ceil(prob_ruido * imagen.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pimienta)) for i in imagen.shape[:2]]
    imagen_ruidosa[tuple(coords)] = 0

    # Guardar la imagen con ruido sintético
    cv2.imwrite('Proyecto/1_imagen_ruidosa.jpg', imagen_ruidosa)

    #  Aplicación de Filtros con Kernel agresivo 7x7 
    kernel_size = 7

    # A. Filtro de Media (Promedio)
    blur_media = cv2.blur(imagen_ruidosa, (kernel_size, kernel_size))

    # B. Filtro Gaussiano
    blur_gauss = cv2.GaussianBlur(imagen_ruidosa, (kernel_size, kernel_size), 0)

    # C. Filtro de Mediana
    blur_mediana = cv2.medianBlur(imagen_ruidosa, kernel_size)

    # Guardar Resultados para Codespaces (Reemplazo de cv2.imshow)
    cv2.imwrite('Proyecto/2_filtro_media.jpg', blur_media)
    cv2.imwrite('Proyecto/3_filtro_gaussiano.jpg', blur_gauss)
    cv2.imwrite('Proyecto/4_filtro_mediana.jpg', blur_mediana)

    print(" TALLER DE LABORATORIO ACTIVIDAD 4 COMPLETADO ")
    print("Se han guardado los siguientes archivos en la carpeta 'Proyecto':")
    print(" 1. Proyecto/1_imagen_ruidosa.jpg  (Imagen original con ruido Sal y Pimienta)")
    print(" 2. Proyecto/2_filtro_media.jpg    (Resultado Filtro de Media 7x7)")
    print(" 3. Proyecto/3_filtro_gaussiano.jpg(Resultado Filtro Gaussiano 7x7)")
    print(" 4. Proyecto/4_filtro_mediana.jpg  (Resultado Filtro de Mediana 7x7)")
    print("\n--- ANÁLISIS CRÍTICO (Punto 4) ---")
    print("El Filtro de Mediana elimina por completo los puntos extremos (0 y 255)")
    print("porque la mediana estadística elige el valor central del conjunto ordenado,")
    print("descartando los valores atípicos.")
    print("En cambio, el Filtro de Media incluye estos valores extremos en la sumatoria,")
    print("distribuyendo el brillo/oscuridad en el vecindario de 7x7 y creando 'manchas grises'.")