import cv2
import numpy as np

# 1. Cargar imagen original a color
ruta_imagen = 'Proyecto/documento.jpg'
imagen_color = cv2.imread(ruta_imagen)

if imagen_color is None:
    print(f"Error: No se pudo cargar la imagen '{ruta_imagen}'.")
else:
   
    # Paso A: Conversión a escala de grises
    imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

    # Paso B: Umbralización 
    _, imagen_binaria = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Paso C: Limpieza Morfológica 
    kernel = np.ones((3, 3), np.uint8)
    imagen_limpia = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)

    # Paso D: Detección de Contornos 
    contornos, jerarquia = cv2.findContours(imagen_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Definir umbral de área para clasificar objetos (Lógica empresarial)
    umbral_area_grande = 1500

    print(" MÉTICAS DE OBJETOS DETECTADOS ")
    
    # Iterar sobre cada contorno encontrado 
    for i, cnt in enumerate(contornos):
        # 1. Calcular el Área
        area = cv2.contourArea(cnt)

        # Filtrar ruido menor (objetos muy pequeños)
        if area > 500:
            print(f"Objeto {i+1}: Área = {area:.2f} píxeles")

            # 2. Calcular el Bounding Box (Rectángulo envolvente)
            x, y, w, h = cv2.boundingRect(cnt)

            # 4. Clasificación según área 
            if area > umbral_area_grande:
                color_box = (255, 0, 0)  # Azul BGR (Objeto Grande)
                etiqueta = "Grande"
            else:
                color_box = (0, 0, 255)  # Rojo BGR (Objeto Pequeño)
                etiqueta = "Pequeno"

            # 3. Dibujar el rectángulo sobre la imagen ORIGINAL a color
            cv2.rectangle(imagen_color, (x, y), (x + w, y + h), color_box, 2)

            # Dibujar el centroide 
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Dibujar punto rojo en el centro y etiqueta
                cv2.circle(imagen_color, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(imagen_color, etiqueta, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 1)

    # Guardar resultado para visualizar en Codespaces
    cv2.imwrite('Proyecto/clasificador_objetos.jpg', imagen_color)
    print("\nProceso finalizado. El resultado visual se guardó en 'Proyecto/clasificador_objetos.jpg'.")