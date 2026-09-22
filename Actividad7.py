import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

print("=" * 60)
print("1. TALLER ANALÍTICO: LA VOTACIÓN ESPACIAL")
print("=" * 60)

# Datos del taller 
nuevo_punto = (30, 40)
puntos = {
    'A': (20, 30),  # NO COMPRA (0)
    'B': (40, 50),  # COMPRA (1)
    'C': (35, 45)   # COMPRA (1)
}

# 1.1 Cálculo de Distancias Euclidianas
distancias = {}
for nombre, coords in puntos.items():
    dist = math.sqrt((nuevo_punto[0] - coords[0])**2 + (nuevo_punto[1] - coords[1])**2)
    distancias[nombre] = dist
    print(f"Distancia desde Nuevo Punto {nuevo_punto} a {nombre}{coords}: {dist:.4f}")

# El vecino más cercano es C (distancia ~7.07) -> Clase: COMPRA
print("\n[Respuesta K=1]: El vecino más cercano es C (distancia 7.07). Clasificación: COMPRA (1)")

# 1.3 Clasificación con K = 3
# Vecinos: A (NO COMPRA), B (COMPRA), C (COMPRA) -> Votación: 2 COMPRA vs 1 NO COMPRA
print("[Respuesta K=3]: Votación entre A, B y C (2 a favor de COMPRA, 1 de NO COMPRA). Clasificación: COMPRA (1)")
print("[Análisis]: No hubo cambio en la decisión, en ambos casos clasifica como COMPRA.")


print("\n" + "=" * 60)
print("2. TALLER DE LABORATORIO: CLASIFICADOR UNIVERSAL")
print("=" * 60)

# 2.1 Dataset ampliado a 10 puntos y 3 columnas: [Edad, Salario, Num_Hijos]
X_entrenamiento = np.array([
    [20, 30, 0],  # Punto A
    [40, 50, 2],  # Punto B
    [35, 45, 1],  # Punto C
    [18, 22, 0],  # Punto D
    [25, 35, 0],  # Punto E
    [50, 80, 3],  # Punto F
    [45, 60, 2],  # Punto G
    [22, 28, 1],  # Punto H
    [60, 90, 1],  # Punto I
    [30, 48, 2]   # Punto J
])

# Etiquetas: 0 = NO COMPRA, 1 = COMPRA
Y_entrenamiento = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1])

# Nuevo cliente a predecir
nuevo_cliente_3d = np.array([[30, 40, 1]])

# Experimento con K = 1
knn_k1 = KNeighborsClassifier(n_neighbors=1)
knn_k1.fit(X_entrenamiento, Y_entrenamiento)
pred_k1 = knn_k1.predict(nuevo_cliente_3d)

# Experimento con K = 5
knn_k5 = KNeighborsClassifier(n_neighbors=5)
knn_k5.fit(X_entrenamiento, Y_entrenamiento)
pred_k5 = knn_k5.predict(nuevo_cliente_3d)

print(f"Evaluando nuevo cliente [Edad: 30, Salario: 40k, Hijos: 1]:")
print(f"-> Predicción con n_neighbors = 1: {pred_k1[0]} ({'COMPRA' if pred_k1[0]==1 else 'NO COMPRA'})")
print(f"-> Predicción con n_neighbors = 5: {pred_k5[0]} ({'COMPRA' if pred_k5[0]==1 else 'NO COMPRA'})")


print("\n" + "=" * 60)
print("3. PREGUNTA DE ANÁLISIS: LA MALDICIÓN DE LA DIMENSIONALIDAD")
print("=" * 60)

respuesta_teorica = """
Si en lugar de 3 columnas tuviéramos 1,000 columnas (como píxeles de una imagen):

1.  La distancia euclidiana entre el punto más cercano 
   y el más lejano es prácticamente idéntica. Perdiendo la capacidad de 
   diferenciar quién es un verdadero 'vecino cercano'
2. El volumen del espacio crece exponencialmente y los puntos 
   quedan excesivamente aislados (espacio esparcido)
3.  El algoritmo pierde eficacia predictiva y su costo computacional 
   se dispara al calcular distancias en alta dimensión.
"""
print(respuesta_teorica)