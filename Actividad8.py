import numpy as np
from sklearn.svm import SVC

# 1. TALLER ANALÍTICO: DIBUJANDO EL MARGEN
# Definí los datos del taller analítico en mi cuaderno
X_analitico = np.array([[2, 2], [3, 3], [4, 2], [6, 6], [7, 8], [8, 7]])
Y_analitico = np.array([0, 0, 0, 1, 1, 1])

# Entrené este modelo lineal simple para validar los vectores de soporte
svm_analitico = SVC(kernel='linear')
svm_analitico.fit(X_analitico, Y_analitico)

print(" 1. TALLER ANALÍTICO ")
print("Vectores de soporte calculados por el modelo:\n", svm_analitico.support_vectors_)

# Mis respuestas al taller analítico:
# 1 y 2. Al trazar la línea recta con regla, los vectores de soporte que sostienen
#        el margen son (3,3) y (4,2) de la Clase A, junto con (6,6) de la Clase B.
# 3. Si agrego el punto (1,1) de la Clase A, la línea NO cambia de posición.
#    Esto pasa porque el punto está lejos de la frontera y el modelo SVM solo
#    utiliza los Vectores de Soporte para definir el hiperplano.

# 2. TALLER DE LABORATORIO: FRONTERAS NO LINEALES
print("\n--- 2. TALLER DE LABORATORIO ---")

# Paso 1: Código base probando el punto [5,4]
modelo_svm = SVC(kernel='linear')
modelo_svm.fit(X_analitico, Y_analitico)

nuevo_punto = np.array([[5, 4]])
pred_base = modelo_svm.predict(nuevo_punto)
print("Paso 1 - Vectores de soporte base:\n", modelo_svm.support_vectors_)
print(f"Paso 1 - Predicción para el punto [5,4]: Clase {pred_base[0]}")

# Pasos 2 y 3: Agregué el punto [5,5] etiquetado como Clase 0
X_alterado = np.vstack([X_analitico, [5, 5]])
Y_alterado = np.append(Y_analitico, 0)

svm_lineal = SVC(kernel='linear')
svm_lineal.fit(X_alterado, Y_alterado)
pred_lineal = svm_lineal.predict(nuevo_punto)
print(f"Paso 3 - Predicción con Kernel Lineal (con el punto [5,5]): Clase {pred_lineal[0]}")

# Paso 4: Cambié el parámetro a kernel 'rbf'
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_alterado, Y_alterado)
pred_rbf = svm_rbf.predict(nuevo_punto)
print(f"Paso 4 - Predicción con Kernel RBF (con el punto [5,5]): Clase {pred_rbf[0]}")

# 3. MI REFLEXIÓN DE LA PREGUNTA

"""
Respuesta a la reflexión:
En mi opinión, un kernel lineal falla cuando las clases no se pueden separar con una 
recta porque una clase está encerrada o rodeada por la otra.

Ejemplos reales donde falla el lineal y se necesita RBF:
- Medicina: Al analizar imágenes para detectar tumores, el tejido enfermo (tumor) 
  suele estar rodeado completamente por tejido sano. Un corte recto no funcionaría, 
  mientras que RBF permite rodear y aislar la zona afectada.
- Reconocimiento Facial: Los cambios de luz y de expresión hacen que las características 
  no sean linealmente separables, por lo que se requieren fronteras curvas.
"""