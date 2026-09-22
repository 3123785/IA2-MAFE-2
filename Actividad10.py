import numpy as np


# PARTE 1: PERCEPTRÓN SIMPLE Y COMPUERTA LÓGICA OR


# 1. Definir la Función de Activación (Escalón)
def funcion_escalon(z):
    if z >= 0:
        return 1
    else:
        return 0

# 2. Definir la Estructura del Perceptrón
def perceptron(X, W, b):
    # Producto punto (Combinación lineal)
    Z = np.dot(X, W) + b
    # Activación
    salida = funcion_escalon(Z)
    return salida

# 3. Datos del problema y pesos ajustados para Compuerta OR
pesos = np.array([0.5, 0.5])  # Vector W
sesgo = -0.2                  # Constante b

# 4. Verificación de la Tabla de Verdad de la Compuerta OR
entradas_evaluar = [
    np.array([1, 1]),
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([0, 0])
]

print("=" * 60)
print("EVALUACIÓN COMPUERTA LÓGICA OR")
print("=" * 60)
for x in entradas_evaluar:
    res = perceptron(x, pesos, sesgo)
    print(f"Entrada: {x} -> El Perceptrón disparó el valor: {res}")



# PARTE 2: RED NEURONAL MULTICAPA (EVALUACIÓN EN LOTE / BATCH)


# Función de Activación: Sigmoide (devuelve un valor entre 0 y 1)
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# 1. ENTRADAS (X): Matriz de 2x3 para procesar 2 clientes en Lote (Batch)
X = np.array([
    [0.5, 0.8, 0.2],  # Cliente 1
    [0.1, 0.9, 0.9]   # Cliente 2
])

# 2. CAPA OCULTA (4 Neuronas)
# Matriz W1 de (3 entradas x 4 neuronas)
W1 = np.array([
    [0.1,  0.2, -0.3,  0.4],
    [-0.5, 0.6,  0.7, -0.8],
    [0.9, -0.1,  0.2,  0.3]
])
b1 = np.array([0.1, -0.2, 0.3, -0.4]) # 4 Sesgos

# --- PROCESO CAPA OCULTA ---
Z1 = np.dot(X, W1) + b1
A1 = sigmoide(Z1) # Salida de la capa oculta

print("\n" + "=" * 60)
print("RED NEURONAL MULTICAPA")
print("=" * 60)
print(" VALORES PUROS (Z1) ")
print(Z1)
print("\n VALORES TRANSFORMADOS POR SIGMOIDE (A1) en rango (0, 1) ")
print(A1)

# 3. CAPA DE SALIDA (1 Neurona)
# Matriz W2 ajustada a (4 entradas ocultas x 1 neurona final)
W2 = np.array([
    [0.5],
    [-0.6],
    [0.7],
    [0.8]
])
b2 = np.array([-0.1])

#  PROCESO CAPA FINAL 
Z2 = np.dot(A1, W2) + b2
Salida_Final = sigmoide(Z2)

print("\n PREDICCIÓN FINAL EN LOTE (PROBABILIDADES) ")
for idx, proba in enumerate(Salida_Final, start=1):
    print(f"Cliente {idx}: Probabilidad = {np.round(proba[0], 4)}")