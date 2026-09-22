import numpy as np


# 1. TALLER ANALÍTICO: CALCULANDO EL DISPARO (PERCEPTRÓN SIMPLE)

def funcion_escalon(z):
    return 1 if z >= 0 else 0


def perceptron_credito(x1, x2, w1, w2, b):
    # 1. Cálculo del valor Z (Combinación lineal)
    z = (x1 * w1) + (x2 * w2) + b
    # 2. Evaluación en la función de activación
    salida = funcion_escalon(z)
    return z, salida

# Datos del problema de crédito
X1_ingresos = 50
X2_deudas = 20
W1 = 0.8
W2 = -0.5
b_sesgo = -10

z_calculado, disparo = perceptron_credito(X1_ingresos, X2_deudas, W1, W2, b_sesgo)

print("=" * 60)
print("1. TALLER ANALÍTICO: EVALUACIÓN DE CRÉDITO")
print("=" * 60)
print(f"Cálculo Forward (Z): {z_calculado}")
print(f"Salida de la Neurona: {disparo}")
print(f"Resultado: {'La neurona SÍ dispara (Aprobado)' if disparo == 1 else 'La neurona NO dispara (Rechazado)'}")

# Análisis del peso negativo W2
"""
ANÁLISIS DE W2 (-0.5):
- Los Ingresos (X1) incrementan la capacidad de pago del cliente (peso positivo W1 = 0.8).
- Las Deudas (X2) representan un riesgo financiero.
- Por ello, el peso negativo (W2 = -0.5) actúa como una penalización: a mayor deuda,
  menor es el valor de Z, reduciendo la probabilidad de aprobar el crédito.
"""

# 2. COMPUERTA LÓGICA OR (PERCEPTRÓN)

def perceptron(X, W, b):
    Z = np.dot(X, W) + b
    return funcion_escalon(Z)

pesos_or = np.array([0.5, 0.5])
sesgo_or = -0.2

entradas_or = [
    np.array([1, 1]),
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([0, 0])
]

print("\n" + "=" * 60)
print("2. EVALUACIÓN COMPUERTA LÓGICA OR")
print("=" * 60)
for x in entradas_or:
    res = perceptron(x, pesos_or, sesgo_or)
    print(f"Entrada: {x} -> Resultado: {res}")

# 3. LABORATORIO PRÁCTICO: RED NEONAL MULTICAPA EN LOTE (BATCH)


def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# 1. ENTRADAS (X): Matriz de 2x3 (2 clientes)
X_clientes = np.array([
    [0.5, 0.8, 0.2],  # Cliente 1
    [0.1, 0.9, 0.9]   # Cliente 2
])

# 2. CAPA OCULTA (4 Neuronas)
W1 = np.array([
    [0.1,  0.2, -0.3,  0.4],
    [-0.5, 0.6,  0.7, -0.8],
    [0.9, -0.1,  0.2,  0.3]
])
b1 = np.array([0.1, -0.2, 0.3, -0.4])

# Proceso Capa Oculta
Z1 = np.dot(X_clientes, W1) + b1
A1 = sigmoide(Z1)

# 3. CAPA DE SALIDA (1 Neurona)
W2 = np.array([
    [0.5],
    [-0.6],
    [0.7],
    [0.8]
])
b2 = np.array([-0.1])

# Proceso Capa Final
Z2 = np.dot(A1, W2) + b2
Salida_Final = sigmoide(Z2)

print("\n" + "=" * 60)
print("3. LABORATORIO PRÁCTICO: RED NEURONAL MULTICAPA")
print("=" * 60)
print("--- VALORES PUROS DE LA CAPA OCULTA (Z1) ---")
print(Z1)

print("\n--- VALORES TRANSFORMADOS POR SIGMOIDE (A1) ---")
print(A1)

print("\n--- PREDICCIÓN FINAL EN LOTE (PROBABILIDADES) ---")
for idx, proba in enumerate(Salida_Final, start=1):
    print(f"Cliente {idx}: Probabilidad = {np.round(proba[0], 4)}")