import numpy as np
# ACTIVIDAD 1 - PYTHON Y ALGEBRA LINEAL

# TALLER ANALITICO 1: INDEXACION Y TENSORES

A = np.array([
    [0, 255, 255, 255, 0],
    [255, 0, 0, 0, 255],
    [255, 0, 128, 0, 255],
    [255, 0, 0, 0, 255],
    [0, 255, 255, 255, 0]
])

valor = A[2, 3]

print("TALLER ANALITICO 1")
print("Valor de A[2,3]:", valor)

alto = 1080
ancho = 1920
canales = 3

cantidad = alto * ancho * canales

print("Cantidad de valores del tensor RGB:", cantidad)



# TALLER DE LABORATORIO 1: TRANSFORMACIONES AFINES


matriz = np.random.randint(200, 255, (5, 5))

alpha = 0.5
beta = -50

matriz_procesada = (alpha * matriz) + beta

matriz_procesada = np.clip(
    matriz_procesada,
    0,
    255
).astype(np.uint8)

print("TALLER DE LABORATORIO 1")

print("Matriz original:")
print(matriz)

print("Matriz procesada:")
print(matriz_procesada)

# TALLER ANALITICO 2: TRANSFORMACIONES

identidad = np.eye(4)

transpuesta = identidad.T

print("TALLER ANALITICO 2")

print("Matriz identidad:")
print(identidad)

print("Matriz transpuesta:")
print(transpuesta)


# Imagen RGB de 200 x 200 x 3
imagen = np.zeros((200, 200, 3))

vector = imagen.flatten()

print("Forma de la imagen:", imagen.shape)
print("Forma del vector:", vector.shape)
print("Cantidad de neuronas:", vector.size)



# TALLER DE LABORATORIO FINAL: PROGRAMANDO UN KERNEL


I = np.array([
    [100, 100, 100],
    [100, 200, 100],
    [100, 100, 100]
])

K = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

resultado = I * K

valor_central = np.sum(resultado)

print("\nTALLER DE LABORATORIO FINAL")

print("Sección de imagen:")
print(I)

print("\nKernel:")
print(K)

print("\nProducto Hadamard:")
print(resultado)

print("\nValor del pixel central:", valor_central)