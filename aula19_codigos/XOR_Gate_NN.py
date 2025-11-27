# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 10:51:36 2025

@author: Fabiano Soares
Exemplo simples de uma Feedforward Neural Network com backpropagation para 
resolver o problema da porta XOR.
"""

import numpy as np

# Função de ativação sigmoid e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Dados de entrada (XOR)
# quatro exemplos: entradas e respectivas saídas
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inicialização dos pesos (aleatória) e bias
np.random.seed(42)  # Para reprodutibilidade
input_size = 2
hidden_size = 2
output_size = 1

# Pesos para a camada oculta
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

# Pesos para a camada de saída
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Hiperparâmetros
lr = 0.1   # taxa de aprendizado
epochs = 10000

for epoch in range(epochs):
    # FORWARD PASS
    # Entrada -> Camada oculta
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    # Camada oculta -> Saída
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Cálculo do erro (MSE)
    loss = np.mean((y - a2) ** 2)

    # BACKPROPAGATION
    # Derivada do erro na saída
    error_output = a2 - y
    delta2 = error_output * sigmoid_deriv(a2)   # Gradiente camada de saída

    # Propaga o erro para a camada oculta
    error_hidden = np.dot(delta2, W2.T)
    delta1 = error_hidden * sigmoid_deriv(a1)   # Gradiente camada oculta

    # Atualização dos pesos e bias (descida do gradiente)
    W2 -= lr * np.dot(a1.T, delta2)
    b2 -= lr * np.sum(delta2, axis=0, keepdims=True)

    W1 -= lr * np.dot(X.T, delta1)
    b1 -= lr * np.sum(delta1, axis=0, keepdims=True)

    # Exibe o erro a cada 1000 épocas
    if epoch % 1000 == 0:
        print(f'Época {epoch} - Erro: {loss:.4f}')

# Teste final
print("\nTestando a rede...:")
for i, sample in enumerate(X):
    z1 = np.dot(sample, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    print(f"Entrada: {sample} -> Saída prevista: {a2[0][0]:.3f} (Label: {y[i][0]})")