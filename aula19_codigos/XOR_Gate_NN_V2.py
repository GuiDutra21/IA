# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 10:51:36 2025

@author: Fabiano Soares
Exemplo simples de uma Feedforward Neural Network com backpropagation para 
resolver o problema da porta XOR.
"""

import numpy as np
import matplotlib.pyplot as plt  # Para visualização do erro

# Definição das funções sigmoid e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Entradas XOR e rótulos desejados
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inicialização dos pesos e bias com semente para reprodutibilidade
np.random.seed(42)
input_size = 2
hidden_size = 2
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Hiperparâmetros
lr = 0.1
epochs = 10000
history_loss = []


"""
Código para mostrar a arquitetura
"""
# Mostrar a Arquitetura

from graphviz import Digraph

def plot_xor_network():
    dot = Digraph(comment='Feedforward Neural Network XOR')

    # Camada de entrada
    dot.node('X1', 'Entrada 1')
    dot.node('X2', 'Entrada 2')

    # Camada oculta
    dot.node('H1', 'Oculta 1')
    dot.node('H2', 'Oculta 2')

    # Camada de saída
    dot.node('O1', 'Saída')

    # Conexões entrada → oculta
    dot.edge('X1', 'H1')
    dot.edge('X1', 'H2')
    dot.edge('X2', 'H1')
    dot.edge('X2', 'H2')

    # Conexões oculta → saída
    dot.edge('H1', 'O1')
    dot.edge('H2', 'O1')

    # Renderiza e mostra o grafo (em ambiente local use view=True para abrir)
    dot.render('xor_network', view=False, format='png')  # Gera 'xor_network.png'
    from IPython.display import Image, display
    display(Image('xor_network.png'))

# Chame a função antes do treinamento ou no início da célula/script
plot_xor_network()

"""
Fim do código para mostrar a arquitetura
"""

for epoch in range(epochs):
    # FORWARD PASS
    z1 = np.dot(X, W1) + b1         # Entrada da camada oculta
    a1 = sigmoid(z1)                # Saída da camada oculta (após ativação)
    z2 = np.dot(a1, W2) + b2        # Entrada da camada de saída
    a2 = sigmoid(z2)                # Saída da rede (previsão)

    # Cálculo do erro (Função de Custo MSE)
    loss = np.mean((y - a2) ** 2)
    history_loss.append(loss)        # Armazena o erro para plot

    # BACKPROPAGATION --------------------------------------------------------
    # Gradiente da camada de saída
    error_output = a2 - y
    delta2 = error_output * sigmoid_deriv(a2)

    # Gradiente da camada oculta
    error_hidden = np.dot(delta2, W2.T)
    delta1 = error_hidden * sigmoid_deriv(a1)

    # Atualização dos parâmetros (Descida do Gradiente)
    W2 -= lr * np.dot(a1.T, delta2)
    b2 -= lr * np.sum(delta2, axis=0, keepdims=True)
    W1 -= lr * np.dot(X.T, delta1)
    b1 -= lr * np.sum(delta1, axis=0, keepdims=True)

    # Exibe detalhes do gradiente e erro a cada 2000 épocas
    if epoch % 2000 == 0:
        print(f"--- Época {epoch} ---")
        print(f"Erro MSE: {loss:.5f}")
        print(f"Gradientes da saída (delta2):\n{delta2}")
        print(f"Gradientes camada oculta (delta1):\n{delta1}\n")

# Visualização da Convergência do Erro
plt.plot(history_loss)
plt.title('Evolução do Erro (MSE) durante o Treinamento')
plt.xlabel('Época')
plt.ylabel('Erro Médio Quadrático (MSE)')
plt.grid(True)
plt.show()

# Resultados finais
print("\nResultados finais na predição do XOR:")
for i, sample in enumerate(X):
    z1 = np.dot(sample, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    print(f"Entrada: {sample} -> Saída prevista: {a2[0][0]:.3f} (Label: {y[i][0]})")


