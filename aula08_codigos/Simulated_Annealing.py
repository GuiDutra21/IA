import math
import random

# Função objetivo: polinômio cúbico com máximos e mínimos locais
def func(x):
    return x**3 - 2*x**2 + 4*x + 1

# Algoritmo Simulated Annealing
def simulated_annealing(start, neighbors, initial_temp, cooling_rate, min_temp):
    current = start
    current_value = func(current)
    temperature = initial_temp
    
    while temperature > min_temp:
        # Escolhe um vizinho aleatório do estado atual
        next_move = random.choice(neighbors(current))
        next_value = func(next_move)
        
        delta = next_value - current_value
        
        if delta > 0:
            # Se o vizinho é melhor, aceita sempre
            current, current_value = next_move, next_value
        else:
            # Se pior, aceita com probabilidade que depende da temperatura
            prob = math.exp(delta / temperature)
            if random.random() < prob:
                current, current_value = next_move, next_value
        
        # Resfria gradualmente a temperatura
        temperature *= cooling_rate  
    
    return current, current_value

# Função geradora de vizinhos (vizinhança = estados adjacentes inteiros)
def neighbors(x):
    candidates = []
    if x - 1 >= 0:
        candidates.append(x - 1)
    if x + 1 <= 100:
        candidates.append(x + 1)
    return candidates

# Parâmetros ajustados para maior chance de máximo global
start = 20
initial_temp = 50.0   # temperatura inicial alta permite aceitar piores no início
cooling_rate = 0.95    # resfriamento lento para mais exploração
min_temp = 0.01

solution, value = simulated_annealing(start, neighbors, initial_temp, cooling_rate, min_temp)
print(f"Melhor solução encontrada: x = {solution}, valor = {value}")
