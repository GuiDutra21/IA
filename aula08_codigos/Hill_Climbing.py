def func(x):
    return -x**3 +3*x**2 + 4*x + 1

def hill_climbing(start, neighbors):
    current = start
    current_value = func(current)
    
    while True:
        next_move = None
        next_value = current_value
        
        for n in neighbors(current):
            value = func(n)
            if value > next_value:
                next_value = value
                next_move = n
        
        if next_move is None:  # nenhum vizinho melhor encontrado
            break
        
        current = next_move
        current_value = next_value
    
    return current, current_value

def neighbors(x):
    # vizinhos inteiros dentro do intervalo [0,20]
    candidate = []
    if x - 1 >= 0:
        candidate.append(x - 1)
    if x + 1 <= 20:
        candidate.append(x + 1)
    return candidate

# Executa o Hill Climbing a partir do 10
start = 10
solution, value = hill_climbing(start, neighbors)
print(f"Melhor solução: x = {solution}, valor = {value}")
