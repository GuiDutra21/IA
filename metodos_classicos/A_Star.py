import numpy as np
inf = np.inf

class no():
    heuristica = 0 # Distancia ate o no de chegada
    
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def __repr__(self):
        return f"{self.id}({self.x}, {self.y}) {self.heristica}"

# distância euclidiana (distância de Manhattan) entre dois pontos
def distacia(no1, no2):
    dx = no1.x - no2.x
    dy = no1.y - no2.y
    return round((dx*dx + dy*dy) ** 0.5, 4)

def a_star(partida, chegada, pontos):
    pontos_copia = pontos.copy()
    atual = partida
    caminho = ''
    prox = None
    caminho += f'{atual.id}'
    
    while True:
        pontos_copia.remove(atual)
        menor = inf
        
        for i in pontos_copia:
            funcao = distacia(atual, i) + i.heuristica 
            if funcao < menor:
                menor = funcao
                prox = i
        # caminho += f" {prox.id}"
        if prox == chegada:
            return caminho
        
        atual = prox
        caminho += f' -> {prox.id}'
        

if __name__ == "__main__":
    A = no('A', -3, 4)
    B = no('B', -3, 3)
    C = no('C', -1, 3)
    D = no('D', -2, 1)
    E = no('E', 3, 3)
    F = no('F', 1, 2)
    G = no('G', -1, -2)
    H = no('H', 2, 0)
    I = no('I', -3, -1)
    J = no('J', 0, -1)
    K = no('K', 3, -1)

    pontos = [A, B, C, D, E, F, G, H, I, J, K]

    print(distacia(I,G))
    
    # partida = input("Informe o ponto de partida OBS (Deve estar entre A e K): ").strip().upper()
    # chegada = input("Informe o ponto de chegada OBS (Deve estar entre A e K): ").strip().upper()
    # partida = pontos[ord(partida) - ord('A')]
    # chegada = pontos[ord(chegada) - ord('A')]
    # for i in pontos:
    #     i.heristica = distacia(i,chegada)
    
    # for p in pontos:
    #     print(p)
    
    # print(a_star(partida,chegada,pontos))
    