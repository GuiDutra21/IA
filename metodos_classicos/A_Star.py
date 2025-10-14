import numpy as np
inf = np.inf

class no():
    
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.vizinhos = {}
        self.custo = [inf, 0] # funcao g(n) e heuristica
        self.caminho = ''
        self.indice = -1 # Indice a heap
    
    def get_custo(self):
        return self.custo[0] + self.custo[1]

    def add_vizinho(self, vizinho, custo):
        """Adiciona uma aresta entre este nó e outro, com o custo especificado """
        self.vizinhos[vizinho] = custo
        
    def __repr__(self):
        return f"{self.id}({self.x}, {self.y}) {self.custo} {self.caminho}"

# Funcao de comparacao dos custos
def menor(no1, no2):
    if no1.get_custo() < no2.get_custo():
        return True
    return False

# distancia euclidiana (distância de Manhattan) entre dois pontos
def distancia(no1, no2):
    dx = no1.x - no2.x
    dy = no1.y - no2.y
    return round((dx*dx + dy*dy) ** 0.5, 4)

def swap(heap, no1, no2):
    """Troca duas posições na heap e atualiza .indice"""
    i, j = no1.indice, no2.indice
    heap[i], heap[j] = heap[j], heap[i]
    heap[i].indice = i # Atualiza os indices mannualmente
    heap[j].indice = j

    
def fiuxUp(heap, no):
    """Corrige a posição do nó na heap, subindo-o enquanto o custo for menor que o do pai."""
    indice = no.indice
    while(indice > 1 and menor(heap[indice], heap[(indice//2)])):
        swap(heap, heap[indice//2], heap[indice])
        indice //= 2

def fixDown(heap, no):
    """Corrige a posição do nó na heap, descendo-o enquanto o custo for maior que o do menor filho."""
    indice = no.indice
    tam_heap = len(heap)
    
    # tem filho?
    while(2 * indice <= tam_heap - 1):
        filho = 2 * indice
      
        # Descobre o maior filho
        if filho < tam_heap - 1 and menor(heap[filho + 1],heap[filho]):
            filho += 1 # Filho a direita
        
        # O filho eh maior, entao pode parar
        if menor(heap[indice],heap[filho]):
            return

        swap(heap, heap[filho], heap[indice])
        indice = filho

def remove(heap):
    swap(heap, heap[1], heap[len(heap) - 1])
    retorno = heap.pop()
    if len(heap) > 1:
        fixDown(heap, heap[1])
    return retorno
    
def atualiza(heap, no, novo_custo):
    no.custo[0] = round(novo_custo,4)
    fiuxUp(heap, no)

def a_star(heap, partida, chegada):
        atualiza(heap, partida, 0)
        visitados = []
        while True:
            atual = remove(heap)
            if atual is None or atual == chegada:
                break
            
            # visitados.append(atual)
            for node, custo_aresta in atual.vizinhos.items():
                novo_custo = atual.get_custo() + custo_aresta
                caminho_acumulado = atual.custo[0] + custo_aresta
                
                if node not in visitados and novo_custo < node.get_custo():
                    atualiza(heap, node, caminho_acumulado)
                    node.caminho = atual.caminho + atual.id + ' -> '
            

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
    
    A.add_vizinho(B, 1)
    A.add_vizinho(C, 2.2361)

    B.add_vizinho(A, 1)
    B.add_vizinho(D, 2.2361)

    C.add_vizinho(A, 2.2361)
    C.add_vizinho(F, 2.2361)
    C.add_vizinho(E, 4)

    D.add_vizinho(B, 2.2361)
    D.add_vizinho(I, 2.2361)
    D.add_vizinho(F, 5.3852)
    D.add_vizinho(G, 3.1623)

    E.add_vizinho(C, 4)
    E.add_vizinho(H, 3.1623)
    E.add_vizinho(K, 4)

    F.add_vizinho(C, 2.2361)
    F.add_vizinho(D, 5.3852)
    F.add_vizinho(H, 2.2361)

    G.add_vizinho(I, 2.2361)
    G.add_vizinho(J, 1.4142)
    G.add_vizinho(K, 4.1231)
    G.add_vizinho(D, 3.1623)

    H.add_vizinho(F, 2.2361)
    H.add_vizinho(E, 3.1623)
    H.add_vizinho(K, 1.4142)
    H.add_vizinho(J, 2.2361)

    I.add_vizinho(D, 2.2361)
    I.add_vizinho(G, 2.2361)

    J.add_vizinho(G, 1.4142)
    J.add_vizinho(H, 2.2361)

    K.add_vizinho(H, 1.4142)
    K.add_vizinho(G, 4.1231)
    K.add_vizinho(E, 4)

    pontos = [A, B, C, D, E, F, G, H, I, J, K]
    heap = [None, A, B, C, D, E, F, G, H, I, J, K]
    
    for i, node in enumerate(heap):
        if node != None:
            node.indice = i
    
    partida = input("Informe o ponto de partida OBS (Deve estar entre A e K): ").strip().upper()
    chegada = input("Informe o ponto de chegada OBS (Deve estar entre A e K): ").strip().upper()
    partida = heap[ord(partida) - ord('A') + 1]
    chegada = heap[ord(chegada) - ord('A') + 1]
    
    # Calcula a distancia de todos os pontos para o ponto de chegada
    for node in heap:
        if node != None:
            node.custo[1] = distancia(node,chegada)
    
    # for p in pontos:
    #     print(p)
    
    a_star(heap, partida, chegada)

    # for i in pontos:
    #     print(i)
    
    print(f"\nO caminho pecorrido foi: {chegada.caminho + chegada.id} com custo {chegada.custo[0]}\n")
    
    caminho_ids = (chegada.caminho + chegada.id).split(' -> ')
    print("Custo de cada nó do caminho:")
    for id_no in caminho_ids:
        no_atual = pontos[ord(id_no) - ord('A')]
        if no_atual:
            print(f"{no_atual.id}: g(n) = {no_atual.custo[0]:.4f} | h(n) = {no_atual.custo[1]:.4f} | f(n) = {no_atual.get_custo():.4f}")