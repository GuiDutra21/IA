import numpy as np
inf = np.inf

class no():
    def __init__(self, id):
        self.id = id
        self.vizinhos = {} # map com os nós vizinhos e seus custos
        self.custo = inf
        self.caminho = ''
        self.indice = -1
        
    def add_vizinho(self, vizinho, custo):
        """Adiciona uma aresta entre este nó e outro, com o custo especificado."""
        self.vizinhos[vizinho] = custo

def swap(heap, no1, no2):
    """Troca duas posições na heap e atualiza .indice"""
    i, j = no1.indice, no2.indice
    heap[i], heap[j] = heap[j], heap[i]
    heap[i].indice = i # Atualiza os indices mannualmente
    heap[j].indice = j

    
def fiuxUp(heap, no):
    """Corrige a posição do nó na heap, subindo-o enquanto o custo for menor que o do pai."""
    indice = no.indice
    while(indice > 1 and heap[(indice//2)].custo > heap[indice].custo):
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
        if filho < tam_heap - 1 and heap[filho + 1].custo < heap[filho].custo:
            filho += 1 # Filho a direita
        
        if heap[indice].custo <= heap[filho].custo:
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
    no.custo = novo_custo
    fiuxUp(heap, no)

def UCS(heap, visitados, inicio):
    
    atualiza(heap,inicio,0)
    
    while len(heap) > 1:
        atual = remove(heap)
        if atual is None:
            break
        
        visitados.append(atual)
        
        for node, custo_aresta in atual.vizinhos.items():
            novo_custo = atual.custo + custo_aresta
            if node not in visitados and novo_custo < node.custo:
                atualiza(heap, node, novo_custo)
                node.caminho = atual.caminho + atual.id + ' -> ' 
        
if __name__ == "__main__":
    # criação dos nós
    A = no('A')
    B = no('B')
    C = no('C')
    D = no('D')
    E = no('E')
    F = no('F')
    G = no('G')
    H = no('H')
    I = no('I')

    # # lista de nós
    nos = [A, B, C, D, E, F, G, H, I]

    # adição das arestas (grafo não direcionado)
    A.add_vizinho(B, 4)
    B.add_vizinho(A, 4)

    A.add_vizinho(H, 8)
    H.add_vizinho(A, 8)

    B.add_vizinho(C, 8)
    C.add_vizinho(B, 8)

    B.add_vizinho(H, 11)
    H.add_vizinho(B, 11)

    C.add_vizinho(D, 7)
    D.add_vizinho(C, 7)

    C.add_vizinho(I, 2)
    I.add_vizinho(C, 2)

    C.add_vizinho(F, 4)
    F.add_vizinho(C, 4)

    D.add_vizinho(E, 9)
    E.add_vizinho(D, 9)

    D.add_vizinho(F, 14)
    F.add_vizinho(D, 14)

    E.add_vizinho(F, 10)
    F.add_vizinho(E, 10)

    F.add_vizinho(G, 2)
    G.add_vizinho(F, 2)

    G.add_vizinho(H, 1)
    H.add_vizinho(G, 1)

    G.add_vizinho(I, 6)
    I.add_vizinho(G, 6)

    H.add_vizinho(I, 7)
    I.add_vizinho(H, 7)

    visitados = []
    
    # heap de minimo
    # Coloquei o None apenas para facilitar o acesso aos indices dos pais e dos filhos nos calulos das funcoes da heap
    heap = [None, A, B, C, D, E, F, G, H, I]
    
    for i, node in enumerate(heap):
        if(node == None):
            continue
        node.indice = i
          
    # for i in heap:
    #     if(i == None):
    #         continue
    #     print(f"{i.id} {i.indice} {i.custo} {i.caminho}")
    
    inicio = input("No inicial: ").strip().upper()
    if not inicio:
        raise ValueError("Entrada vazia")
    
    indice = ord(inicio) - ord('A') + 1
    
    if indice < 1 or indice >= len(heap):
        raise IndexError("Nó inicial fora do intervalo")
    UCS(heap, visitados, heap[indice])
    
    # debug = [A, B, C, D, E, F, G, H, I]
    # for i in debug:
    #         if(i == None):
    #             continue
    #         print(f"{i.id} {i.custo} {i.caminho}")
    # print()
    
    objetivo = input("No de destino: ").strip().upper()
    alvo = None
    for n in nos:
        if n.id == objetivo:
            alvo = n
            break
    if alvo is None:
        raise ValueError("Nó objetivo não encontrado")
    
    print(f"Custo de {inicio} ate {objetivo} foi de {alvo.custo} e o caminho foi {alvo.caminho}{objetivo}")
    