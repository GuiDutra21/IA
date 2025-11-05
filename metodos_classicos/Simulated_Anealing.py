import random
import math
import matplotlib.pyplot as plt

class item():
    """Representa um item da mochila com nome, peso e utilidade"""
    def __init__(self,nome,peso,utilidade):
        self.nome = nome
        self.peso = peso
        self.utilidade = utilidade
        
def solucao_inicial(items):
    """Gera uma solução inicial aleatória"""
    return [random.randint(0,1) for _ in range(len(items))]

def peso_total(solucao, items):
    """Calcula o peso total de uma solução"""
    return sum(items[i].peso * solucao[i] for i in range(len(items)))

def util(solucao, items):
    """Calcula a utilidade total sem considerar a capacidade"""
    return sum(items[i].utilidade * solucao[i] for i in range(len(items)))

def utilidade_total(solucao, items, capacidade):
    """Avalia a utilidade de uma solução com penalização se exceder a capacidade"""
    peso = peso_total(solucao,items)
    utilidade = util(solucao,items)
    if peso > capacidade:
        # Se ultrapassar o limite consideramos muito baixo o valor
        return -(utilidade + peso)
    return utilidade

def vizinho(solucao):
    """Gera uma solução vizinha por flip de um bit aleatório"""
    nova = solucao.copy()
    i = random.randint(0, len(solucao) - 1)
    nova[i] = 1 - nova[i]  # inverte um item aleatório
    return nova

def print_resultado(solucao, melhor_utilidade, items):
    """Imprime os itens escolhidos, peso total e utilidade"""
    print("Items da solucao encontrada:")
    for i in range(len(solucao)):
        if solucao[i]:
            print(f"{items[i].nome}, peso: {items[i].peso} e utilidade: {items[i].utilidade}")
    print(f"Peso total: {peso_total(solucao,items)}")
    print(f"Utilidade total: {melhor_utilidade}")
            

def simulated_annealing(temp_inicial, temp_min, taxa_resfriamento, items, capacidade):
    """Executa o algoritmo de Simulated Annealing para o problema da mochila"""
    atual = solucao_inicial(items)
    melhor = atual
    atual_utilidade = utilidade_total(atual,items, capacidade)
    melhor_utilidade = atual_utilidade
    temperatura = temp_inicial

    historico = []

    while temperatura > temp_min:
        # Gera um estado vizinho
        candidato = vizinho(atual)
        candidato_utilidade = utilidade_total(candidato, items, capacidade)
        
        delta = candidato_utilidade - atual_utilidade
        
        historico.append(atual_utilidade) # Apenas para mostrar o grafico no final
        
        if delta > 0:
            atual, atual_utilidade = candidato, candidato_utilidade
        
            if atual_utilidade > melhor_utilidade:
                melhor, melhor_utilidade = atual, atual_utilidade  # Aatualiza o melhor
        else:
            prob = math.exp(delta / temperatura)
            if random.random() < prob:
                atual, atual_utilidade = candidato, candidato_utilidade
        
        temperatura *= taxa_resfriamento  # Aplica o resfriamento

    return melhor, melhor_utilidade, historico
    
if __name__ == "__main__":

    items = [
        item("Barraca compacta", 4.0, 90),
        item("Saco de dormir leve", 3.0, 80),
        item("Fogareiro portátil", 2.0, 65),
        item("Garrafa térmica", 1.0, 30),
        item("Kit primeiros socorros", 1.5, 70),
        item("Lanterna + pilhas", 0.8, 40),
        item("Corda de escalada", 2.5, 50),
        item("GPS / bússola", 0.5, 60),
        item("Roupas térmicas", 2.0, 75),
        item("Comida extra", 3.0, 55),
    ]

    capacidade = 15.0  # kg

    melhor, melhor_utilidade, historico = simulated_annealing(
        temp_inicial=1000,
        temp_min=0.1,
        taxa_resfriamento=0.9,
        items=items,
        capacidade=capacidade
    )
    print_resultado(melhor, melhor_utilidade, items)
    
    # Mostra o grafico da evolucao da utilidade
    plt.plot(historico)
    plt.title("Evolução da utilidade ao longo do tempo")
    plt.xlabel("Iteração")
    plt.ylabel("Utilidade total")
    plt.grid(True)
    plt.show()