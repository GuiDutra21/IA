import random

TAM_POPULACAO = 10
TAM_VETOR = 10
TAM_NUMEROS = 1000
QTD_SELECIONADOS = 4
TAXA_MUTACAO = 0.05

def gera_populacao():
    """ Gera populcao aletoria de tamanho TAM_POPULACAO"""
    populacao = []
    for _ in range(TAM_POPULACAO):
        temp = [random.randint(0, TAM_NUMEROS - 1) for _ in range(TAM_VETOR)]
        populacao.append(temp)
    return populacao

def eh_primo(n):
    """Retorna True se n for primo, False caso contrário."""
    if n < 2:
        return False
    
    if n != 2 and n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def fitness(individuo):
    """Mede a proximidade com a solucao (array so de primos)"""
    fit = 0
    fit = sum(1 for i in individuo if eh_primo(i))    
    return fit

def verifica_resposta(individuo):
    """Verifica se todos os elementos sao primoss"""
    return fitness(individuo) == len(individuo)

def selecao(populacao):
    """Seleciona QTD_SELECIONADOS indivíduos com maior fitness."""
    avaliados = [(ind, fitness(ind)) for ind in populacao]
    
    # Ordena do maior para o menor fitness
    avaliados.sort(key=lambda x: x[1], reverse=True)
    
    # Seleciona os QTD_SELECIONADOS melhores
    pais = [ind for ind, fit in avaliados[:QTD_SELECIONADOS]]
    return pais
    
def crossover(pais):
    """Realiza crossover entre dois pais escolhidos aleatoriamente"""
    p1, p2 = random.sample(pais, 2)
    ponto = random.randint(1, TAM_VETOR - 1)
    filho1 = p1[:ponto] + p2[ponto:]
    filho2 = p2[:ponto] + p1[ponto:]
    return [filho1, filho2]

def mutacao(individuo):
    """Muda aleatoriamente alguns genes."""
    for i in range(len(individuo)):
        if random.random() < TAXA_MUTACAO:
            individuo[i] = random.randint(0, TAM_NUMEROS - 1)
    return individuo

if __name__ == "__main__":
    populacao = gera_populacao()
    geracao = 0
    melhor = populacao[0]
    fit = 0

    while True:
        geracao += 1

        # Avalia a população
        populacao.sort(key=fitness, reverse=True)
        fit = fitness(populacao[0])
        
        # Atualiza o melhor apenas quando melhora
        if fit > fitness(melhor):
            melhor = populacao[0]
            print(f"Geração {geracao} | Fitness: {fit}")

            # Verifica se encontrou a solução
            if verifica_resposta(melhor):
                print("\nSolução encontrada!")
                print(melhor)
                break

        # Seleção
        pais = selecao(populacao)

        # Gera filhos
        filhos = []
        for _ in range(TAM_POPULACAO // 2):
            novos = crossover(pais)
            filhos.extend(novos)

        # Aplica mutação
        filhos = [mutacao(f) for f in filhos]

        # Nova geração
        populacao = filhos[:TAM_POPULACAO]