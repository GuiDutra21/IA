import numpy as np

# P(X_t | X_t-1)
matriz_de_transicao = np.array([
    [0.4, 0.5, 0.1],  # Se estava Feliz → [Feliz, Neutro, Irritado]
    [0.2, 0.6, 0.2],  # Se estava Neutro → [Feliz, Neutro, Irritado]
    [0.1, 0.4, 0.5]   # Se estava Irritado → [Feliz, Neutro, Irritado]
])

# Probabilidade da criança ganhar brinquedo dado o humor
matriz_de_observacao_brinquedo = np.array([
    [0.65, 0, 0],    # Se pais Felizes: 65% chance de ganhar brinquedo
    [0, 0.4, 0],    # Se pais Neutros: 40% chance de ganhar brinquedo
    [0, 0, 0.2]     # Se pais Irritados: 20% chance de ganhar brinquedo
])

# Probabilidade da criança nao ganhar brinquedo dado o humor
matriz_de_observacao_nao_brinquedo = np.array([
    [0.35, 0, 0],    # Se pais Felizes: 35% chance de nao ganhar brinquedo
    [0, 0.6, 0],    # Se pais Neutros: 60% chance de nao ganhar brinquedo
    [0, 0, 0.8]     # Se pais Irritados: 80% chance de nao ganhar brinquedo
])

def forward(observacoes):
    """
    Algoritmo Forward (Filtragem) para calcular P(X_t | e_{1:t})
    Onde X_t é o humor dos pais e e_t eh se a crianca ganhou brinquedo
    """
    
    # Estados inicias(P(X0)) do humor dos pais
    probabilidade_inicial = np.array([0.25, 0.6, 0.15])  # Feliz = 25%, Neutro = 60%, Irritado = 15%
    
    lista_forward = [probabilidade_inicial]
    humor_t_menos_1_dado_brinquedo_1_t_menos_1 = probabilidade_inicial # P(x_t-1 | e_{1:t-1})
    
    print(f"Forward - Prior: {probabilidade_inicial}")
    
    for dia, observacao in enumerate(observacoes, 1):
        
        # Passo de Predição 
        # P(X_t | e_{1:t-1}) eh calculado multuplicando P(X_t | x_t-1) com P(x_t-1 | e_{1:t-1})
        humor_t_dado_brinquedo_1_t_menos_1 = np.dot(matriz_de_transicao, humor_t_menos_1_dado_brinquedo_1_t_menos_1)
        # print(f"Dia {dia} - P_redição (apos transicao): {humor_t_dado_brinquedo_1_t_menos_1}")
        
        # Passo de Atualizacao
        if observacao: # Se ganhou brinquedo P(X_t | e_{1:t}) = α P(e_t | X_t ) * P(X_t | e_{1:t-1})
            humor_t_dado_brinquedo_1_t = np.dot(matriz_de_observacao_brinquedo, humor_t_dado_brinquedo_1_t_menos_1)
            
        else: # Se nao ganhou brinquedo P(X_t | e_{1:t}) = α P(~e_t | X_t ) * P(X_t | e_{1:t-1})
            humor_t_dado_brinquedo_1_t = np.dot(matriz_de_observacao_nao_brinquedo, humor_t_dado_brinquedo_1_t_menos_1)
        
        # Normalização (calcula o α)
        humor_t_dado_brinquedo_1_t = humor_t_dado_brinquedo_1_t / humor_t_dado_brinquedo_1_t.sum()
        
        # print(f"Dia {dia} - Atualizado (apos observacao): {humor_t_dado_brinquedo_1_t}")
        print(f"P(Humor_{dia}|e_1:{dia}) = {humor_t_dado_brinquedo_1_t}")
        
        lista_forward.append(humor_t_dado_brinquedo_1_t)
        
        # Atualiza para a proxima iteracao
        humor_t_menos_1_dado_brinquedo_1_t_menos_1 = humor_t_dado_brinquedo_1_t
    
    return lista_forward

def backward(observacoes, print_output=True):
    """Algoritmo Backward para calcular P(e_{k+1:t} | X_k)"""
    
    # Reverte a lista, pois o backward executa do final para o incio
    observacoes = observacoes[::-1]
    
    # Valor incial
    beta = np.array([1.0, 1.0, 1.0])
    
    lista_backwards = [beta] # P(e_{k+1:t} | X_k)
    
    if print_output:
        print(f"Backwards {len(observacoes)}:{len(observacoes)} = {beta}")
    
    for i, observacao in enumerate(observacoes):
        
        if observacao:  # Se ganhou brinquedo temp = P(e_k | X_k) * P(e_{k+1:t} | X_k)
            temp = np.dot(matriz_de_observacao_brinquedo, beta)
            
        else:  # Se nao ganhou brinquedo temp = P(~e_k | X_k) * P(e_{k+1:t} | X_k)
            temp = np.dot(matriz_de_observacao_nao_brinquedo, beta)
        
        # b = temp *  P(X_t | X_t-1)
        b = np.dot(temp, matriz_de_transicao)
        
        if print_output:
            print(f"Backwards {len(observacoes)-i-1}:{len(observacoes)} = {b}")
        
        # Normalização
        b = b / b.sum()
        lista_backwards.append(b)
        
        # Atualiza para a proxima iteracao
        beta = b 
    
    return lista_backwards

def smoothing(forward_list, backward_list, total_dias):
    """ Algoritmo de Suavização: P(X_k | e_{1:t}) usando forward-backward """
    
    print("Smoothing:")
    # Volta a lista backward para a ordem crescente
    backward_list = backward_list[::-1]
    
    
    for i, b in enumerate(backward_list):
        
        # Combina forward e backward
        # P(X_k | e_{1:t}) = P(X_k | e_{1:k}) * P(e_{k+1:t} | X_k)
        humor_t_dado_brinquedo_1_t = np.multiply(forward_list[i], b)
        
        # Normalizacao
        humor_t_dado_brinquedo_1_t = humor_t_dado_brinquedo_1_t / humor_t_dado_brinquedo_1_t.sum()
        
        if i > 0:  # Pula o tempo 0
            print(f"P(Humor_{i}|e_1:{total_dias}) = {humor_t_dado_brinquedo_1_t}")

def executa_cenario(observacoes_semana):
    """ Apenas para chamar as funções e realizar os prints """
    # Executando Forward
    forward_list = forward(observacoes_semana)
    print()
    
    # Executando Backwards
    backward_list = backward(observacoes_semana)
    print()
    
    # Executando Smoothing
    smoothing(forward_list, backward_list, len(observacoes_semana))
    
    print()
    print("ANÁLISE DOS RESULTADOS:")
    
    # Análise final
    humor_final = forward_list[-1]
    estado_mais_provavel = np.argmax(humor_final)
    estados = ["Feliz", "Neutro", "Irritado"]
    
    print(f"Estado final mais provável: {estados[estado_mais_provavel]} ({humor_final[estado_mais_provavel]:.1%})")
    print(f"Distribuição final: Feliz {humor_final[0]:.1%}, Neutro {humor_final[1]:.1%}, Irritado {humor_final[2]:.1%}\n")


if __name__ == '__main__':
    print("CENARIO: HUMOR DOS PAIS vs BRINQUEDOS DA CRIANCA")
    print("Estados: [Feliz, Neutro, Irritado]")
    print("Observacoes: True=ganhou brinquedo, False=nao ganhou\n")
    
    # Observacao igual a True => ganhou brinquedo
    # Observacao igual a False => nao ganhou brinquedo

    # Cenario 1
    observacoes_semana_boa = [True, True, False, True, True] 

    # Cenario 2  
    observacoes_semana_ruim = [False, False, True, False, False] 

    # Cenario 3
    observacoes_semana_mista = [False,True,False,True,True,False] 
    
    print("Cenario 1")
    print(f"Sequência de observações: {observacoes_semana_boa}")
    print("(True = ganhou brinquedo, False = não ganhou)\n")
    executa_cenario(observacoes_semana_boa)
 
    print("Cenario 2")
    print(f"Sequência de observações: {observacoes_semana_ruim}")
    print("(True = ganhou brinquedo, False = não ganhou)\n")
    executa_cenario(observacoes_semana_ruim)
    
    print("Cenario 3")
    print(f"Sequência de observações: {observacoes_semana_mista}")
    print("(True = ganhou brinquedo, False = não ganhou)\n")
    executa_cenario(observacoes_semana_mista)