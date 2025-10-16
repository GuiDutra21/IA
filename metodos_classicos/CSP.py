TAM_TABULEIRO = 8

def csp_damas():
    variaveis = list(range(TAM_TABULEIRO))  # Representam as colunas
    dominios = {v: list(range(TAM_TABULEIRO)) for v in variaveis}  # Linhas possíveis

    def atende_restricao(atribuicoes, var, valor):
        """Verifica se a atribuição é consistente com as restrições."""
        
        for outra_var, outra_valor in atribuicoes.items():    
            # Mesma linha
            if valor == outra_valor:
                return False
            
            # Mesma diagonal, verifica se a diferenca entre as linhas e as colunas sao iguais
            if abs(valor - outra_valor) == abs(var - outra_var):
                return False
            
        return True

    def forward_checking(dominios, var, valor):
        """Remove valores inconsistentes dos domínios das variáveis não atribuidas"""
        
        novos_dominios = {v: list(vals) for v, vals in dominios.items()}
        
        for v in novos_dominios:
            
            # Remove valores que violam a mesma linha
            if v != var and valor in novos_dominios[v]:
                novos_dominios[v].remove(valor)
        
            # Remove valores que violam a diagonal
            for val in list(novos_dominios[v]):
                if abs(valor - val) == abs(var - v):
                    novos_dominios[v].remove(val)
        
        return novos_dominios

    def escolher_var_MRV(atribuicoes, dominios):
        """Escolhe a variável não atribuída com menor número de valores restantes"""
        
        nao_atribuidas = [v for v in dominios if v not in atribuicoes]
        
        return min(nao_atribuidas, key=lambda v: len(dominios[v]))

    def backtrack(atribuicoes, dominios):
        """Realiza as atribuicoes e quando encontra um cenario incoscistente retorna"""
        
        # Caso base: todas atribuidas
        if len(atribuicoes) == len(variaveis):
            return atribuicoes

        var = escolher_var_MRV(atribuicoes, dominios)
        
        for valor in list(dominios[var]):
            
            if atende_restricao(atribuicoes, var, valor):
                novas_atribuicoes = atribuicoes.copy()
                novas_atribuicoes[var] = valor

                novos_dominios = forward_checking(dominios, var, valor)
                
                # Checa falha do FC
                if any(len(vals) == 0 for v, vals in novos_dominios.items() if v not in novas_atribuicoes):
                    continue

                resultado = backtrack(novas_atribuicoes, novos_dominios)
                if resultado is not None:
                    return resultado

        return None

    solucao = backtrack({}, dominios)
    return solucao


if __name__ == "__main__":
    
    sol = csp_damas()
    
    print("Solução:")
    
    for linha, col in sol.items():
        print(f"Coluna {linha + 1} → Linha {col + 1}")