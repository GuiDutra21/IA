from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Income (Renda) 
# Renda = 0 representa renda baixa
# Renda = 1 representa renda alta

# Deposit (Depósito) 
# Deposito = 0 representa deposito pequeno
# Deposito = 1 representa deposito grande

# Payment (Pagamento)
# Pagamento = 0 representa pagamento em dia
# Pagamento = 1 representa que não pagou
# OBS: tambem achei estranho a ordem para pagamento kk

# Housing (Moradia)
# Moradia = 0 representa que não possui imovel
# Moradia = 1 representa que possui imovel

# Security (Garantia)
# Garantia = 0 representa nao deu garantia no emprestimo
# Garantia = 1 representa deu garantia no emprestimo

# Estrutura do grafo da rede
model = DiscreteBayesianNetwork([
    ("Renda", "Deposito"),
    ("Renda", "Pagamento"),
    ("Deposito", "Pagamento"),
    ("Moradia", "Garantia"),
    ("Pagamento", "Garantia"),
])

# Definindo CPDs (tabulares)
cpd_renda = TabularCPD('Renda', 2,[[0.7],[0.3]]) # Renda = 0, Renda = 1, nessa ordem

cpd_deposito = TabularCPD('Deposito',2,
    [
        [0.4,0.9], # Deposito = 0 | Renda = 0, Renda = 1
        [0.6,0.1]  # Deposito = 1 | Rennda = 0, Renda = 1
    ],
    evidence=['Renda'],
    evidence_card=[2]
)

cpd_pagamento = TabularCPD('Pagamento', 2,
    [
        # Pagamento = 0 | R=0 e D=0, R=0 e D=1, R=1 e D=0, R=1 e D=1  
        [0.4,0.55,0.5,0.95], 
        # Pagamento = 1 | R=0 e D=0, R=0 e D=1, R=1 e D=0, R=1 e D=1
        [0.6,0.45,0.5,0.05],
    ],
        evidence=['Renda','Deposito'],
        evidence_card = [2,2]
)

cpd_moradia = TabularCPD('Moradia',2,[[0.65],[0.35]]) # Moraria = 0, Moradia =1, nessa ordem

cpd_garantia = TabularCPD('Garantia',2,
    [
        # Garantia = 0 | M=0 e P=0, M=0 e P=1, M=1 e P=0, M=1 e P=1
        [0.69,0.25,0.5,0.99],
        # Garantia = 1 | M=0 e P=0, M=0 e P=1, M=1 e P=0, M=1 e P=1
        [0.31,0.75,0.5,0.01],
    ],
    evidence=["Moradia","Pagamento"],
    evidence_card=[2,2]
)

# Adicionando CPDs a rede
model.add_cpds(cpd_renda,cpd_deposito,cpd_pagamento,cpd_moradia, cpd_garantia)

# Checagem da integridade
# print(model.check_model())

# Inferencias
from pgmpy.inference import VariableElimination
infer = VariableElimination(model)

# Consulta 1
print('Probabilidade de uma pessoa ter renda alta ou baixa dado que faz o pagamento em dia:')
q = infer.query(variables=['Renda'], evidence={'Pagamento':0})
print(q)
print('Renda = 0 signfica renda baixa')
print('Renda = 1 signfica renda alta\n')

# Consulta 2
print('Probabilidade da pessoa pagar em dia ou nao dado que tem moradia e deu garantia no emprestimo:')
q = infer.query(variables=['Pagamento'], evidence={'Moradia':1,'Garantia':1})
print(q)
print('Pagamento = 0 representa pagamento em dia')
print('Pagamento = 1 representa que nao pagou\n')

# Consulta 3
print('Probabilidade da pessoa pagar em dia ou nao dado que tem moradia, mas nao deu garantia no emprestimo:')
q = infer.query(variables=['Pagamento'], evidence={'Moradia':1,'Garantia':0})
print(q)
print('Pagamento = 0 representa pagamento em dia')
print('Pagamento = 1 representa que nao pagou\n')

# Consulta 4
print('Probabilidade da pessoa pagar em dia ou nao dado que nao tem moradia, mas deu garantia no emprestimo:')
q = infer.query(variables=['Pagamento'], evidence={'Moradia':0,'Garantia':1})
print(q)
print('Pagamento = 0 representa pagamento em dia')
print('Pagamento = 1 representa que nao pagou')