from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

model = DiscreteBayesianNetwork([
    ("VisitaAsia", "Tuberculose"),
    ("Fumante", "Bronquite"),
    ("Fumante", "CancerPulmao"),
    ("Tuberculose", "DoencaPulmonar"),
    ("CancerPulmao", "DoencaPulmonar"),
    ("DoencaPulmonar", "RaioX"),
    ("Bronquite", "Tosse"),
    ("DoencaPulmonar", "Tosse")
])

# Probabilidade de ter visitado Ásia recente (baixa)
cpd_asia = TabularCPD("VisitaAsia", 2, [[0.99], [0.01]])  # 0: não visitou (99%), 1: visitou (1%)

# Probabilidade de ter tuberculose dado viagem à Ásia
cpd_tub = TabularCPD("Tuberculose", 2,
    [[0.999, 0.95],   # 0: não tem tuberculose → 95% se visitou Ásia, 99.9% se não
     [0.001, 0.05]],  # 1: tem tuberculose → 5% se visitou Ásia, 0.1% se não
    evidence=["VisitaAsia"], evidence_card=[2])

# Probabilidade de ser fumante (20% da população adulta)
cpd_fumante = TabularCPD("Fumante", 2, [[0.8], [0.2]]) # 0: não é fumante (80%), 1: é fumante (20%)

# Risco de câncer de pulmão dado tabagismo
cpd_cancer = TabularCPD("CancerPulmao", 2,
    [[0.998, 0.95],   # 0: não tem câncer → 99.8% não fumante, 95% fumante
     [0.002, 0.05]],  # 1: câncer → 0.2% não fumante, 5% fumante
    evidence=["Fumante"], evidence_card=[2])

# Probabilidade de bronquite dado tabagismo
cpd_bronc = TabularCPD("Bronquite", 2,
    [[0.97, 0.85],   # 0: não tem bronquite → 97% não fumante, 85% fumante
     [0.03, 0.15]],  # 1: tem bronquite → 3% não fumante, 15% fumante
    evidence=["Fumante"], evidence_card=[2])

# Doença pulmonar (verdadeira se tuberculose ou câncer, lógica OU)
cpd_dp = TabularCPD("DoencaPulmonar", 2,
     # Colunas: nTnC | TnC | nTC | TC
    [[1, 0, 0, 0],     # 0: Não tem doença
     [0, 1, 1, 1]],    # 1: Tem doença
    evidence=["Tuberculose", "CancerPulmao"],
    evidence_card=[2, 2])

# Raio-X alterado se doença pulmonar (alta sensibilidade)
cpd_raiox = TabularCPD("RaioX", 2,
    [[0.98, 0.05],      # 0: RaioX normal → 98% sem doença, 5% com doença
     [0.02, 0.95]],     # 1: RaioX alterado → 2% sem doença, 95% com doença
    evidence=["DoencaPulmonar"], evidence_card=[2])

# Tosse: comum em bronquite + doença pulmonar
cpd_tosse = TabularCPD("Tosse", 2,
    [[0.95, 0.15, 0.10, 0.05],  # 0: Sem tosse
     [0.05, 0.85, 0.90, 0.95]], # 1: Com tosse
    evidence=["Bronquite", "DoencaPulmonar"], evidence_card=[2, 2])

model.add_cpds(
    cpd_asia, cpd_tub, cpd_fumante, cpd_cancer, cpd_bronc,
    cpd_dp, cpd_raiox, cpd_tosse
)

print(model.check_model())



#############################################################################
## Inferências
from pgmpy.inference import VariableElimination
infer = VariableElimination(model)

# Probabilidade marginal (ex: ter tuberculose sem evidências):
resultado = infer.query(variables=["Tuberculose"])
print(resultado)

# Adicionando evidências observadas (ex: paciente tem tosse e RaioX alterado):
resultado = infer.query(variables=["Tuberculose"], evidence={"Tosse": 1, "RaioX": 1})
print(resultado)

# Adicionando evidências observadas (ex: paciente tem tosse, RaioX alterado e Visita a Ásia verdadeiro):
resultado = infer.query(variables=["Tuberculose"], evidence={"Tosse": 1, "RaioX": 1, "VisitaAsia": 1})
print(resultado)


# Fazer MAP Query (probabilidade máxima a posteriori) para diagnóstico mais provável:
map_result = infer.map_query(
    variables=["Tuberculose", "CancerPulmao", "Bronquite"],
    evidence={"Tosse": 1, "RaioX": 1}
)
print(map_result)

# Qual a probabilidade de tuberculose dado que o paciente visitou a Ásia?
resultado = infer.query(variables=["Tuberculose"], evidence={"VisitaAsia": 1})
print(resultado)

# Qual a probabilidade de câncer dado que é fumante e RaioX alterado?
resultado = infer.query(variables=["CancerPulmao"], evidence={"Fumante": 1, "RaioX": 1})
print(resultado)

# Qual o diagnóstico mais provável, dado tosse e raio-x normais?
print(infer.map_query(
    variables=["Tuberculose", "CancerPulmao", "Bronquite"],
    evidence={"Tosse": 1, "RaioX": 0}
))
