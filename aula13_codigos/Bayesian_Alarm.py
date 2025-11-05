from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Estrutura do grafo da rede
model = DiscreteBayesianNetwork([
    ("Roubo", "Alarme"),
    ("Terremoto", "Alarme"),
    ("Alarme", "JohnLiga"),
    ("Alarme", "MaryLiga")
])

# Definindo CPDs (tabulares)
cpd_roubo = TabularCPD("Roubo", 2, [[0.999], [0.001]]) # Roubo = 0, Roubo = 1 nessa ordem
cpd_terremoto = TabularCPD("Terremoto", 2, [[0.998], [0.002]]) # 
cpd_alarme = TabularCPD("Alarme", 2,
    [
        [0.999, 0.71, 0.06, 0.05], # Alarme = 0 | R=0 e T=0, R=0 e T=1, R=1 e T=0, R=1 e T=1
        [0.001, 0.29, 0.94, 0.95], # Alarme = 1 | R=0 e T=0, R=0 e T=1, R=1 e T=0, R=1 e T=1
    ],   
    evidence=["Roubo", "Terremoto"],
    evidence_card=[2, 2])

# John: primeiro linha = P(John=0 | A=...), segunda = P(John=1 | A=...)
# Desejamos P(John=1|A=1)=0.90 e P(John=1|A=0)=0.05
cpd_johnliga = TabularCPD("JohnLiga", 2,
    [[0.95, 0.10],   # John=0 | A=0, A=1
     [0.05, 0.90]],  # John=1 | A=0, A=1
    evidence=["Alarme"],
    evidence_card=[2])

# Mary: desejamos P(M=1|A=1)=0.70 e P(M=1|A=0)=0.01
cpd_maryliga = TabularCPD("MaryLiga", 2,
    [[0.99, 0.30],   # Mary=0 | A=0, A=1
     [0.01, 0.70]],  # Mary=1 | A=0, A=1
    evidence=["Alarme"],
    evidence_card=[2])

# Adicionando CPDs à rede
model.add_cpds(cpd_roubo, cpd_terremoto, cpd_alarme, cpd_johnliga, cpd_maryliga)

# Checagem da integridade
print(model.check_model())


############################

from pgmpy.inference import VariableElimination
infer = VariableElimination(model)
resultado = infer.map_query(['Roubo'], evidence={'JohnLiga': 1, 'MaryLiga': 1})
print(resultado)
resultado = infer.query(['Roubo'], evidence={'JohnLiga': 1, 'MaryLiga': 1})
print(resultado)
