from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Define the network structure
model = DiscreteBayesianNetwork([
    ('Cloudy', 'Sprinkler'), 
    ('Cloudy', 'Rain'),
    ('Sprinkler', 'WetGrass'), 
    ('Rain', 'WetGrass')
])

# 2. Define Conditional Probability Distributions (CPDs)
cpd_cloudy = TabularCPD(variable='Cloudy', variable_card=2,
                        values=[[0.5], [0.5]])

cpd_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
                           values=[[0.5, 0.9],
                                   [0.5, 0.1]],
                           evidence=['Cloudy'], evidence_card=[2])

cpd_rain = TabularCPD(variable='Rain', variable_card=2,
                      values=[[0.8, 0.2],
                              [0.2, 0.8]],
                      evidence=['Cloudy'], evidence_card=[2])

cpd_wet_grass = TabularCPD(variable='WetGrass', variable_card=2,
                           values=[[1.0, 0.1, 0.1, 0.01],
                                   [0.0, 0.9, 0.9, 0.99]],
                           evidence=['Sprinkler', 'Rain'], evidence_card=[2, 2])

# 3. Add CPDs to the model
model.add_cpds(cpd_cloudy, cpd_sprinkler, cpd_rain, cpd_wet_grass)

# 4. Check model validity
assert model.check_model(), "Modelo inválido!"

# 5. Perform inference (e.g., query probability of Rain given WetGrass)
inference = VariableElimination(model)
query_result = inference.query(variables=['Rain'], evidence={'WetGrass': 1}) # WetGrass=1 means True
print(query_result)
