import numpy as np
inf = np.inf

graph=[['Sibiu','Fagaras',99],
       ['Sibiu','Rimnicu Vilcea',80],
       ['Fagaras','Bucharest',211],
       ['Rimnicu Vilcea','Pitesti',97],
       ['Pitesti','Bucharest',101]]

nodes = ['Sibiu','Fagaras','Rimnicu Vilcea','Pitesti','Bucharest']

def UCS(graph, costs, frontier, visited, cur_node):
  if cur_node in frontier:
    frontier.remove(cur_node)
  visited.add(cur_node)
  for i in graph:
    if(i[0] == cur_node and costs[i[0]]+i[2] < costs[i[1]]):
      frontier.add(i[1])
      costs[i[1]] = costs[i[0]]+i[2]
      path[i[1]] = path[i[0]] + ' -> ' + i[1]
  costs[cur_node] = inf #retorna o valor do custo para infinito
  frontier_city = min(costs, key=costs.get)
  if frontier_city not in visited:
    UCS(graph, costs, frontier, visited, frontier_city)


''' Inicio do programa'''

costs = dict()
path = dict()

'''define todos os custos como infinito inicialmente'''
for i in nodes:
  costs[i] = inf
  path[i] = ' '
  
  
frontier = set()
visited = set()

'''Entra com o estado de início e o estado alvo para fazer a busca'''
start_node = input("Enter the Start State: ")
frontier.add(start_node)
path[start_node] = start_node
costs[start_node] = 0
UCS(graph, costs, frontier, visited, start_node)
goal_node = input("Enter the Goal State: ")
print("Path with least cost is: ",path[goal_node])