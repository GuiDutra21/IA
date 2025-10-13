import numpy as np
inf = np.inf


graph=[['A','B',1,3],
       ['A','C',2,4],
       ['A','H',7,0],
       ['B','D',4,2],
       ['B','E',6,6],
       ['C','F',3,3],
       ['C','G',2,1],
       ['D','E',7,6],
       ['D','H',5,0],
       ['F','H',1,0],
       ['G','H',2,0]]

nodes = ['A','B','C','D','E','F','G', 'H']
def A_star(graph, costs, frontier, visited, cur_node):
    if cur_node in frontier:
        frontier.remove(cur_node)
    visited.add(cur_node)
    for i in graph:
        if(i[0] == cur_node and costs[i[0]]+i[2]+i[3] < costs[i[1]]):
            frontier.add(i[1])
            costs[i[1]] =  costs[i[0]]+i[2]+i[3]
            path[i[1]] = path[i[0]] + ' -> ' + i[1]
    costs[cur_node] = inf
    frontier_min = min(costs, key=costs.get)
    if frontier_min not in visited:
        A_star(graph, costs, frontier, visited, frontier_min)
        
''' Inicio do programa'''
costs = dict()
temp_cost = dict()
path = dict()

'''define todos os custos como infinito inicialmente'''
for i in nodes:
    costs[i] = inf
    path[i] = ' '
frontier = set()
visited = set()

'''Entra com o estado de início e o estado alvo para fazer a buca'''
start_node = input("Enter the Start Node: ")
frontier.add(start_node)
path[start_node] = start_node
costs[start_node] = 0
A_star(graph, costs, frontier, visited, start_node)
goal_node = input("Enter the Goal Node: ")
print("Path with least cost is: ",path[goal_node])