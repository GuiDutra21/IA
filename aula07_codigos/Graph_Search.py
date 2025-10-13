# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 19:29:10 2025

@author: soares
"""

from collections import deque

def graph_search(graph, start, goal):
    visited = set()
    queue = deque([(start, [start])])  # (nó_atual, caminho_até_agora)
    
    while queue:
        current_node, path = queue.popleft()
        
        if current_node == goal:
            return path
        
        visited.add(current_node)
        
        for neighbor in graph.get(current_node, []):
            if neighbor not in visited and all(neighbor != n for n, _ in queue):
                queue.append((neighbor, path + [neighbor]))
    
    return None  # caminho não encontrado

# Exemplo de grafo com ciclos
graph = {
    'A': ['B'],
    'B': ['C', 'D'],
    'C': ['D'],
    'D': ['A']  # ciclo para A
}

start_node = 'A'
goal_node = 'D'

path = graph_search(graph, start_node, goal_node)
print(path)  # Saída: ['A', 'B', 'D']
