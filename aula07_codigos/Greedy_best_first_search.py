graph = {
'Sibiu':[('Rimnicu Vilcea',193), ('Fagaras',176)],
'Fagaras':[('Bucharest',0)],
'Rimnicu Vilcea':[('Pitesti',100)],
'Pitesti':[('Bucharest',0)],
'Bucharest':[]
}
def bfs(start, target, graph, queue=[], visited=[]):
    if start not in visited:
        print(start)
        visited.append(start)
    queue=queue+[x for x in graph[start] if x[0][0] not in visited]
    queue.sort(key=lambda x:x[1])
    if queue[0][0]==target:
        print(queue[0][0])
    else:
        processing=queue[0]
        queue.remove(processing)
        bfs(processing[0], target, graph, queue, visited)
bfs('Sibiu', 'Bucharest', graph)