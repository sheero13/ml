#maxflow.py
# from collections import deque

# # BFS to find if there is a path from source to sink
# def bfs(residual_graph, source, sink, parent):
#     num_vertices = len(residual_graph)
#     visited = [False] * num_vertices
#     queue = deque()
#     queue.append(source)
#     visited[source] = True
#     parent[source] = -1

#     while queue:
#         u = queue.popleft()
#         for v in range(num_vertices):
#             if not visited[v] and residual_graph[u][v] > 0:
#                 queue.append(v)
#                 visited[v] = True
#                 parent[v] = u
#                 if v == sink:
#                     return True
#     return False

# # Edmonds-Karp algorithm to compute max flow
# def edmonds_karp(graph, source, sink):
#     num_vertices = len(graph)
#     residual_graph = [row[:] for row in graph]
#     parent = [0] * num_vertices
#     max_flow = 0

#     while bfs(residual_graph, source, sink, parent):
#         # Find minimum residual capacity along the path found by BFS
#         path_flow = float('inf')
#         s = sink
#         while s != source:
#             path_flow = min(path_flow, residual_graph[parent[s]][s])
#             s = parent[s]

#         # Add path flow to overall flow
#         max_flow += path_flow

#         # Update residual capacities
#         v = sink
#         while v != source:
#             u = parent[v]
#             residual_graph[u][v] -= path_flow
#             residual_graph[v][u] += path_flow
#             v = parent[v]

#     return max_flow

# # Example graph
# graph = [
#     [0, 16, 13, 0, 0, 0],
#     [0, 0, 10, 12, 0, 0],
#     [0, 4, 0, 0, 14, 0],
#     [0, 0, 9, 0, 0, 20],
#     [0, 0, 0, 7, 0, 4],
#     [0, 0, 0, 0, 0, 0]
# ]

# source_node = 0
# sink_node = 5
# max_flow_value = edmonds_karp(graph, source_node, sink_node)
# print(f"The maximum possible flow is: {max_flow_value}")

#-------------------------------------------------------------------------------------------

#graph_coloring.py

# def is_safe(v, graph, colors, c):
#     for i in range(len(graph)):
#         if graph[v][i] == 1 and colors[i] == c:
#             return False
#     return True

# # Utility function for backtracking
# def graph_coloring_util(graph, m, colors, v):
#     num_vertices = len(graph)
#     if v == num_vertices:
#         return True

#     for c in range(1, m + 1):
#         if is_safe(v, graph, colors, c):
#             colors[v] = c
#             if graph_coloring_util(graph, m, colors, v + 1):
#                 return True
#             colors[v] = 0 
#     return False

# def graph_coloring(graph, m):
#     num_vertices = len(graph)
#     colors = [0] * num_vertices

#     if not graph_coloring_util(graph, m, colors, 0):
#         print(f"Solution does not exist for {m} colors.")
#         return None

#     print(f"A valid coloring exists with {m} colors:")
#     return colors

# graph = [
#     [0, 1, 1, 1],
#     [1, 0, 1, 0],
#     [1, 1, 0, 1],
#     [1, 0, 1, 0]
# ]

# # Using 3 colors
# m = 3
# coloring_solution = graph_coloring(graph, m)
# if coloring_solution:
#     color_map = {1: "Red", 2: "Green", 3: "Blue"}
#     for i, color_num in enumerate(coloring_solution):
#         print(f"Vertex {i} ---> Color: {color_map[color_num]}")

# # Trying with 2 colors (not possible)
# print("\n Trying with 2 colors ")
# graph_coloring(graph, 2)
