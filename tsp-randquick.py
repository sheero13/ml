#TSP.py
# import math

# def distance(a, b):
#     return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# def tsp_nearest_neighbor(cities):
#     n = len(cities)
#     visited = [False] * n
#     path = [0]
#     visited[0] = True
#     total_distance = 0

#     for _ in range(n - 1):
#         last = path[-1]
#         nearest_city = None
#         nearest_dist = float('inf')
#         for i in range(n):
#             if not visited[i]:
#                 dist = distance(cities[last], cities[i])
#                 if dist < nearest_dist:
#                     nearest_city = i
#                     nearest_dist = dist
#         path.append(nearest_city)
#         visited[nearest_city] = True
#         total_distance += nearest_dist

#     total_distance += distance(cities[path[-1]], cities[0])
#     path.append(0)

#     return path, total_distance

# # Get input from user
# n = int(input("Enter number of cities: "))
# cities = []
# for i in range(n):
#     x, y = map(float, input(f"Enter coordinates of city {i} (x y): ").split())
#     cities.append((x, y))

# path, total_distance = tsp_nearest_neighbor(cities)

# print("\nVisited order of cities:", path)
# print("Total distance of the tour:", round(total_distance, 2))

# print("\nApproximation ratio: Depends on city distribution (usually <= 2 for metric TSP).")
# print("Time complexity: O(n²)")

#-----------------------------------------------------------------------------------------------------
#rand_quick.py

# import random
# import time
# import sys

# sys.setrecursionlimit(10000)  # allow deeper recursion safely

# # ---------- Standard Quick Sort ----------
# def quicksort(arr):
#     if len(arr) <= 1:
#         return arr
#     pivot = arr[len(arr) // 2]  # safer pivot (middle element)
#     left = [x for x in arr if x < pivot]
#     middle = [x for x in arr if x == pivot]
#     right = [x for x in arr if x > pivot]
#     return quicksort(left) + middle + quicksort(right)

# # ---------- Randomized Quick Sort ----------
# def randomized_quicksort(arr):
#     if len(arr) <= 1:
#         return arr
#     pivot = random.choice(arr)
#     left = [x for x in arr if x < pivot]
#     middle = [x for x in arr if x == pivot]
#     right = [x for x in arr if x > pivot]
#     return randomized_quicksort(left) + middle + randomized_quicksort(right)

# # ---------- Performance Comparison ----------
# n = int(input("Enter number of elements: "))
# arr = list(range(n))  # worst-case sorted array for deterministic

# arr1 = arr.copy()
# arr2 = arr.copy()

# # Standard Quick Sort timing
# start = time.time()
# quicksort(arr1)
# end = time.time()
# print(f"\nStandard Quick Sort Time: {end - start:.6f} seconds")

# # Randomized Quick Sort timing
# start = time.time()
# randomized_quicksort(arr2)
# end = time.time()
# print(f"Randomized Quick Sort Time: {end - start:.6f} seconds")

# print("\nTime Complexity (average case): O(n log n)")
# print("Worst Case (Deterministic): O(n²)")
# print("Worst Case (Randomized): O(n log n) expected")
