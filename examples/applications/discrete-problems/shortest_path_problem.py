#!/usr/bin/env python
# Created by "Thieu" at 15:29, 07/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# In this example, the graph is represented as a NumPy array where each element represents the cost or distance between two nodes.
#
# Note that this implementation assumes that the graph is represented by a symmetric matrix, where graph[i,j] represents
# the distance between nodes i and j. If your graph representation is different, you may need to modify the code accordingly.
#
# Please keep in mind that this implementation is a basic example and may not be optimized for large-scale problems.
# Further modifications and optimizations may be required depending on your specific use case.

import numpy as np
from mealpy import PermutationVar, WOA, Problem

# Define the graph representation
graph = np.array([
    [0, 2, 4, 0, 7, 9],
    [2, 0, 1, 4, 2, 8],
    [4, 1, 0, 1, 3, 0],
    [6, 4, 5, 0, 3, 2],
    [0, 2, 3, 3, 0, 2],
    [9, 0, 4, 2, 2, 0]
])


class ShortestPathProblem(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        self.data = data
        self.eps = 1e10         # Penalty function for vertex with 0 connection
        super().__init__(bounds, minmax, **kwargs)

    # Calculate the fitness of an individual
    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        individual = x_decoded["path"]

        total_distance = 0
        for idx in range(len(individual) - 1):
            start_node = individual[idx]
            end_node = individual[idx + 1]
            weight = self.data[start_node, end_node]
            if weight == 0:
                return self.eps
            total_distance += weight
        return total_distance


num_nodes = len(graph)
bounds = PermutationVar(valid_set=list(range(0, num_nodes)), name="path")
problem = ShortestPathProblem(bounds=bounds, minmax="min", data=graph)

model = WOA.OriginalWOA(epoch=100, pop_size=20)
model.solve(problem)

print(f"Best agent: {model.g_best}")                    # Encoded solution
print(f"Best solution: {model.g_best.solution}")        # Encoded solution
print(f"Best fitness: {model.g_best.target.fitness}")
print(f"Best real scheduling: {model.problem.decode_solution(model.g_best.solution)}")      # Decoded (Real) solution
