#!/usr/bin/env python
# Created by "Thieu" at 16:31, 07/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# Location optimization, also known as facility location optimization or site location optimization, is a field of study in operations research
# and mathematical optimization that focuses on determining the optimal locations for facilities or resources. It involves making
# decisions about where to locate facilities, such as warehouses, distribution centers, healthcare clinics, schools, or other
# types of facilities, to meet specific objectives and constraints.

# The goal of location optimization is to find the best configuration of facility locations that minimizes costs, maximizes efficiency,
# improves service coverage, or achieves other desired objectives. The specific objectives can vary depending on the application
# and industry. Some common objectives include minimizing transportation costs, reducing customer waiting times, maximizing market
# coverage, minimizing facility construction costs, or balancing the workload among facilities.

# Location optimization problems typically involve a set of potential locations, each with associated costs, capacities, and other constraints.
# The problem also considers the demand or service requirements that need to be satisfied by the facilities. The decision variables in location
# optimization are the selection of facility locations and the allocation of demand or resources to those locations.

# Solving a location optimization problem often requires considering factors such as transportation costs, travel distances,
# population density, market demand, geographical features, infrastructure availability, and other relevant spatial or non-spatial data.
# Mathematical modeling techniques, such as integer programming, network optimization, or heuristic algorithms, are commonly
# used to formulate and solve location optimization problems.

# Location optimization has applications in various industries and sectors, including supply chain and logistics, retail planning,
# healthcare resource allocation, emergency service planning, telecommunications network design, and many others. By strategically determining
# the optimal locations of facilities, organizations can improve operational efficiency, reduce costs, enhance customer service, and
# make informed decisions about resource allocation.


#  Let's consider an example of location optimization in the context of a retail company that wants to open a certain
#  number of new stores in a region to maximize market coverage while minimizing operational costs.

# The company wants to open five new stores in a region with several potential locations. The objective is to determine the optimal
# locations for these stores while considering factors such as population density and transportation costs. The goal is to maximize
# market coverage by locating stores in areas with high demand while minimizing the overall transportation costs required to serve customers.

# By applying location optimization techniques, the retail company can make informed decisions about where to open new stores,
# considering factors such as population density and transportation costs. This approach allows the company to maximize market coverage,
# make efficient use of resources, and ultimately improve customer service and profitability.

# Note that this example is a simplified illustration, and in real-world scenarios, location optimization problems can involve more
# complex constraints, additional factors, and larger datasets. However, the general process remains similar, involving data analysis,
# mathematical modeling, and optimization techniques to determine the optimal locations for facilities.

import numpy as np
from mealpy import BinaryVar, WOA, Problem

# Define the coordinates of potential store locations
locations = np.array([
    [2, 4],
    [5, 6],
    [9, 3],
    [7, 8],
    [1, 10],
    [3, 2],
    [5, 5],
    [8, 2],
    [7, 6],
    [1, 9]
])
# Define the transportation costs matrix based on the Euclidean distance between locations
distance_matrix = np.linalg.norm(locations[:, np.newaxis] - locations, axis=2)

# Define the number of stores to open
num_stores = 5

# Define the maximum distance a customer should travel to reach a store
max_distance = 10

data = {
    "num_stores": num_stores,
    "max_distance": max_distance,
    "penalty": 1e10
}


class LocationOptProblem(Problem):
    def __init__(self, bounds=None, minmax=None, data=None, **kwargs):
        self.data = data
        self.eps = 1e10
        super().__init__(bounds, minmax, **kwargs)

    # Define the fitness evaluation function
    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        x = x_decoded["placement_var"]
        total_coverage = np.sum(x)
        total_dist = np.sum(x[:, np.newaxis] * distance_matrix)
        if total_dist == 0:                 # Penalize solutions with fewer stores
            return self.eps
        if total_coverage < self.data["num_stores"]:    # Penalize solutions with fewer stores
            return self.eps
        return total_dist


bounds = BinaryVar(n_vars=len(locations), name="placement_var")
problem = LocationOptProblem(bounds=bounds, minmax="min", data=data)

model = WOA.OriginalWOA(epoch=50, pop_size=20)
model.solve(problem)

print(f"Best agent: {model.g_best}")                    # Encoded solution
print(f"Best solution: {model.g_best.solution}")        # Encoded solution
print(f"Best fitness: {model.g_best.target.fitness}")
print(f"Best real scheduling: {model.problem.decode_solution(model.g_best.solution)}")      # Decoded (Real) solution
