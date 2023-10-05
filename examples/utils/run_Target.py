#!/usr/bin/env python
# Created by "Thieu" at 16:50, 05/10/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.utils.target import Target


# Example usage
def objective_function(solution):
    # This function returns a list of two objectives for the given solution
    return [solution[0] ** 2, solution[1] ** 2]

# Define the solution and weights
solution = [2, 3]
weights = [0.5, 0.5]

# Evaluate the objectives for the solution
objectives = objective_function(solution)

# Create a Target instance with the objectives
target = Target(objectives, weights)

# # Calculate the fitness value
# fitness = target.calculate_fitness(weights)
print(target.fitness)
print(target.objectives)
print(target.weights)
# Print the target
print(target)
