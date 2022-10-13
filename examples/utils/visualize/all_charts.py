#!/usr/bin/env python
# Created by "Thieu" at 10:11, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from mealpy.bio_based import SMA
from mealpy.utils.visualize import *
import numpy as np

## Multi-objective but single fitness function. By using weighting method to convert from multiple to single.


def fitness_function(solution):
    t1 = solution[0] ** 2
    t2 = ((2 * solution[1]) / 5) ** 2
    t3 = 0
    for i in range(3, len(solution)):
        t3 += (1 + solution[i]**2)**0.5
    return [t1, t2, t3]


problem = {
    "fit_func": fitness_function,
    "lb": [-10, -5, -15, -20],
    "ub": [10, 5, 15, 20],
    "minmax": "min",
    "obj_weights": [0.2, 0.5, 0.3]
}

## Run the algorithm
model = SMA.BaseSMA(epoch=100, pop_size=50)
best_position, best_fitness = model.solve(problem)
print(f"Best solution: {best_position}, Best fitness: {best_fitness}")

export_convergence_chart(model.history.list_global_best_fit, title='Global Best Fitness')            # Draw global best fitness found so far in previous generations
export_convergence_chart(model.history.list_current_best_fit, title='Local Best Fitness')             # Draw current best fitness in each previous generation
export_convergence_chart(model.history.list_epoch_time, title='Runtime chart', y_label="Second")        # Draw runtime for each generation

## On the exploration and exploitation in popular swarm-based metaheuristic algorithms

# This exploration/exploitation chart should draws for single algorithm and single fitness function
export_explore_exploit_chart([model.history.list_exploration, model.history.list_exploitation])  # Draw exploration and exploitation chart

# This diversity chart should draws for multiple algorithms for a single fitness function at the same time to compare the diversity spreading
export_diversity_chart([model.history.list_diversity], list_legends=['GA'])        # Draw diversity measurement chart

## Because convergence chart is formulated from objective values and weights, thus we also want to draw objective charts to understand the convergence
# Need a little bit more pre-processing
global_obj_list = np.array([agent[1][1] for agent in model.history.list_global_best])     # 2D array / matrix 2D
global_obj_list = [global_obj_list[:,idx] for idx in range(0, len(global_obj_list[0]))]     # Make each obj_list as a element in array for drawing
export_objectives_chart(global_obj_list, title='Global Objectives Chart')

current_obj_list = np.array([agent[1][1] for agent in model.history.list_current_best])  # 2D array / matrix 2D
current_obj_list = [current_obj_list[:, idx] for idx in range(0, len(current_obj_list[0]))]  # Make each obj_list as a element in array for drawing
export_objectives_chart(current_obj_list, title='Local Objectives Chart')

## Drawing trajectory of some agents in the first and second dimensions
# Need a little bit more pre-processing
pos_list = []
list_legends = []
dimension = 2
y_label = f"x{dimension+1}"
for i in range(0, 5, 2):   # Get the third dimension of position of the first 3 solutions
    x = [pop[0][0][dimension] for pop in model.history.list_population]
    pos_list.append(x)
    list_legends.append(f"Agent {i+1}.")
    # pop[0]: Get the first solution
    # pop[0][0]: Get the position of the first solution
    # pop[0][0][0]: Get the first dimension of the position of the first solution
export_trajectory_chart(pos_list, list_legends=list_legends, y_label=y_label)


### Or better to use the API
## You can access all of available figures via object "history" like this:
model.history.save_global_objectives_chart(filename="results/goc")
model.history.save_local_objectives_chart(filename="results/loc")
model.history.save_global_best_fitness_chart(filename="results/gbfc")
model.history.save_local_best_fitness_chart(filename="results/lbfc")
model.history.save_runtime_chart(filename="results/rtc")
model.history.save_exploration_exploitation_chart(filename="results/eec")
model.history.save_diversity_chart(filename="results/dc")
model.history.save_trajectory_chart(list_agent_idx=[3, 5], selected_dimensions=[2], filename="results/tc")

