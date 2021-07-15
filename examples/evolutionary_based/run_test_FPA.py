#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:47, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.evolutionary_based.FPA import BaseFPA
from mealpy.utils.visualize import export_convergence_chart, export_explore_exploit_chart, \
    export_diversity_chart, export_objectives_chart, export_trajectory_chart
from numpy import array

## Setting parameters
problem1 = {
    "obj_func": F3,
    "lb": [-3, -5, 1, -10, -10, -20, -5],
    "ub": [5, 10, 100, 30, 10, 20, 5],
    "minmax": "min",
    "verbose": True,
}

# A - Different way to provide lower bound and upper bound. Here are some examples:

## 1. When you have different lower bound and upper bound for each parameters
md1 = BaseFPA(problem1, epoch=10, pop_size=50)
best_pos1, best_fit1 = md1.train()
print(md1.solution[1])

## 2. When you have same lower bound and upper bound for each parameters, then you can use:
##      + int or float: then you need to specify your problem size (number of dimensions)
problem2 = {
    "obj_func": F5,
    "lb": -10,
    "ub": 30,
    "minmax": "min",
    "verbose": True,
    "problem_size": 30,  # Remember the keyword "problem_size"
}
md2 = BaseFPA(problem2, epoch=10, pop_size=50)  # Remember the keyword "problem_size"
md2.train()
print(md2.solution[1])

##      + array: 2 ways
problem3 = {
    "obj_func": F5,
    "lb": [-5],
    "ub": [10],
    "minmax": "min",
    "verbose": True,
    "problem_size": 30,  # Remember the keyword "problem_size"
}
md3 = BaseFPA(problem3, epoch=10, pop_size=50)  # Remember the keyword "problem_size"
md3.train()
print(md3.solution[1])

problem_size = 30
problem4 = {
    "obj_func": F5,
    "lb": [-5] * problem_size,
    "ub": [10] * problem_size,
    "minmax": "min",
    "verbose": True,
}
md4 = BaseFPA(problem4, epoch=10, pop_size=50)  # Remember the keyword "problem_size"
md4.train()
print(md4.solution[1])


# B - Test with algorithm has batch size idea


# C - Test with different variants of this algorithm


# D - Drawing all available figures

## Multi-objective but single fitness/target function. By using weighting method to convert from multiple to single.
def obj_function(solution):
    t1 = solution[0] ** 2
    t2 = ((2 * solution[1]) / 5) ** 2
    t3 = 0
    for i in range(3, len(solution)):
        t3 += (1 + solution[i] ** 2) ** 0.5
    return [t1, t2, t3]


## Setting parameters
problem_size = 30
problem5 = {
    "obj_func": obj_function,
    "lb": [-5] * problem_size,
    "ub": [10] * problem_size,
    "minmax": "min",
    "verbose": True,
}
md5 = BaseFPA(problem5, epoch=10, pop_size=50)  # Remember the keyword "problem_size"
best_position, best_fitness = md5.train()

export_convergence_chart(md5.history_list_g_best_fit, title='Global Best Fitness')  # Draw global best fitness found so far in previous generations
export_convergence_chart(md5.history_list_c_best_fit, title='Local Best Fitness')  # Draw current best fitness in each previous generation
export_convergence_chart(md5.history_list_epoch_time, title='Runtime chart', y_label="Second")  # Draw runtime for each generation

## On the exploration and exploitation in popular swarm-based metaheuristic algorithms

# This exploration/exploitation chart should draws for single algorithm and single fitness function
export_explore_exploit_chart([md5.history_list_explore, md5.history_list_exploit])  # Draw exploration and exploitation chart

# This diversity chart should draws for multiple algorithms for a single fitness function at the same time to compare the diversity spreading
export_diversity_chart([md5.history_list_div], list_legends=['GA'])  # Draw diversity measurement chart

## Because convergence chart is formulated from objective values and weights,
## thus we also want to draw objective charts to understand the convergence
## Need a little bit more pre-processing


global_obj_list = array([agent[1][-1] for agent in md5.history_list_g_best])  # 2D array / matrix 2D
global_obj_list = [global_obj_list[:, idx] for idx in range(0, len(global_obj_list[0]))]  # Make each obj_list as a element in array for drawing
export_objectives_chart(global_obj_list, title='Global Objectives Chart')

current_obj_list = array([agent[1][-1] for agent in md5.history_list_c_best])
current_obj_list = [current_obj_list[:, idx] for idx in range(0, len(current_obj_list[0]))]  # Make each obj_list as a element in array for drawing
export_objectives_chart(current_obj_list, title='Local Objectives Chart')

## Drawing trajectory of some agents in the first and second dimensions
# Need a little bit more pre-processing
pos_list = []
list_legends = []
dimension = 2
y_label = f"x{dimension + 1}"
for i in range(0, 5, 2):  # Get the third dimension of position of the first 3 solutions
    x = [pop[0][0][dimension] for pop in md5.history_list_pop]
    pos_list.append(x)
    list_legends.append(f"Agent {i + 1}.")
    # pop[0]: Get the first solution
    # pop[0][0]: Get the position of the first solution
    # pop[0][0][0]: Get the first dimension of the position of the first solution
export_trajectory_chart(pos_list, list_legends=list_legends, y_label=y_label)


