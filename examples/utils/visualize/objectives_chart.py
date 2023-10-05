#!/usr/bin/env python
# Created by "Thieu" at 11:35, 11/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy import FloatVar, BBO
from mealpy.utils.visualize import *

## Define your own fitness function
# Multi-objective but single fitness/target value. By using weighting method to convert from multiple objectives to single target

def fitness_function(solution):
    f1 = (np.sum(solution ** 2) - np.mean(solution)) / len(solution)
    f2 = np.sum(np.sqrt(np.abs(solution)))
    f3 = np.sum(np.mean(solution ** 2) - solution)
    return [f1, f2, f3]


problem = {
    "fit_func": fitness_function,
    "bounds": FloatVar(n_vars=8, lb=[-10, -5, -15, -20, -10, -15, -10, -30], ub=[10, 5, 15, 20, 50, 30, 100, 85]),
    "minmax": "min",
    "obj_weights": [0.2, 0.5, 0.3]
}

## Run the algorithm
model = BBO.OriginalBBO(epoch=100, pop_size=50)
g_best = model.solve(problem)
print(f"Best solution: {g_best.solution}, Best fitness: {g_best.fitness}")


## Because convergence chart is formulated from objective values and weights, thus we also want to draw objective charts to understand the convergence
# Need a little bit more pre-processing
global_obj_list = np.array([agent[-1][-1] for agent in model.history.list_global_best])  # 2D array / matrix 2D
global_obj_list = [global_obj_list[:, idx] for idx in range(0, len(global_obj_list[0]))]  # Make each obj_list as a element in array for drawing
export_objectives_chart(global_obj_list, title='Global Objectives Chart', filename="global-objective-chart")

current_obj_list = np.array([agent[-1][-1] for agent in model.history.list_current_best])  # 2D array / matrix 2D
current_obj_list = [current_obj_list[:, idx] for idx in range(0, len(current_obj_list[0]))]  # Make each obj_list as a element in array for drawing
export_objectives_chart(current_obj_list, title='Local Objectives Chart', filename="local-objective-chart")

# Parameter for this function
# data: optimizer.history_list_g_best_fit -> List of global best fitness found so far in each previous generation
# title: title of the figure
# list_legends: list of line's name
# list_styles: matplotlib API, default = None
# list_colors: matplotlib API, default = None
# x_label: string, default = "#Iteration"
# y_label: string, default = "Function Value"
# filename: string, default = "Objective-chart"
# exts: matplotlib API, default = (".png", ".pdf") --> save figure in format of png and pdf
# verbose: show the figure on Python IDE, default = True
