#!/usr/bin/env python
# Created by "Thieu" at 11:34, 11/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy import FloatVar, BBO
from mealpy.utils.visualize import *


## Define your own fitness function
# Multi-objective but single fitness/target value. By using weighting method to convert from multiple objectives to single target

def fitness_function(solution):
    f1 = (np.sum(solution**2) - np.mean(solution)) / len(solution)
    f2 = np.sum(np.sqrt(np.abs(solution)))
    f3 = np.sum(np.mean(solution**2) - solution)
    return [f1, f2, f3]


problem = {
    "fit_func": fitness_function,
    "bounds": FloatVar(lb=[-10, -5, -15, -20, -10, -15, -10, -30], ub=[10, 5, 15, 20, 50, 30, 100, 85]),
    "minmax": "min",
    "obj_weights": [0.2, 0.5, 0.3]
}

## Run the algorithm
model = BBO.OriginalBBO(epoch=100, pop_size=50)
g_best = model.solve(problem)
print(f"Best solution: {g_best.solution}, Best fitness: {g_best.fitness}")


## Draw convergence chart for globest solution found so far in each previous generation
export_convergence_chart(model.history.list_global_best_fit, title='Global Best Fitness', filename="Global-best-convergence-chart")

# Parameter for this function
# data: optimizer.history_list_g_best_fit -> List of global best fitness found so far in each previous generation
# title: title of the figure
# linestyle: matplotlib API, default = "-"
# color: matplotlib API, default = "b"  -> Blue
# x_label: string, default = "#Iteration"
# y_label: string, default = "Function Value"
# filename: string, default = "convergence_chart"
# exts: matplotlib API, default = (".png", ".pdf") --> save figure in format of png and pdf
# verbose: show the figure on Python IDE, default = True

## Draw convergence chart for current best solution in each generation
export_convergence_chart(model.history.list_current_best_fit, title='Local Best Fitness', filename='Current-best-convergence-chart')

## Draw runtime for each generation
export_convergence_chart(model.history.list_epoch_time, title='Runtime chart', y_label="Second", filename='Runtime-per-epoch-chart')
