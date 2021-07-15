#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 11:35, 11/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from mealpy.evolutionary_based.GA import BaseGA
from mealpy.utils.visualize import *
from numpy import array, sum, mean, sqrt


## Define your own fitness function
# Multi-objective but single fitness/target value. By using weighting method to convert from multiple objectives to single target

def obj_function(solution):
    f1 = (sum(solution ** 2) - mean(solution)) / len(solution)
    f2 = sum(sqrt(abs(solution)))
    f3 = sum(mean(solution ** 2) - solution)
    return [f1, f2, f3]


## Setting parameters
verbose = True
epoch = 100
pop_size = 50

lb1 = [-10, -5, -15, -20, -10, -15, -10, -30]
ub1 = [10, 5, 15, 20, 50, 30, 100, 85]

optimizer = BaseGA(obj_function, lb1, ub1, "min", verbose, epoch, pop_size, obj_weight=[0.2, 0.5, 0.3])
best_position, best_fitness, g_best_fit_list, c_best_fit_list = optimizer.train()
print(best_position)


## Because convergence chart is formulated from objective values and weights, thus we also want to draw objective charts to understand the convergence
# Need a little bit more pre-processing
global_obj_list = array([agent[-1][-1] for agent in optimizer.g_best_list])  # 2D array / matrix 2D
global_obj_list = [global_obj_list[:, idx] for idx in range(0, len(global_obj_list[0]))]  # Make each obj_list as a element in array for drawing
export_objectives_chart(global_obj_list, title='Global Objectives Chart', filename="global-objective-chart")

current_obj_list = array([agent[-1][-1] for agent in optimizer.c_best_list])  # 2D array / matrix 2D
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

