#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:33, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import sum
from mealpy.evolutionary_based.GA import BaseGA

def my_objective_function(solution):
    return sum(solution**2)

root_paras = {
    "problem_size": 30,
    "domain_range": [-15, 15],
    "print_train": True,
    "objective_func": my_objective_function
}

## Setting parameters
epoch = 100
pop_size = 50
pc = 0.95
pm = 0.025

md = BaseGA(root_paras, epoch, pop_size, pc, pm)
best_position, best_fit, list_loss = md._train__()
print(best_fit)
