#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:33, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
# -------------------------------------------------------------------------------------------------------%

from mealpy.evolutionary_based.GA import BaseGA


def my_elliptic_function(solution=None):
    solution = solution.reshape((-1))
    result = 0
    for i in range(len(solution)):
        result += (10 ** 6) ** (i / (len(solution) - 1)) * solution[i] ** 2
    return result


## Setting parameters
obj_func = my_elliptic_function
# lb = [-15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3]
# ub = [15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3]
lb = [-100, ] * 15
ub = [100, ] * 15
verbose = True
epoch = 1000
pop_size = 50

pc = 0.95
pm = 0.025

md1 = BaseGA(obj_func, lb, ub, verbose, epoch, pop_size, pc, pm)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)
