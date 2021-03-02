#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 11:57, 02/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.swarm_based.SalpSO import BaseSalpSO

## Setting parameters
obj_func = F19
# lb = [-15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3]
# ub = [15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3]
lb = [-100]
ub = [100]
problem_size = 100
batch_size = 25
verbose = True
epoch = 100
pop_size = 50

md1 = BaseSalpSO(obj_func, lb, ub, problem_size, batch_size, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)