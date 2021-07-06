#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 16:56, 05/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.dummy.RRO import OriginalRRO, IRRO, BaseRRO

# Setting parameters
obj_func = F20
verbose = True
epoch = 100
pop_size = 50

# A - Different way to provide lower bound and upper bound. Here are some examples:

## 1. When you have different lower bound and upper bound for each parameters
lb1 = [-3, -5, 1]
ub1 = [5, 10, 100]

md1 = OriginalRRO(obj_func, lb1, ub1, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution)

md1 = IRRO(obj_func, lb1, ub1, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution)

md1 = BaseRRO(obj_func, lb1, ub1, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution)