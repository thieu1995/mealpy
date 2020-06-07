#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 17:51, 06/06/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.math_based.SCA import BaseSCA

## Setting parameters
obj_func = F2
lb = [-15]
ub = [15]
problem_size = 30
batch_size = [1, 5, 10, 25, 50]
verbose = False
epoch = 1000
pop_size = 50

for batch in batch_size:
    md1 = BaseSCA(obj_func, lb, ub, problem_size, batch, verbose, epoch, pop_size)
    best_pos1, best_fit1, list_loss1 = md1.train()
    print(best_fit1)