#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:53, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from mealpy.human_based.BSO import BaseBSO, ImprovedBSO
from opfunu.cec_basic.cec2014_nobias import *

## Setting parameters
objective_func = F1
problem_size = 2000
domain_range = [-15, 15]
log = True

epoch = 100
pop_size = 50

md1 = BaseBSO(objective_func, problem_size, domain_range, log, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1._train__()
print(best_fit1)
print("========================================================")

md2 = ImprovedBSO(objective_func, problem_size, domain_range, log, epoch, pop_size)
best_pos2, best_fit2, list_loss2 = md2._train__()
print(best_fit2)
print("========================================================")
