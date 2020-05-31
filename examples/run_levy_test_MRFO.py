#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:40, 30/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from mealpy.swarm_based.MRFO import OriginalMRFO, BaseMRFO, LevyMRFO
from opfunu.cec_basic.cec2014_nobias import *

## Setting parameters
objective_func = F1
problem_size = 3000
domain_range = [-15, 15]
log = True
epoch = 100
pop_size = 50

md1 = OriginalMRFO(objective_func, problem_size, domain_range, log, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1._train__()
print(best_fit1)
print("========================================================")

md2 = BaseMRFO(objective_func, problem_size, domain_range, log, epoch, pop_size)
best_pos2, best_fit2, list_loss2 = md2._train__()
print(best_fit2)
print("========================================================")

md3 = LevyMRFO(objective_func, problem_size, domain_range, log, epoch, pop_size)
best_pos3, best_fit3, list_loss3 = md3._train__()
print(best_fit3)
