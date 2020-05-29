#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:05, 29/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.swarm_based.SpaSA import BaseSpaSA

## Setting parameters
objective_func = F21
problem_size = 3000
domain_range = [-150, 150]
log = True

epoch = 100
pop_size = 50

md1 = BaseSpaSA(objective_func, problem_size, domain_range, log, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1._train__()
print(best_fit1)
print("========================================================")
