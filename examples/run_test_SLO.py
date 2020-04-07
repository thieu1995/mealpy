#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:28, 07/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.type_based.uni_modal import Functions
from mealpy.swarm_based.SLO import BaseSLO, ImprovedSLO

t1 = Functions()

## Setting parameters
objective_func = t1._sum_squres__
problem_size = 3000
domain_range = [-15, 15]
log = True

epoch = 100
pop_size = 50

md1 = BaseSLO(objective_func, problem_size, domain_range, log, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1._train__()
print(best_fit1)
print("========================================================")

md2 = ImprovedSLO(objective_func, problem_size, domain_range, log, epoch, pop_size)
best_pos2, best_fit2, list_loss2 = md2._train__()
print(best_fit2)
