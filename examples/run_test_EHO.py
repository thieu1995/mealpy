#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:10, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.type_based.uni_modal import Functions
from mealpy.swarm_based.EHO import BaseEHO, LevyEHO

t1 = Functions()

## Setting parameters
objective_func = t1._chung_reynolds__
problem_size = 300
domain_range = [-15, 15]
log = True

epoch = 100
pop_size = 50
alpha = 0.5
beta = 0.1
n_clans = 5

md1 = BaseEHO(objective_func, problem_size, domain_range, log, epoch, pop_size, alpha, beta, n_clans)
best_pos1, best_fit1, list_loss1 = md1._train__()
print(best_fit1)
print("========================================================")

md2 = LevyEHO(objective_func, problem_size, domain_range, log, epoch, pop_size, alpha, beta, n_clans)
best_pos2, best_fit2, list_loss2 = md2._train__()
print(best_fit2)
