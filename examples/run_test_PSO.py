#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:34, 29/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.swarm_based.PSO import BasePSO, PPSO, PSO_W, HPSO_TVA

## Setting parameters
objective_func = F25
problem_size = 300
domain_range = [-15, 15]
log = True

epoch = 100
pop_size = 50

# md1 = BasePSO(objective_func, problem_size, domain_range, log, epoch, pop_size)
# best_pos1, best_fit1, list_loss1 = md1._train__()
# print(best_fit1)
#
# print("========================================================")
#
# md2 = PPSO(objective_func, problem_size, domain_range, log, epoch, pop_size)
# best_pos2, best_fit2, list_loss2 = md2._train__()
# print(best_fit2)
#
# print("========================================================")
#
# md2 = PSO_W(objective_func, problem_size, domain_range, log, epoch, pop_size)
# best_pos2, best_fit2, list_loss2 = md2._train__()
# print(best_fit2)

print("========================================================")

md2 = HPSO_TVA(objective_func, problem_size, domain_range, log, epoch, pop_size)
best_pos2, best_fit2, list_loss2 = md2._train__()
print(best_fit2)