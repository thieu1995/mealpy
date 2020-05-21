#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 01:57, 27/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.probabilistic_based.CEM import CEBaseLCBO, CEBaseSSDO, CEBaseLCBONew
from mealpy.human_based.LCBO import BaseLCBO, LevyLCBO, ImprovedLCBO

## Setting parameters
objective_func = F22
problem_size = 100
domain_range = [-100, 100]
log = True
epoch = 500
pop_size = 50

md1 = CEBaseSSDO(objective_func, problem_size, domain_range, log, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1._train__()
print(best_fit1)
print("========================================================")

md2 = CEBaseLCBO(objective_func, problem_size, domain_range, log, epoch, pop_size)
best_pos2, best_fit2, list_loss2 = md2._train__()
print(best_fit2)
print("========================================================")

md2 = CEBaseLCBONew(objective_func, problem_size, domain_range, log, epoch, pop_size)
best_pos2, best_fit2, list_loss2 = md2._train__()
print(best_fit2)
print("========================================================")

#
# md1 = BaseLCBO(objective_func, problem_size, domain_range, log, epoch, pop_size)
# best_pos1, best_fit1, list_loss1 = md1._train__()
# print(best_fit1)
# print("========================================================")
#
# md2 = LevyLCBO(objective_func, problem_size, domain_range, log, epoch, pop_size)
# best_pos2, best_fit2, list_loss2 = md2._train__()
# print(best_fit2)
# print("========================================================")
#
# md2 = ImprovedLCBO(objective_func, problem_size, domain_range, log, epoch, pop_size)
# best_pos2, best_fit2, list_loss2 = md2._train__()
# print(best_fit2)
# print("========================================================")

