#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 20:16, 19/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.type_based.uni_modal import Functions
from opfunu.cec_basic.cec2014_nobias import *
from mealpy.probabilistic_based.CEM import BaseCEM, CEBaseLCBO

t1 = Functions()

## Setting parameters
objective_func = t1._sum_squres__           # F1
problem_size = 200
domain_range = [-150, 150]
log = True
epoch = 100
pop_size = 50
n_best = 10
alpha = 0.7

md1 = BaseCEM(objective_func, problem_size, domain_range, log, epoch, pop_size, n_best, alpha)
best_pos1, best_fit1, list_loss1 = md1._train__()
print(best_fit1)
print("========================================================")

# md2 = CEBaseLCBO(objective_func, problem_size, domain_range, log, epoch, pop_size, alpha)
# best_pos2, best_fit2, list_loss2 = md2._train__()
# print(best_fit2)
# print("========================================================")
#


