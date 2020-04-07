#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 20:43, 07/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.type_based.uni_modal import Functions
from mealpy.bio_based.AAA import OriginalAAA

t1 = Functions()

## Setting parameters
objective_func = t1._sum_squres__
problem_size = 200
domain_range = [-15, 15]
log = True

epoch = 1000
pop_size = 50
energy = 0.3
delta = 2
ap = 0.5

md1 = OriginalAAA(objective_func, problem_size, domain_range, log, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1._train__()
print(best_fit1)
print("========================================================")
