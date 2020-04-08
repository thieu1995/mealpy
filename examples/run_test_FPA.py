#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:47, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.type_based.uni_modal import Functions
from mealpy.evolutionary_based.FPA import BaseFPA

t1 = Functions()

## Setting parameters
objective_func = t1._sum_squres__
problem_size = 30
domain_range = [-15, 15]
log = True

epoch = 100
pop_size = 50
p = 0.8

md1 = BaseFPA(objective_func, problem_size, domain_range, log, epoch, pop_size, p)
best_pos1, best_fit1, list_loss1 = md1._train__()
print(best_fit1)