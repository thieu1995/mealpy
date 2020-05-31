#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:47, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from mealpy.evolutionary_based.FPA import BaseFPA
from opfunu.cec_basic.cec2014_nobias import *

## Setting parameters
objective_func = F1
problem_size = 30
domain_range = [-15, 15]
log = True

epoch = 100
pop_size = 50
p = 0.8

md1 = BaseFPA(objective_func, problem_size, domain_range, log, epoch, pop_size, p)
best_pos1, best_fit1, list_loss1 = md1._train__()
print(best_fit1)