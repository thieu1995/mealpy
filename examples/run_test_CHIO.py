#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:21, 09/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from mealpy.human_based.CHIO import BaseCHIO, OriginalCHIO
from opfunu.cec_basic.cec2014_nobias import *

## Setting parameters
objective_func = F1
problem_size = 20
domain_range = [-15, 15]
log = True
epoch = 1000
pop_size = 50
brr = 0.06
max_age = 150

md1 = BaseCHIO(objective_func, problem_size, domain_range, log, epoch, pop_size, brr, max_age)
best_pos1, best_fit1, list_loss1 = md1._train__()
print(best_fit1)
print("========================================================")