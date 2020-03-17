#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:11, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.type_based.uni_modal import Functions
from mealpy.evolutionary_based.GA import BaseGA
t1 = Functions()

root_paras = {
    "problem_size": 30,
    "domain_range": [-15, 15],
    "print_train": True,
    "objective_func": t1._sum_squres__
}

## Setting parameters
epoch = 100
pop_size = 50
pc = 0.95
pm = 0.025

md = BaseGA(root_paras, epoch, pop_size, pc, pm)
best_position, list_loss = md._train__()
print(list_loss)
