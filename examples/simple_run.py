#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:11, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.type_based.uni_modal import Functions
from mapy.evolutionary_based.GA import BaseGA
t1 = Functions()

## Setting parameters
ga_paras = {
    "epoch": 100,
    "pop_size": 50,
    "pc": 0.95,
    "pm": 0.025
}

root_algo_paras = {
    "problem_size": 30,
    "domain_range": [-15, 15],
    "print_train": True,
    "objective_func": t1._sum_squres__
}

md = BaseGA(root_algo_paras=root_algo_paras, ga_paras=ga_paras)
best_position, list_loss = md._train__()
print(list_loss)
