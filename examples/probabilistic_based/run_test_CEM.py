#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:02, 08/06/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.probabilistic_based.CEM import BaseCEM, CEBaseSBO, CEBaseSSDO, CEBaseLCBO, CEBaseLCBONew

# BaseCEM < CEBaseSBO < CEBaseLCBO < CEBaseLCBONew < CEBaseSSDO
# 992611825880, 1531140193, 230612, 234783, 238326

## Setting parameters
obj_func = F19
# lb = [-15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3]
# ub = [15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3]
lb = [-100]
ub = [100]
problem_size = 2000
batch_size = 25
verbose = True
epoch = 100
pop_size = 50

md1 = BaseCEM(obj_func, lb, ub, problem_size, batch_size, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)


md1 = CEBaseSBO(obj_func, lb, ub, problem_size, batch_size, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

md1 = CEBaseSSDO(obj_func, lb, ub, problem_size, batch_size, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

md1 = CEBaseLCBO(obj_func, lb, ub, problem_size, batch_size, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

md1 = CEBaseLCBONew(obj_func, lb, ub, problem_size, batch_size, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)
