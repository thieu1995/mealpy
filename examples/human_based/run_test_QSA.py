#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:43, 07/06/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.human_based.QSA import BaseQSA, OppoQSA, LevyQSA, ImprovedQSA, OriginalQSA


# Setting parameters
obj_func = F5
verbose = False
epoch = 500
pop_size = 50

# A - Different way to provide lower bound and upper bound. Here are some examples:

## 1. When you have different lower bound and upper bound for each parameters
lb1 = [-3, -5, 1]
ub1 = [5, 10, 100]

md1 = BaseQSA(obj_func, lb1, ub1, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[1])

## 2. When you have same lower bound and upper bound for each parameters, then you can use:
##      + int or float: then you need to specify your problem size (number of dimensions)
problemSize = 10
lb2 = -5
ub2 = 10
md2 = BaseQSA(obj_func, lb2, ub2, verbose, epoch, pop_size, problem_size=problemSize)  # Remember the keyword "problem_size"
best_pos1, best_fit1, list_loss1 = md2.train()
print(md2.solution[1])

##      + array: 2 ways
lb3 = [-5]
ub3 = [10]
md3 = BaseQSA(obj_func, lb3, ub3, verbose, epoch, pop_size, problem_size=problemSize)  # Remember the keyword "problem_size"
best_pos1, best_fit1, list_loss1 = md3.train()
print(md3.solution[1])

lb4 = [-5] * problemSize
ub4 = [10] * problemSize
md4 = BaseQSA(obj_func, lb4, ub4, verbose, epoch, pop_size)  # No need the keyword "problem_size"
best_pos1, best_fit1, list_loss1 = md4.train()
print(md4.solution[1])


# B - Test with algorithm has batch size idea


# C - Test with different variants of this algorithm

md1 = OppoQSA(obj_func, lb4, ub4, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

md1 = LevyQSA(obj_func, lb4, ub4, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

md1 = ImprovedQSA(obj_func, lb4, ub4, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

md1 = OriginalQSA(obj_func, lb4, ub4, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

