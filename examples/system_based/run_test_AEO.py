#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:06, 06/06/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.system_based.AEO import BaseAEO, OriginalAEO, AdaptiveAEO, ImprovedAEO, EnhancedAEO, ModifiedAEO

# Setting parameters
obj_func = F5
verbose = False
epoch = 500
pop_size = 50

# A - Different way to provide lower bound and upper bound. Here are some examples:

## 1. When you have different lower bound and upper bound for each parameters
lb1 = [-3, -5, 1]
ub1 = [5, 10, 100]

md1 = OriginalAEO(obj_func, lb1, ub1, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[1])

## 2. When you have same lower bound and upper bound for each parameters, then you can use:
##      + int or float: then you need to specify your problem size (number of dimensions)
problemSize = 10
lb2 = -5
ub2 = 10
md2 = OriginalAEO(obj_func, lb2, ub2, verbose, epoch, pop_size, problem_size=problemSize)     # Remember the keyword "problem_size"
best_pos1, best_fit1, list_loss1 = md2.train()
print(md2.solution[1])

##      + array: 2 ways
lb3 = [-5]
ub3 = [10]
md3 = OriginalAEO(obj_func, lb3, ub3, verbose, epoch, pop_size, problem_size=problemSize)   # Remember the keyword "problem_size"
best_pos1, best_fit1, list_loss1 = md3.train()
print(md3.solution[1])

lb4 = [-5] * problemSize
ub4 = [10] * problemSize
md4 = OriginalAEO(obj_func, lb4, ub4, verbose, epoch, pop_size)  # No need the keyword "problem_size"
best_pos1, best_fit1, list_loss1 = md4.train()
print(md4.solution[1])


# B - Test with algorithm has batch size idea

## 1. Not using batch size idea

md5 = BaseAEO(obj_func, lb4, ub4, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md5.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

## 2. Using batch size idea
batchIdea = True
batchSize = 5

md6 = BaseAEO(obj_func, lb4, ub4, verbose, epoch, pop_size, batch_idea=batchIdea, batch_size=batchSize)     # Remember the keywords
best_pos1, best_fit1, list_loss1 = md6.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)


## C - Test with different variants of this algorithm

md1 = AdaptiveAEO(obj_func, lb4, ub4, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

md1 = ImprovedAEO(obj_func, lb4, ub4, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

md1 = EnhancedAEO(obj_func, lb4, ub4, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)

md1 = ModifiedAEO(obj_func, lb4, ub4, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[0])
print(md1.solution[1])
print(md1.loss_train)