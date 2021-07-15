#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 22:08, 22/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.evolutionary_based.GA import BaseGA

## Setting parameters
problem = {
    "obj_func": F5,
    "lb": [-3, -5, 1, -10],
    "ub": [5, 10, 100, 30],
    "minmax": "min",
    "verbose": True,
}

# A - Different way to provide lower bound and upper bound. Here are some examples:

## 1. When you have different lower bound and upper bound for each parameters
md1 = BaseGA(problem, epoch=10, pop_size=50)
best_pos1, best_fit1 = md1.train()
print(md1.solution[1])



## 2. When you have same lower bound and upper bound for each parameters, then you can use:
##      + int or float: then you need to specify your problem size (number of dimensions)
problem = {
    "obj_func": F5,
    "lb": -10,
    "ub": 30,
    "minmax": "min",
    "verbose": True,
    "problem_size": 30,  # Remember the keyword "problem_size"
}
md2 = BaseGA(problem, epoch=10, pop_size=50)  # Remember the keyword "problem_size"
md2.train()
print(md2.solution[1])

##      + array: 2 ways
problem = {
    "obj_func": F5,
    "lb": [-5],
    "ub": [10],
    "minmax": "min",
    "verbose": True,
    "problem_size": 30,  # Remember the keyword "problem_size"
}
md3 = BaseGA(problem, epoch=10, pop_size=50)  # Remember the keyword "problem_size"
md3.train()
print(md3.solution[1])


problem_size = 30
problem = {
    "obj_func": F5,
    "lb": [-5] * problem_size,
    "ub": [10] * problem_size,
    "minmax": "min",
    "verbose": True,
}
md4 = BaseGA(problem, epoch=10, pop_size=50)  # Remember the keyword "problem_size"
md4.train()
print(md4.solution[1])


# B - Test with algorithm has batch size idea


# C - Test with different variants of this algorithm

