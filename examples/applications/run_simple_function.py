#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:49, 10/11/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from mealpy.bio_based import SMA
import numpy as np

def obj_function(solution):
    return np.sum(solution**2)

problem_dict1 = {
    "obj_func": obj_function,
    "lb": [-100, ] * 30,
    "ub": [100, ] * 30,
    "minmax": "min",
    "verbose": True,
}

## Run the algorithm
model1 = SMA.BaseSMA(problem_dict1, epoch=100, pop_size=50, pr=0.03)
model1.solve()