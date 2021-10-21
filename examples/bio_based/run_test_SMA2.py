#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 11:20, 20/10/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.bio_based import SMA
from mealpy.swarm_based import HGS
from mealpy.human_based import GSKA
from mealpy.physics_based import EO
from mealpy.evolutionary_based import MA, FPA, ES, EP, DE
from mealpy.problem import Problem
from mealpy.utils.termination import Termination

# Setting parameters

# A - Different way to provide lower bound and upper bound. Here are some examples:

## A1. When you have different lower bound and upper bound for each parameters
problem_dict1 = {
    "obj_func": F5,
    "lb": [-3, -5, 1, -10, ],
    "ub": [5, 10, 100, 30, ],
    "minmax": "min",
    "verbose": True,
}

if __name__ == "__main__":
    problem_obj1 = Problem(problem_dict1)
    ### Your parameter problem can be an instane of Problem class or just dict like above
    model1 = DE.JADE(problem_obj1, epoch=100, pop_size=50)
    model1.solve(mode="sequential")