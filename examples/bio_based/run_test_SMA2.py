#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 11:20, 20/10/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from opfunu.cec_basic.cec2014_nobias import *
from mealpy.bio_based import SMA, VCS, BBO, EOA, IWO, SBO, WHO
from mealpy.physics_based import EO
from mealpy.evolutionary_based import MA, FPA, ES, EP, DE, GA, CRO
from mealpy.probabilistic_based import CEM
from mealpy.music_based import HS
from mealpy.system_based import WCA, GCO, AEO
from mealpy.math_based import AOA, HC, SCA
from mealpy.human_based import BRO, CA, FBIO, SARO, SSDO, TLO, GSKA, LCO, ICA, BSO, QSA, CHIO
from mealpy.physics_based import ArchOA, ASO, EFO, HGSO, MVO, WDO, SA, TWO, NRO
from mealpy.swarm_based import ABC, ACOR, AO, BA, WOA, SSA, SLO, SHO, SSO, NMRA, MSA, MRFO, MFO, JA
from mealpy.swarm_based import GOA, CSA, BSA, ALO, BeesA, BES, FFA, FOA, PFA, COA, FA, SFO, SSpiderA, SSpiderO
from mealpy.swarm_based import HHO, GWO, EHO, CSO, DO, SRSR, PSO, BFO, HGS

from mealpy.problem import Problem
from mealpy.utils.termination import Termination
import numpy as np

# Setting parameters

# A - Different way to provide lower bound and upper bound. Here are some examples:

# def objective(x):
#     return x[0]**2 + (x[1] + 1)**2 - 5 * np.cos(1.5* x[0] + 1.5) - 3 * cos(2 * x[0] - 1.5)
#     # return (x[0]-3.14)**2 + (x[1] - 2.72)**2 + np.sin(3*x[0]+1.41) + sin(4*x[1] - 1.73)
#     # return np.sum(x**2)
#
# ## A1. When you have different lower bound and upper bound for each parameters
# problem_dict1 = {
#     "obj_func": objective,
#     "lb": [-10, -10],
#     "ub": [10, 10 ],
#     "minmax": "min",
#     "verbose": True,
# }
#
# if __name__ == "__main__":
#     problem_obj1 = Problem(problem_dict1)
#     ### Your parameter problem can be an instane of Problem class or just dict like above
#     model1 = CEM.BaseCEM(problem_obj1, epoch=50, pop_size=50)
#     model1.solve(mode="thread")
#     print(model1.solution[0])


VALUES = np.array([
    360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147,
    78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28,
    87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276,
    312
])
WEIGHTS = np.array([
    7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
    42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
    3, 86, 66, 31, 65, 0, 79, 20, 65, 52, 13
])
CAPACITY = 850

# Constraint optimization problem


LB = [0] * 50
UB = [1.99] * 50


def objective_function(solution):
    # solution =  np.clip(solution, LB, UB)
    def punish(value):
        return 0 if value <= CAPACITY else value

    solution_int = solution.astype(int)
    current_capacity = np.sum(solution_int * WEIGHTS)
    temp = np.sum(solution_int * VALUES) - punish(current_capacity)
    return temp.item()


problem_dict1 = {
    "obj_func": objective_function,
    "lb": LB,
    "ub": UB,
    "minmax": "max",
    "verbose": True,
}

## Run the algorithm

# from mealpy.swarm_based import HHO, GWO, EHO, CSO, DO, SRSR, PSO, BFO, HGS

model1 = MSA.BaseMSA(problem_dict1, epoch=100, pop_size=50)
model1.solve()

print(model1.solution[0])
print(model1.solution[0].astype(int))

