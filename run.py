#!/usr/bin/env python
# Created by "Thieu" at 16:53, 20/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.cec_based.cec2017 import F292017
from mealpy.swarm_based import WOA
from mealpy.bio_based.BMO import OriginalBMO
from mealpy.bio_based.TPO import OriginalTPO
from mealpy.swarm_based.EHO import OriginalEHO
from mealpy.swarm_based.ESOA import OriginalESOA

from mealpy import BBO as T1
from mealpy.bio_based import BBO as T2
from mealpy.bio_based.BBO import BaseBBO

# from mealpy.utils.problem import Problem
from mealpy import Problem

ndim = 30
f18 = F292017(ndim, f_bias=0)

def fitness(solution):
    # time.sleep(5)
    fit = f18.evaluate(solution)
    return fit

# print(type(fitness))

problem_dict1 = {
    "fit_func": fitness,
    "lb": f18.lb.tolist(),
    "ub": f18.ub.tolist(),
    "minmax": "min",
}

term_dict1 = {
    "max_epoch": 1000,
    "max_fe": 180000,  # 100000 number of function evaluation
    "max_time": 1000,  # 10 seconds to run the program
    "max_early_stop": 15  # 15 epochs if the best fitness is not getting better we stop the program
}

epoch = 1000
pop_size = 50

class Squared(Problem):
    def __init__(self, lb, ub, minmax, name="Squared", **kwargs):
        super().__init__(lb, ub, minmax, **kwargs)
        self.name = name

    def fit_func(self, solution):
        return np.sum(solution ** 2)


P1 = Squared(lb=[-10, ] * 100, ub=[10, ] * 100, minmax="min")

if __name__ == "__main__":
    model = WOA.OriginalWOA(epoch, pop_size)
    model = OriginalBMO(epoch, pop_size)
    model = OriginalTPO(epoch, pop_size)
    model = OriginalEHO(epoch, pop_size)
    model = OriginalESOA(epoch, pop_size)
    model = T1.BaseBBO(epoch, pop_size)
    model = T2.OriginalBBO(epoch, pop_size)
    model = BaseBBO(epoch, pop_size)
    best_position, best_fitness = model.solve(P1, mode="thread", n_workers=4, termination=term_dict1)

    print(best_position)
    print(model.get_parameters())
    print(model.get_attributes()["epoch"])
