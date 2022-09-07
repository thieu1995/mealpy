#!/usr/bin/env python
# Created by "Thieu" at 17:31, 07/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.bio_based import BBO
from mealpy.utils.termination import Termination


#### Pass termination as dictionary python

def fitness(solution):
    return np.sum(solution**2)

p1 = {
    "lb": [-30] * 10,
    "ub": [30] * 10,
    "minmax": "min",
    "fit_func": fitness,
    "name": "Test Function"
}

term1 = {
    "mode": "FE",
    "quantity": 10000,
}

model = BBO.BaseBBO(epoch=10, pop_size=50)
# best_position, best_fitness = model.solve(p1, termination=term1)
# print(model.get_parameters())
# print(model.get_name())
# print(model.problem.get_name())
# print(model.termination.get_name())
# print(model.get_attributes()["solution"])


#### Pass termination as an instance of Termination class

term2 = Termination(mode="TB", quantity=5)
best_position, best_fitness = model.solve(p1, termination=term2)
print(model.get_parameters())
print(model.get_name())
print(model.problem.get_name())
print(model.termination.get_name())
print(model.get_attributes()["solution"])
