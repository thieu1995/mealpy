#!/usr/bin/env python
# Created by "Thieu" at 13:03, 18/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from opfunu.cec_based.cec2017 import F292017
from mealpy.bio_based import BBO
from mealpy.utils.problem import Problem


#### Solve problem with dictionary definition

# f18 = F292017(30, f_bias=0)
#
# def fitness(solution):
#     return f18.evaluate(solution)
#
# p1 = {
#     "lb": f18.lb.tolist(),
#     "ub": f18.ub.tolist(),
#     "minmax": "min",
#     "fit_func": fitness,
#     "name": "F18"
# }
#
# model = BBO.BaseBBO(epoch=10, pop_size=50)
# best_position, best_fitness = model.solve(p1)
# print(model.get_parameters())
# print(model.get_name())
# print(model.get_attributes()["mr"])


#### Solve problem with custom child class of Problem class.

class Squared(Problem):
    def __init__(self, lb, ub, minmax, name="Squared", **kwargs):
        super().__init__(lb, ub, minmax, **kwargs)
        self.name = name

    def fit_func(self, solution):
        return [np.sum(solution ** 2), np.sum(solution[:5]**3)]

# p2 = Squared(lb=[-10, ] * 20, ub=[10, ] * 20, minmax="min", obj_weights=[0.5, 0.5])
# model = BBO.BaseBBO(epoch=10, pop_size=50)
# best_position, best_fitness = model.solve(p2)
# print(model.get_parameters())
# print(model.get_name())
# print(model.get_attributes()["mr"])


#### Solve multiple problem using the same model

# p3 = Squared(lb=[-10, ] * 20, ub=[10, ] * 20, minmax="min", obj_weights=[0.2, 0.8])
# p4 = Squared(lb=[-10, ] * 20, ub=[10, ] * 20, minmax="min", obj_weights=[0.7, 0.3])
# p5 = Squared(lb=[-10, ] * 20, ub=[10, ] * 20, minmax="min", obj_weights=[1.0, 2.5])
#
# model = BBO.BaseBBO(epoch=10, pop_size=50)
#
# for prob in [p3, p4, p5]:
#     best_position, best_fitness = model.solve(prob)
#     print(model.get_parameters())
#     print(model.get_name())
#     print(model.get_attributes()["mr"])



#### Solve problem with difference termination

p6 = Squared(lb=[-10, ] * 20, ub=[10, ] * 20, minmax="min", obj_weights=[0.2, 0.8])

term1 = {
    "mode": "FE",
    "quantity": 10000,
}
term2 = {
    "mode": "ES",
    "quantity": 20,
}

model = BBO.BaseBBO(epoch=10, pop_size=50)


for term in [term1, term2]:
    best_position, best_fitness = model.solve(p6, termination=term)
    print(model.get_parameters())
    print(model.get_name())
    print(model.problem.get_name())
    print(model.get_attributes()["solution"])
