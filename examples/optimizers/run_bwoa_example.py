#!/usr/bin/env python
# Created by "Thieu" at 18:37, 27/10/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy import FloatVar, BWOA, APO, GRSA, KLA


def objective_function(solution):
    return np.sum(solution ** 2)


problem = {
    "bounds": FloatVar(lb=(-10.0,) * 30, ub=(10.0,) * 30, name="x"),
    "minmax": "min",
    "obj_func": objective_function,
    "name": "Sphere",
}

model = BWOA.OriginalBWOA(epoch=100, pop_size=50, pp=0.6, cr=0.44, pm=0.4)
model = APO.OriginalAPO(epoch=100, pop_size=50, pf_max=0.1, n_pairs=2)
model = GRSA.OriginalGRSA(epoch=100, pop_size=50, n_geometry=5, w_max=0.9, w_min=0.4, g_max=0.5, g_min=0.1)
model = KLA.OriginalKLA(epoch=100, pop_size=50)
g_best = model.solve(problem, seed=10)
print(f"Best fitness: {g_best.target.fitness}")
