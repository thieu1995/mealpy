#!/usr/bin/env python
# Created by "Thieu" at 10:30, 02/01/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy import FloatVar, MGOA


def objective_function(solution):
    return np.sum(solution ** 2)


problem = {
    "bounds": FloatVar(lb=(-100.0,) * 30, ub=(100.0,) * 30, name="delta"),
    "minmax": "min",
    "obj_func": objective_function
}

model = MGOA.OriginalMGOA(epoch=200, pop_size=50)
g_best = model.solve(problem, seed=10)
print(f"Best fitness: {g_best.target.fitness}")
print(f"Best solution shape: {g_best.solution.shape}")
