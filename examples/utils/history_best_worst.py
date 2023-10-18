#!/usr/bin/env python
# Created by "Thieu" at 23:53, 27/08/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy import FloatVar, BBO


problem = {
    "obj_func": lambda sol: np.sum(sol**2),
    "bounds": FloatVar(lb=(-100.,)*20, ub=(100.,)*20, name="delta"),
    "minmax": "min",
}

## Run the algorithm
model = BBO.OriginalBBO(epoch=100, pop_size=50)
g_best = model.solve(problem)
print(f"Best solution: {g_best.solution}, Best fitness: {g_best.target.fitness}")

print("List current worst")
print(model.history.list_current_worst)

print("List global worst")
print(model.history.list_global_worst)
