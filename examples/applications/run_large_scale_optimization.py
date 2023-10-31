#!/usr/bin/env python
# Created by "Thieu" at 15:22, 31/10/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import FloatVar, SHADE
import numpy as np

def objective_function(solution):
    return np.sum(solution**2)

problem = {
    "obj_func": objective_function,
    "bounds": FloatVar(lb=(-1000., )*10000, ub=(1000.,)*10000),     # 10000 dimensions
    "minmax": "min",
    "log_to": "console",
}

## Run the algorithm
optimizer = SHADE.OriginalSHADE(epoch=10000, pop_size=100)
g_best = optimizer.solve(problem)
print(f"Best solution: {g_best.solution}, Best fitness: {g_best.target.fitness}")
