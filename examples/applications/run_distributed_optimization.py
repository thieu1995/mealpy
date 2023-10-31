#!/usr/bin/env python
# Created by "Thieu" at 15:19, 31/10/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import FloatVar, SMA
import numpy as np


def objective_function(solution):
    return np.sum(solution**2)

problem = {
    "obj_func": objective_function,
    "bounds": FloatVar(lb=(-100., )*100, ub=(100., )*100),
    "minmax": "min",
    "log_to": "console",
}

## Run distributed SMA algorithm using 10 threads
optimizer = SMA.OriginalSMA(epoch=10000, pop_size=100, pr=0.03)
optimizer.solve(problem, mode="thread", n_workers=10)        # Distributed to 10 threads
print(f"Best solution: {optimizer.g_best.solution}, Best fitness: {optimizer.g_best.target.fitness}")

## Run distributed SMA algorithm using 8 CPUs (cores)
optimizer.solve(problem, mode="process", n_workers=8)        # Distributed to 8 cores
print(f"Best solution: {optimizer.g_best.solution}, Best fitness: {optimizer.g_best.target.fitness}")
