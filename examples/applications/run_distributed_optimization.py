#!/usr/bin/env python
# Created by "Thieu" at 15:19, 31/10/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

"""
### Distributed Optimization / Parallelization Optimization

For a deeper understanding of parallelization and distributed optimization concepts in metaheuristics,
we recommend reading the comprehensive article: [MEALPY: An open-source library for latest meta-heuristic algorithms in Python](https://doi.org/10.1016/j.sysarc.2023.102871)

It's important to note that not all metaheuristics can be run in parallel. MEALPY provides built-in support for
distributed execution where applicable, allowing you to leverage multiple threads or CPU cores to speed up computation.
"""


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
