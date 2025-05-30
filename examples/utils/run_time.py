#!/usr/bin/env python
# Created by "Thieu" at 23:26, 29/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import time
import numpy as np
from mealpy import FloatVar, BBO, GA, PSO


def fitness(solution):
    """Example objective function (e.g., sphere function)"""
    return np.sum(solution**2)


problem_dict1 = {
    "obj_func": fitness,
    "bounds": FloatVar(lb=(-100.,)*30, ub=(100.,)*30),
    "minmax": "min",
}

termination = {
    "max_time": 0.2,  # Maximum time in seconds
}

# Test BBO
t1 = time.time()
model_bbo = BBO.OriginalBBO(epoch=1000, pop_size=50)
g_best = model_bbo.solve(problem_dict1, termination=termination)
print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
print(f"Time taken: {time.time() - t1:.6f} seconds")

# Test GA
t2 = time.time()
model_ga = GA.BaseGA(epoch=1000, pop_size=50)
g_best_ga = model_ga.solve(problem_dict1, termination=termination)
print(f"Solution: {g_best_ga.solution}, Fitness: {g_best_ga.target.fitness}")
print(f"Time taken: {time.time() - t2:.6f} seconds")

# Test PSO
t3 = time.time()
model_pso = PSO.OriginalPSO(epoch=1000, pop_size=50)
g_best_pso = model_pso.solve(problem_dict1, termination=termination)
print(f"Solution: {g_best_pso.solution}, Fitness: {g_best_pso.target.fitness}")
print(f"Time taken: {time.time() - t3:.6f} seconds")
