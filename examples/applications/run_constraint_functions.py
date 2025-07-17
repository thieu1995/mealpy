#!/usr/bin/env python
# Created by "Thieu" at 17:33, 06/11/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from mealpy import FloatVar, SMA
import numpy as np


## Link: https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119136507.app2
def objective_function(solution):
    def g1(x):
        return 2*x[0] + 2*x[1] + x[9] + x[10] - 10
    def g2(x):
        return 2 * x[0] + 2 * x[2] + x[9] + x[10] - 10
    def g3(x):
        return 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10
    def g4(x):
        return -8*x[0] + x[9]
    def g5(x):
        return -8*x[1] + x[10]
    def g6(x):
        return -8*x[2] + x[11]
    def g7(x):
        return -2*x[3] - x[4] + x[9]
    def g8(x):
        return -2*x[5] - x[6] + x[10]
    def g9(x):
        return -2*x[7] - x[8] + x[11]

    def violate(value):
        return 0 if value <= 0 else value

    fx = 5 * np.sum(solution[:4]) - 5*np.sum(solution[:4]**2) - np.sum(solution[4:13])

    ## Increase the punishment for g1 and g4 to boost the algorithm (You can choice any constraint instead of g1 and g4)
    fx += violate(g1(solution))**2 + violate(g2(solution)) + violate(g3(solution)) + \
            2*violate(g4(solution)) + violate(g5(solution)) + violate(g6(solution))+ \
            violate(g7(solution)) + violate(g8(solution)) + violate(g9(solution))
    return fx

problem = {
    "obj_func": objective_function,
    "bounds": FloatVar(lb=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ub=[1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 1]),
    "minmax": "min",
    "log_to": "console"
}

## Run the algorithm
optimizer = SMA.OriginalSMA(epoch=100, pop_size=50, pr=0.03)
optimizer.solve(problem)
print(f"Best solution: {optimizer.g_best.solution}, Best fitness: {optimizer.g_best.target.fitness}")
