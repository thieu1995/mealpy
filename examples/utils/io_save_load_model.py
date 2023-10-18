#!/usr/bin/env python
# Created by "Thieu" at 20:17, 27/08/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy import FloatVar, BBO
from mealpy.utils import io


def objective_function(solution):
    return np.sum(solution**2)


problem = {
    "obj_func": objective_function,
    "bounds": FloatVar(lb=(-100.,)*20, ub=(100.,)*20, name="delta"),
    "minmax": "min",
}

## Run the algorithm
model = BBO.OriginalBBO(epoch=100, pop_size=50)
g_best = model.solve(problem)
print(f"Best solution: {g_best.solution}, Best fitness: {g_best.target.fitness}")

io.save_model(model, "model.pkl")

new_model = io.load_model("model.pkl")
print(f"Best solution: {new_model.g_best.solution}, Best fitness: {new_model.g_best.target.fitness}")
