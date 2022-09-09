#!/usr/bin/env python
# Created by "Thieu" at 17:56, 06/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from opfunu.cec_based.cec2017 import F292017
from mealpy.swarm_based import PSO
from mealpy.math_based import GBO, PSS
from mealpy.physics_based import HGSO
from mealpy.human_based import SARO


f18 = F292017(ndim=30, f_bias=0)
problem_dict1 = {
    "fit_func": f18.evaluate,
    "lb": f18.lb.tolist(),
    "ub": f18.ub.tolist(),
    "minmax": "min",
}

## Define a model 1 time, and train model multiple times with different stopping conditions
model = HGSO.OriginalHGSO(epoch=200, pop_size=50)

## 1. Epoch/Maximum Generation (default)
term_dict1 = {
    "mode": "MG",
    "quantity": 50  # 100000 number of function evaluation
}
best_position, best_fitness = model.solve(termination=term_dict1)
print(f"Solution: {best_position}, Fitness: {best_fitness}")


## 2. Number of Function Evaluation
term_dict2 = {
    "mode": "FE",
    "quantity": 2000  # 2000 number of function evaluation
}
best_position, best_fitness = model.solve(termination=term_dict2)
print(f"Solution: {best_position}, Fitness: {best_fitness}")


## 3. Time bound
term_dict3 = {
    "mode": "TB",
    "quantity": 15  # 15 seconds
}
best_position, best_fitness = model.solve(termination=term_dict3)
print(f"Solution: {best_position}, Fitness: {best_fitness}")


## 4. Early Stopping
term_dict4 = {
    "mode": "ES",
    "quantity": 10  # 10 patients
}
best_position, best_fitness = model.solve(termination=term_dict4)
print(f"Solution: {best_position}, Fitness: {best_fitness}")
