#!/usr/bin/env python
# Created by "Thieu" at 17:31, 07/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy import FloatVar, BBO, Termination
from opfunu.cec_based.cec2017 import F292017

## 1) Single termination
f18 = F292017(ndim=30)

problem_dict1 = {
    "obj_func": f18.evaluate,
    "bounds": FloatVar(lb=f18.lb, ub=f18.ub),
    "minmax": "min",
}

## Define a model 1 time, and train model multiple times with different stopping conditions
model = BBO.OriginalBBO(epoch=200, pop_size=50)

## 1. Epoch/Maximum Generation (default)
g_best = model.solve(problem_dict1, termination={"max_epoch": 100})
print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")


## 2. Number of Function Evaluation
g_best = model.solve(problem_dict1, termination={"max_fe": 10000})
print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")


## 3. Time bound
g_best = model.solve(problem_dict1, termination={"max_time": 5.5})
print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")


## 4. Early Stopping
g_best = model.solve(problem_dict1, termination={"max_early_stop": 5})
print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")


## 2) Combine all of them
# Define an example objective function (e.g., sphere function)
def fitness(solution):
    return np.sum(solution**2)


term_dict = {
    "max_epoch": 100,
    "max_fe": 2000,  # 2000 number of function evaluation
    "max_time": 1.5,     # 1.5 seconds to run the program
    "max_early_stop": 15    # 15 epochs if the best fitness is not getting better we stop the program
}

# Define the problem dimension and search space (e.g., for a 30-D problem with [-10, 10] bounds for each variable)
p1 = {
    "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    "minmax": "min",
    "obj_func": fitness,
    "name": "Test Function"
}

#### Pass termination as dictionary python
model = BBO.OriginalBBO(epoch=10, pop_size=50)
best_agent = model.solve(p1, termination=term_dict)
print(best_agent.solution)
print(best_agent.target.fitness)
print(model.get_parameters())
print(model.get_name())
print(model.problem.get_name())
print(model.termination.get_name())
print(model.get_attributes()["g_best"])

#### Pass termination as an instance of Termination class
term2 = Termination(max_epoch=10, max_time=1.5)
best_agent = model.solve(p1, termination=term2)
print(best_agent.solution)
print(best_agent.target.fitness)
print(model.get_parameters())
print(model.get_name())
print(model.problem.get_name())
print(model.termination.get_name())
print(model.get_attributes()["g_best"])
