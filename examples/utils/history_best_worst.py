#!/usr/bin/env python
# Created by "Thieu" at 23:53, 27/08/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.evolutionary_based.GA import BaseGA


def fitness_function(solution):
    ## Define your own fitness function
    return np.sum(solution**2)


problem = {
    "fit_func": fitness_function,
    "lb": [-100, ] * 50,
    "ub": [100, ] * 50,
    "minmax": "min",
}

## Run the algorithm
model = BaseGA(epoch=100, pop_size=50)
best_position, best_fitness = model.solve(problem)
print(f"Best solution: {best_position}, Best fitness: {best_fitness}")

print("List current worst")
print(model.history.list_current_worst)

print("List global worst")
print(model.history.list_global_worst)




