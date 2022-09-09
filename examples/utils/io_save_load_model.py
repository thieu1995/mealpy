#!/usr/bin/env python
# Created by "Thieu" at 20:17, 27/08/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.utils import io


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

io.save_model(model, "model.pkl")

new_model = io.load_model("model.pkl")
print(f"Best solution: {new_model.solution[0]}, Best fitness: {new_model.solution[1][0]}")
