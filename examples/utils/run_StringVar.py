#!/usr/bin/env python
# Created by "Thieu" at 13:34, 27/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy import StringVar, Problem


## 1. Show error
# bounds = [
#     StringVar(n_vars=-2, valid_sets=("auto", "hello", "case", "nano")),
# ]

## 2. One variable
# bounds = [
#     StringVar(n_vars=1, valid_sets=("auto", 12, "hello", 5, "case", "nano", 1)),
# ]

## 3. Multiple variables
bounds = [
    StringVar(4, (("auto", "backward", "forward"),
                  ("leaf", "branch", "root"),
                  (0.01, "auto", 0.1, "adaptive", 0.05, "modified"),
                  ("random", "tournament", "roulette", "round-robin")), name="delta")
]

problem = Problem(bounds, fit_func=lambda sol: np.sum(sol**2))
print(f"Problem: {problem}")
print(f"Bounds: {problem.bounds}")

## Generate encoded solution (the real-value solution)
x = problem.generate_solution()
print(x)

x = problem.generate_solution(encoded=False)    # Real world (actual solution - decoded solution) for the problem
x1 = problem.encode_solution(x)                 # Optimizer solution (encoded solution) for the problem
x2 = problem.correct_solution(x1)               # Correct the solution (encoded and bounded solution) for the problem
x3 = problem.decode_solution(x1)                # Real world (actual solution - decoded solution) for the problem
print(f"Real value solution: {x}")
print(f"Encoded solution: {x1}")
print(f"Bounded solution: {x2}")
print(f"Real value solution after decoded: {x3}")
