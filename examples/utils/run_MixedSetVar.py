#!/usr/bin/env python
# Created by "Thieu" at 10:19, 31/10/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy import MixedSetVar, Problem


## 1. One variable
bounds = [
    MixedSetVar(valid_sets=(0.1, "auto", 1.4, "hello", True, "case", "nano", 0, 2)),
]

## 2. Multiple variables
bounds = [
    MixedSetVar(valid_sets=(("auto", 2, 3, "backward", "forward", True),
                          (1, 0, 10, "leaf", "branch", "root", False),
                          (0.01, "auto", 0.1, "adaptive", 0.05, "modified"),
                          ("random", 0, 2, 4, "tournament", "roulette", "round-robin")), name="delta"),
    MixedSetVar(valid_sets=(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000), name="epoch")
]

problem = Problem(bounds, obj_func=lambda sol: np.sum(sol**2))
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
