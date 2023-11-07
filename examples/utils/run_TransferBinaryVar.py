#!/usr/bin/env python
# Created by "Thieu" at 20:30, 27/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy import TransferBinaryVar, Problem

print(f"Supported transfer functions: {TransferBinaryVar.SUPPORTED_TF_FUNCS}")

# ## 1. Show error
# bounds = [
#     TransferBinaryVar(n_vars=-2, name="delta", tf_func="vstf_01"),
# ]

## 2. One variable
# bounds = [
#     TransferBinaryVar(n_vars=1, name="delta", tf_func="vstf_01"),
# ]

# ## 3. Multiple variables
bounds = [
    TransferBinaryVar(n_vars=11, name="delta", tf_func="sstf_02"),
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

## Supported feature selection by setting all_zeros=False
bounds = [
    TransferBinaryVar(n_vars=2, name="delta", tf_func="sstf_02", all_zeros=False),
]
problem = Problem(bounds, obj_func=lambda sol: np.sum(sol**2))
print(problem.generate_solution(encoded=False))
## It will never generate a solution with all variables = 0.
