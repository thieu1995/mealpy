#!/usr/bin/env python
# Created by "Thieu" at 19:52, 27/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy import IntegerVar, FloatVar, StringVar, PermutationVar, BinaryVar, BoolVar, Problem


# 1. IntegerVar and StringVar
bounds = [
    IntegerVar(lb=(-10, -15, -4), ub=(5, 7, 2), name="Pc"),
    StringVar(valid_sets=(("auto", "adaptive", "modified"),
                          (0.01, 0.1, "auto", 0.05, "curve")), name="alpha"),
    IntegerVar(lb=(0, 0), ub=(11, 15), name="Ld")
]

# 2. StringVar and FloatVar
bounds = [
    StringVar(valid_sets=(("auto", "adaptive", "modified"),
                          (0.01, 0.1, "auto", 0.05, "curve")), name="alpha"),
    FloatVar(lb=(1.5, -5.5, -4.0), ub=(15.5, 7.5, 2.5), name="Pc"),
    FloatVar(lb=(0.5, 0.2), ub=(18.0, 20.2), name="Ld")
]

# 3. BinaryVar and StringVar
bounds = [
    BinaryVar(n_vars=3, name="Pc"),
    StringVar(valid_sets=(("auto", "adaptive", "modified"),
                          (0.01, 0.1, "auto", 0.05, "curve")), name="alpha"),
    BinaryVar(n_vars=2, name="Ld")
]

# 4. IntegerVar, FloatVar, StringVar, and PermutationVar
bounds = [
    FloatVar(lb=(0.5, 0.2), ub=(18.0, 20.2), name="Lx"),
    IntegerVar(lb=(0, 0), ub=(11, 15), name="Ld"),
    StringVar(valid_sets=(("auto", "adaptive", "modified"),
                          (0.01, 0.1, "auto", 0.05, "curve")), name="alpha"),
    FloatVar(lb=(1.5, -5.5, -4.0), ub=(15.5, 7.5, 2.5), name="Pc"),
    PermutationVar(valid_set=(9, 4, 6, 2, 0), name="beta")
]

## 5. IntegerVar, FloatVar, StringVar, BinaryVar, BoolVar, and PermutationVar
bounds = [
    FloatVar(lb=(0.5, 0.2), ub=(18.0, 20.2), name="Lx"),
    IntegerVar(lb=(0, 0), ub=(11, 15), name="Ld"),
    StringVar(valid_sets=(("auto", "adaptive", "modified"),
                          (0.01, 0.1, "auto", 0.05, "curve")), name="alpha"),
    FloatVar(lb=(1.5, -5.5, -4.0), ub=(15.5, 7.5, 2.5), name="Pc"),
    PermutationVar(valid_set=(9, 4, 6, 2, 0), name="beta"),
    BinaryVar(n_vars=3, name="delta"),
    BoolVar(n_vars=2, name="gama")
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
