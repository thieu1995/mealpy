#!/usr/bin/env python
# Created by "Thieu" at 16:19, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
#
# Examples:
# >>>
# >>> from mealpy.swarm_based import PSO
# >>> import numpy as np
# >>>
# >>> def fitness_function(solution):
# >>>     return np.sum(solution ** 2)
# >>>
# >>> problem = {
# >>>    "fit_func": fitness_function,
# >>>    "lb": [-100, ] * 30,
# >>>    "ub": [100, ] * 30,
# >>>    "minmax": "min",
# >>>    "save_population": True,
# >>>    "log_to": "file",
# >>>    "log_file": "mealpy.log",
# >>>    "name": Square",
# >>> }
# >>>
# >>> ## Run the algorithm
# >>> model = PSO.C_PSO(epoch=5, pop_size=50, name="C-PSO")
# >>> best_position, best_fitness = model.solve(problem)
# >>> print(f"Best solution: {best_position}, Best fitness: {best_fitness}")


__version__ = "2.5.1"

from . import bio_based
from . import evolutionary_based
from . import human_based
from . import math_based
from . import music_based
from . import physics_based
from . import swarm_based
from . import system_based
from . import utils
from . import tuner
from . import multitask
from . import optimizer
