#!/usr/bin/env python
# Created by "Thieu" at 20:21, 19/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.physics_based import TWO
from mealpy.optimizer import Optimizer
import numpy as np
import pytest


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def problem():
    def fitness_function(solution):
        return np.sum(solution ** 2)

    problem = {
        "fit_func": fitness_function,
        "lb": [-10, -10, -10, -10, -10],
        "ub": [10, 10, 10, 10, 10],
        "minmax": "min",
        "log_to": None
    }
    return problem


def test_TWO_results(problem):
    models = [
        TWO.BaseTWO(problem, epoch=100, pop_size=50),
        TWO.OppoTWO(problem, epoch=100, pop_size=50),
        TWO.LevyTWO(problem, epoch=100, pop_size=50),
        TWO.EnhancedTWO(problem, epoch=100, pop_size=50),
    ]
    for model in models:
        best_position, best_fitness = model.solve()
        assert isinstance(model, Optimizer)
        assert isinstance(best_position, np.ndarray)
        assert len(best_position) == len(problem["lb"])
