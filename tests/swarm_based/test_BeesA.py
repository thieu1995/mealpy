#!/usr/bin/env python
# Created by "Thieu" at 11:41, 20/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.swarm_based import BeesA
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


def test_BeesA_results(problem):
    models = [
        BeesA.ProbBeesA(problem, epoch=10, pop_size=50, recruited_bee_ratio=0.1, dance_factor=(0.1, 0.99)),
        BeesA.BaseBeesA(problem, epoch=10, pop_size=50, site_ratio=(0.5, 0.4), site_bee_ratio=(0.1, 2.0), dance_factor=(0.1, 0.99)),
    ]
    for model in models:
        best_position, best_fitness = model.solve()
        assert isinstance(model, Optimizer)
        assert isinstance(best_position, np.ndarray)
        assert len(best_position) == len(problem["lb"])
