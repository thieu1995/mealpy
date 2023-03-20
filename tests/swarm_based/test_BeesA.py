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

        BeesA.ProbBeesA(epoch=10, pop_size=10, recruited_bee_ratio=0.1, dance_radius=0.1, dance_reduction=0.99),
        BeesA.OriginalBeesA(epoch=10, pop_size=10, selected_site_ratio=0.5, elite_site_ratio=0.4, selected_site_bee_ratio=0.1, elite_site_bee_ratio=2.0,
                            dance_radius=0.1, dance_reduction=0.99, ),
    ]
    for model in models:
        best_position, best_fitness = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(best_position, np.ndarray)
        assert len(best_position) == len(problem["lb"])
