#!/usr/bin/env python
# Created by "Thieu" at 23:02, 19/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.system_based import AEO
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
    }
    return problem


def test_AEO_results(problem):
    models = [
        AEO.AugmentedAEO(epoch=100, pop_size=50),
        AEO.OriginalAEO(epoch=10, pop_size=50),
        AEO.ModifiedAEO(epoch=10, pop_size=50),
        AEO.EnhancedAEO(epoch=10, pop_size=50),
        AEO.ImprovedAEO(epoch=10, pop_size=50)
    ]
    for model in models:
        best_position, best_fitness = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(best_position, np.ndarray)
        assert len(best_position) == len(problem["lb"])
