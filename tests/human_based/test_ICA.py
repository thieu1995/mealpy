#!/usr/bin/env python
# Created by "Thieu" at 11:23, 18/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.human_based import ICA
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
        "log_to": None,
    }
    return problem


def test_ICA_results(problem):
    models = [
        ICA.BaseICA(problem, epoch=10, pop_size=50, empire_count=5, assimilation_coeff=1.5,
                    revolution_prob=0.05, revolution_rate=0.1, revolution_step_size=0.1, zeta=0.1)
    ]
    for model in models:
        best_position, best_fitness = model.solve()
        assert isinstance(model, Optimizer)
        assert isinstance(best_position, np.ndarray)
        assert len(best_position) == len(problem["lb"])

