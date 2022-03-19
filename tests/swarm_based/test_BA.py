#!/usr/bin/env python
# Created by "Thieu" at 01:14, 20/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.swarm_based import BA
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


def test_BA_results(problem):
    models = [
        BA.OriginalBA(problem, epoch=10, pop_size=50, loudness=0.8, pulse_rate=0.95, pulse_frequency=(0, 10)),
        BA.BaseBA(problem, epoch=10, pop_size=50, loudness=(1.0, 2.0), pulse_rate=(0.15, 0.85), pulse_frequency=(0, 10)),
        BA.ModifiedBA(problem, epoch=10, pop_size=50, pulse_rate=0.95, pulse_frequency=(0, 10))
    ]
    for model in models:
        best_position, best_fitness = model.solve()
        assert isinstance(model, Optimizer)
        assert isinstance(best_position, np.ndarray)
        assert len(best_position) == len(problem["lb"])
