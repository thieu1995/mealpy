#!/usr/bin/env python
# Created by "Thieu" at 12:13, 20/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.swarm_based import BFO
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


def test_BFO_results(problem):
    models = [
        BFO.OriginalBFO(epoch=10, pop_size=50, Ci=0.01, Ped=0.25, Nc=5, Ns=4, attract_repels=(0.1, 0.2, 0.1, 10)),
        BFO.ABFO(epoch=10, pop_size=50, Ci=(0.1, 0.001), Ped=0.01, Ns=4, N_minmax=(1, 40)),
    ]
    for model in models:
        best_position, best_fitness = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(best_position, np.ndarray)
        assert len(best_position) == len(problem["lb"])
