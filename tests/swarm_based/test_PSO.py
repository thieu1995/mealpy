#!/usr/bin/env python
# Created by "Thieu" at 10:54, 21/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.swarm_based import PSO
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


def test_PSO_results(problem):
    models = [
        PSO.BasePSO(problem, epoch=100, pop_size=50, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9),
        PSO.C_PSO(problem, epoch=10, pop_size=50, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9),
        PSO.CL_PSO(problem, epoch=10, pop_size=50, c_local=1.2, w_min=0.4, w_max=0.9, max_flag=7),
        PSO.PPSO(problem, epoch=10, pop_size=50),
        PSO.HPSO_TVAC(problem, epoch=10, pop_size=50, ci=0.5, cf=0.2)
    ]
    for model in models:
        best_position, best_fitness = model.solve()
        assert isinstance(model, Optimizer)
        assert isinstance(best_position, np.ndarray)
        assert len(best_position) == len(problem["lb"])
