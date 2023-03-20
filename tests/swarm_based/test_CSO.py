#!/usr/bin/env python
# Created by "Thieu" at 14:56, 20/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.swarm_based import CSO
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


def test_CSO_results(problem):
    models = [
        CSO.OriginalCSO(epoch=10, pop_size=50, mixture_ratio=0.15, smp=5, spc=False, cdc=0.8, srd=0.15, c1=0.4, w_minmax=(0.4, 0.9), selected_strategy=1)]
    for model in models:
        best_position, best_fitness = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(best_position, np.ndarray)
        assert len(best_position) == len(problem["lb"])
