#!/usr/bin/env python
# Created by "Thieu" at 09:35, 17/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.evolutionary_based import DE
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


def test_DE_results(problem):
    models = [
        DE.BaseDE(epoch=10, pop_size=50, wf=0.1, cr=0.9, strategy=5),
        DE.JADE(epoch=20, pop_size=50, miu_f=0.5, miu_cr=0.5, pt=0.1, ap=0.1),
        DE.SADE(epoch=20, pop_size=50),
        DE.SHADE(epoch=20, pop_size=50, miu_f=0.5, miu_cr=0.5),
        DE.L_SHADE(epoch=20, pop_size=50, miu_f=0.5, miu_cr=0.5),
        DE.SAP_DE(epoch=20, pop_size=50, branch="ABS"),
    ]
    for model in models:
        best_position, best_fitness = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(best_position, np.ndarray)
        assert len(best_position) == len(problem["lb"])

