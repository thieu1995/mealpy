#!/usr/bin/env python
# Created by "Thieu" at 11:28, 22/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.swarm_based import WOA
from mealpy.tuner import Tuner
import numpy as np
import pytest


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def model():
    model = WOA.OriginalWOA(epoch=10, pop_size=50)
    return model


@pytest.fixture(scope="module")
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


def test_amend_position(model, problem):
    para_grids = {
        "epoch": [10, 20],
        "pop_size": [50, 100]
    }
    tuner = Tuner(model, para_grids)
    tuner.execute(problem)
    tuner.resolve()


@pytest.mark.parametrize("para_grids",
                         [
                             ({"epoch": [10, 20], "pop_size": [50, 100]}),
                             ({"epoch": (10,), "pop_size": (50,)}),
                             ({"epoch": 10, "pop_size": 50}),
                         ])
def test_para_grids(problem, para_grids, request):
    def fitness_function(solution):
        return np.sum(solution ** 2)

    problem = {
        "fit_func": fitness_function,
        "lb": [-10, -10, -10, -10, -10],
        "ub": [10, 10, 10, 10, 10],
        "minmax": "min",
        "log_to": None,
    }
    model = WOA.OriginalWOA(epoch=10, pop_size=50)
    try:
        tuner = Tuner(model, para_grids)
        tuner.execute(problem)
        pos, fit = tuner.resolve()
        assert type(pos) == np.ndarray
    except TypeError:
        assert True
