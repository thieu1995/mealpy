#!/usr/bin/env python
# Created by "Thieu" at 10:02, 06/01/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest
from mealpy import FloatVar, AHO


@pytest.fixture(scope="module")
def problem():
    def objective_function(solution):
        return np.sum(solution ** 2)

    return {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-100., ] * 30, ub=[100., ] * 30),
        "minmax": "min",
        "log_to": None,
    }


def test_OriginalAHO_results(problem):
    epoch = 10
    pop_size = 50
    model = AHO.OriginalAHO(epoch=epoch, pop_size=pop_size)
    g_best = model.solve(problem)
    
    assert isinstance(model.get_parameters(), dict)
    assert isinstance(g_best.solution, np.ndarray)
    assert len(g_best.solution) == 30


def test_OriginalAHO_with_custom_params(problem):
    epoch = 10
    pop_size = 50
    theta = np.pi / 6
    omega = 0.05
    model = AHO.OriginalAHO(epoch=epoch, pop_size=pop_size, theta=theta, omega=omega)
    g_best = model.solve(problem)
    
    assert isinstance(g_best.solution, np.ndarray)
    assert model.theta == theta
    assert model.omega == omega