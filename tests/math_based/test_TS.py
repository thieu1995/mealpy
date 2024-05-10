#!/usr/bin/env python
# Created by "Thieu" at 08:22, 17/06/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from unittest import mock

from mealpy import FloatVar, TS, Optimizer
import numpy as np
import pytest


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def problem():
    def objective_function(solution):
        return np.sum(solution ** 2)

    problem = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-10, -15, -4, -2, -8], ub=[10, 15, 12, 8, 20]),
        "minmax": "min",
    }
    return problem


def test_TS_results(problem):
    models = [
        TS.OriginalTS(epoch=100, pop_size=2, tabu_size=5, neighbour_size=10, perturbation_scale=0.05),
    ]
    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)


def test_TS_no_candidates(problem):
    """
    Test that TS.OriginalTS does not break when evolve yields no candidates
    because of closeness to the current position or filtered out by the tabu list.
    """
    ts_optimizer = TS.OriginalTS(epoch=1)

    def allclose_mock(*args, **kwargs):
        return True

    with mock.patch('mealpy.math_based.TS.np.allclose', new=allclose_mock):
        ts_optimizer.solve(problem)
