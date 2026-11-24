#!/usr/bin/env python
# Created by "Thieu" at 11:28, 22/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import FloatVar, WOA, Tuner
from mealpy.tuner import ParameterGrid
import numpy as np
import pytest


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def model():
    model = WOA.OriginalWOA(epoch=10, pop_size=50)
    return model


@pytest.fixture(scope="module")
def problem():
    def objective_function(solution):
        return np.sum(solution ** 2)

    problem = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-10, -10, -10, -10, -10], ub=[10, 10, 10, 10, 10]),
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
    def objective_function(solution):
        return np.sum(solution ** 2)

    problem = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-10, -15, -4, -2, -8], ub=[10, 15, 12, 8, 20]),
        "minmax": "min",
        "log_to": None,
    }
    model = WOA.OriginalWOA(epoch=10, pop_size=50)
    try:
        tuner = Tuner(model, para_grids)
        tuner.execute(problem)
        best = tuner.resolve()
        assert isinstance(best.solution, np.ndarray)
    except TypeError:
        assert True

@pytest.mark.parametrize("param_grid", [
    {"epoch": [10, 20], "pop_size": [50, 100, 150]},
    {"epoch": [10], "pop_size": np.array([50, 100]), "a": ["x", "y", "z"]},
    [{"epoch": [10, 20], "pop_size": [50, 100]}, {}, {"a": [1, 2, 3], "b": [True, False]}],
])
def test_parameter_grid_getitem_matches_iteration(param_grid):
    grid = ParameterGrid(param_grid)
    params = [grid[idx] for idx in range(len(grid))]
    assert params == list(grid)
    assert [list(p) for p in params] == [list(p) for p in grid]
    with pytest.raises(IndexError):
        grid[len(grid)]
