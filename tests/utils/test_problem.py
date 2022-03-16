#!/usr/bin/env python
# Created by "Thieu" at 07:57, 16/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.utils.problem import Problem
import numpy as np
import pytest


def fitness_function(solution):
    return np.sum(solution[1:]**2) - solution[0]


def fit_function(solution):
    return np.sum(solution ** 2)


def amend_position(solution, lb, ub):
    return np.clip(solution, lb, ub)


problem = {
    "fit_func": fit_function,
    "lb": [-10, -10, -10, -10, -10],
    "ub": [10, 10, 10, 10, 10],
    "minmax": "min",
    "log_to": None,
    "log_file": "records.log",
    "save_population": False,
    "obj_weights": [0.3, 0.5, 0.2],
    "amend_position": amend_position
}


@pytest.mark.parametrize("my_problem, my_func, system_code",
                         [
                             (problem, None, 0),
                             (problem, "hello", 0),
                             (problem, -10, 0),
                             (problem, [10], 0),
                             (problem, (0, 9), 0),
                         ])
def test_fit_func(my_problem, my_func, system_code):
    my_problem["fit_func"] = my_func
    with pytest.raises(SystemExit) as e:
        prob = Problem(problem=my_problem)
    assert e.type == SystemExit
    assert e.value.code == system_code
