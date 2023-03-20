#!/usr/bin/env python
# Created by "Thieu" at 00:36, 17/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.bio_based import VCS
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
    }
    return problem


def test_OriginalVCS_results(problem):
    epoch = 10
    pop_size = 50
    lamda = 0.5
    xichma = 0.3
    model = VCS.OriginalVCS(epoch, pop_size, lamda, xichma)
    best_position, best_fitness = model.solve(problem)
    assert isinstance(model, Optimizer)
    assert isinstance(best_position, np.ndarray)
    assert len(best_position) == len(problem["lb"])


def test_BaseVCS_results(problem):
    epoch = 10
    pop_size = 50
    lamda = 0.5
    xichma = 0.3
    model = VCS.BaseVCS(epoch, pop_size, lamda, xichma)
    best_position, best_fitness = model.solve(problem)
    assert isinstance(model, Optimizer)
    assert isinstance(best_position, np.ndarray)
    assert len(best_position) == len(problem["lb"])


@pytest.mark.parametrize("problem, epoch, system_code",
                         [
                             (problem, None, 0),
                             (problem, "hello", 0),
                             (problem, -10, 0),
                             (problem, [10], 0),
                             (problem, (0, 9), 0),
                             (problem, 0, 0),
                             (problem, float("inf"), 0),
                         ])
def test_epoch_VCS(problem, epoch, system_code):
    pop_size = 50
    algorithms = [VCS.OriginalVCS, VCS.BaseVCS]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(epoch, pop_size)
        assert e.type == ValueError


@pytest.mark.parametrize("problem, pop_size, system_code",
                         [
                             (problem, None, 0),
                             (problem, "hello", 0),
                             (problem, -10, 0),
                             (problem, [10], 0),
                             (problem, (0, 9), 0),
                             (problem, 0, 0),
                             (problem, float("inf"), 0),
                         ])
def test_pop_size_VCS(problem, pop_size, system_code):
    epoch = 10
    algorithms = [VCS.OriginalVCS, VCS.BaseVCS]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(epoch, pop_size)
        assert e.type == ValueError


@pytest.mark.parametrize("problem, lamda, system_code",
                         [
                             (problem, None, 0),
                             (problem, "hello", 0),
                             (problem, -1.0, 0),
                             (problem, [10], 0),
                             (problem, (0, 9), 0),
                             (problem, 0, 0),
                             (problem, 1, 0),
                             (problem, 1.1, 0),
                             (problem, -0.01, 0),
                         ])
def test_lamda_VCS(problem, lamda, system_code):
    epoch = 10
    pop_size = 50
    algorithms = [VCS.OriginalVCS, VCS.BaseVCS]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(epoch, pop_size, lamda=lamda)
        assert e.type == ValueError
