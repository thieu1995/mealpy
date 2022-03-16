#!/usr/bin/env python
# Created by "Thieu" at 00:16, 15/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.bio_based import BBO
from mealpy.optimizer import Optimizer
import numpy as np
import pytest


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def problem():
    def fitness_function(solution):
        return np.sum(solution ** 2)
    problem = {
        "fit_func": fitness_function,
        "lb": [-10, -15, -4, -2, -8],
        "ub": [10, 15, 12, 8, 20],
        "minmax": "min",
        "log_to": None,
    }
    return problem


def test_OriginalBBO_results(problem):
    epoch = 10
    pop_size = 50
    p_m = 0.01
    elites = 2
    model = BBO.OriginalBBO(problem, epoch, pop_size, p_m, elites)
    best_position, best_fitness = model.solve()
    assert isinstance(model, Optimizer)
    assert isinstance(best_position, np.ndarray)
    assert len(best_position) == len(problem["lb"])


def test_BaseBBO_results(problem):
    epoch = 10
    pop_size = 50
    p_m = 0.01
    elites = 2
    model = BBO.BaseBBO(problem, epoch, pop_size, p_m, elites)
    best_position, best_fitness = model.solve()
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
def test_epoch_BBO(problem, epoch, system_code):
    pop_size = 50
    p_m = 0.01
    elites = 2
    algorithms = [BBO.OriginalBBO, BBO.BaseBBO]
    for algorithm in algorithms:
        with pytest.raises(SystemExit) as e:
            model = algorithm(problem, epoch, pop_size, p_m, elites)
        assert e.type == SystemExit
        assert e.value.code == system_code


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
def test_pop_size_BBO(problem, pop_size, system_code):
    epoch = 10
    p_m = 0.01
    elites = 2
    algorithms = [BBO.OriginalBBO, BBO.BaseBBO]
    for algorithm in algorithms:
        with pytest.raises(SystemExit) as e:
            model = algorithm(problem, epoch, pop_size, p_m, elites)
        assert e.type == SystemExit
        assert e.value.code == system_code


@pytest.mark.parametrize("problem, p_m, system_code",
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
def test_p_m_BBO(problem, p_m, system_code):
    epoch = 10
    pop_size = 50
    elites = 2
    algorithms = [BBO.OriginalBBO, BBO.BaseBBO]
    for algorithm in algorithms:
        with pytest.raises(SystemExit) as e:
            model = algorithm(problem, epoch, pop_size, p_m, elites)
        assert e.type == SystemExit
        assert e.value.code == system_code


@pytest.mark.parametrize("problem, elites, system_code",
                         [
                             (problem, None, 0),
                             (problem, "hello", 0),
                             (problem, -1.0, 0),
                             (problem, [10], 0),
                             (problem, (0, 9), 0),
                             (problem, 1, 0),
                             (problem, 50, 0),
                             (problem, 100, 0),
                             (problem, 1.6, 0),
                         ])
def test_elites_BBO(problem, elites, system_code):
    epoch = 10
    pop_size = 50
    p_m = 0.01
    algorithms = [BBO.OriginalBBO, BBO.BaseBBO]
    for algorithm in algorithms:
        with pytest.raises(SystemExit) as e:
            model = algorithm(problem, epoch, pop_size, p_m, elites)
        assert e.type == SystemExit
        assert e.value.code == system_code
