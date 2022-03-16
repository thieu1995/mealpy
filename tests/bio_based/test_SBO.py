#!/usr/bin/env python
# Created by "Thieu" at 21:05, 16/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.bio_based import SBO
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


def test_OriginalSBO_results(problem):
    epoch = 10
    pop_size = 50
    alpha = 0.94
    p_m = 0.05
    psw = 0.02
    model = SBO.OriginalSBO(problem, epoch, pop_size, alpha, p_m, psw)
    best_position, best_fitness = model.solve()
    assert isinstance(model, Optimizer)
    assert isinstance(best_position, np.ndarray)
    assert len(best_position) == len(problem["lb"])


def test_BaseSBO_results(problem):
    epoch = 10
    pop_size = 50
    alpha = 0.94
    p_m = 0.05
    psw = 0.02
    model = SBO.BaseSBO(problem, epoch, pop_size, alpha, p_m, psw)
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
def test_epoch_SBO(problem, epoch, system_code):
    pop_size = 50
    algorithms = [SBO.OriginalSBO, SBO.BaseSBO]
    for algorithm in algorithms:
        with pytest.raises(SystemExit) as e:
            model = algorithm(problem, epoch, pop_size)
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
def test_pop_size_SBO(problem, pop_size, system_code):
    epoch = 10
    algorithms = [SBO.OriginalSBO, SBO.BaseSBO]
    for algorithm in algorithms:
        with pytest.raises(SystemExit) as e:
            model = algorithm(problem, epoch, pop_size)
        assert e.type == SystemExit
        assert e.value.code == system_code


@pytest.mark.parametrize("problem, alpha, system_code",
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
def test_alpha_SBO(problem, alpha, system_code):
    epoch = 10
    pop_size = 50
    algorithms = [SBO.OriginalSBO, SBO.BaseSBO]
    for algorithm in algorithms:
        with pytest.raises(SystemExit) as e:
            model = algorithm(problem, epoch, pop_size, alpha=alpha)
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
def test_p_m_SBO(problem, p_m, system_code):
    epoch = 10
    pop_size = 50
    algorithms = [SBO.OriginalSBO, SBO.BaseSBO]
    for algorithm in algorithms:
        with pytest.raises(SystemExit) as e:
            model = algorithm(problem, epoch, pop_size, p_m=p_m)
        assert e.type == SystemExit
        assert e.value.code == system_code


@pytest.mark.parametrize("problem, psw, system_code",
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
def test_p_m_SBO(problem, psw, system_code):
    epoch = 10
    pop_size = 50
    algorithms = [SBO.OriginalSBO, SBO.BaseSBO]
    for algorithm in algorithms:
        with pytest.raises(SystemExit) as e:
            model = algorithm(problem, epoch, pop_size, psw=psw)
        assert e.type == SystemExit
        assert e.value.code == system_code
