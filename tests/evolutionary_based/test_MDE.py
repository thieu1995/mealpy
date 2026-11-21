#!/usr/bin/env python
# Created by "A. E. Karkinli" on 21/11/2026
# Github: https://github.com/Aekarkinli
# ------------------------------------------%

from mealpy import FloatVar, MDE, Optimizer, get_optimizer_by_name
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
        "log_to": None,
    }
    return problem


def test_MDE_results(problem):
    models = [
        MDE.OriginalMDE(epoch=20, pop_size=30),
        MDE.OriginalMDE(epoch=20, pop_size=10),
    ]
    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)
        assert len(model.pop) == model.pop_size
        assert np.all(g_best.solution >= model.problem.lb) and np.all(g_best.solution <= model.problem.ub)


def test_MDE_function_evaluations(problem):
    model = MDE.OriginalMDE(epoch=10, pop_size=30)
    model.solve(problem, termination={"max_fe": 10000})
    # pop_size evaluations for the initial population + (pop_size // 3) per epoch
    assert model.nfe_counter == 30 + 10 * 10


@pytest.mark.parametrize("mode", ["single", "swarm", "thread"])
def test_MDE_modes(problem, mode):
    model = MDE.OriginalMDE(epoch=10, pop_size=30)
    g_best = model.solve(problem, mode=mode, n_workers=2)
    assert len(g_best.solution) == len(model.problem.lb)


def test_MDE_seed_reproducibility(problem):
    g1 = MDE.OriginalMDE(epoch=30, pop_size=30).solve(problem, seed=42)
    g2 = MDE.OriginalMDE(epoch=30, pop_size=30).solve(problem, seed=42)
    assert np.array_equal(g1.solution, g2.solution)
    assert g1.target.fitness == g2.target.fitness


def test_MDE_maximization():
    problem = {
        "obj_func": lambda x: -np.sum((x - 1.0) ** 2),
        "bounds": FloatVar(lb=[-5.] * 5, ub=[5.] * 5),
        "minmax": "max",
        "log_to": None,
    }
    g_best = MDE.OriginalMDE(epoch=200, pop_size=30).solve(problem, seed=1)
    assert g_best.target.fitness > -1e-3


def test_MDE_registered():
    assert "OriginalMDE" in get_optimizer_by_name("MDE")


@pytest.mark.parametrize("pop_size", [5, 8])
def test_MDE_invalid_pop_size(pop_size):
    with pytest.raises(ValueError):
        MDE.OriginalMDE(epoch=10, pop_size=pop_size)
