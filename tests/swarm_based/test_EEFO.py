import numpy as np
import pytest
from mealpy import FloatVar, Optimizer
from mealpy.swarm_based.EEFO import EEFO

@pytest.fixture(scope="module")
def problem():
    def objective_function(solution):
        return np.sum(solution ** 2)

    problem = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-10, -15, -4, -2, -8], ub=[10, 15, 12, 8, 20]),
        "minmax": "min",
        "log_to": None
    }
    return problem

def test_EEFO_results(problem):
    models = [
        EEFO(epoch=10, pop_size=20)
    ]
    for model in models:
        g_best = model.solve(problem)

        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)
        assert isinstance(g_best.target.fitness, (float, np.floating))