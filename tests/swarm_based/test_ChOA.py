import numpy as np
import pytest

from mealpy import FloatVar
from mealpy.swarm_based import ChOA


@pytest.fixture(scope="module")
def problem():
    def objective_function(solution):
        return np.sum(solution ** 2)

    return {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-10, -10, -10], ub=[10, 10, 10]),
        "minmax": "min",
        "log_to": None
    }


def test_original_choa(problem):
    model = ChOA.OriginalChOA(epoch=10, pop_size=20)
    g_best = model.solve(problem)

    assert g_best.target.fitness < 100
    assert isinstance(g_best.solution, np.ndarray)
    assert len(g_best.solution) == 3
