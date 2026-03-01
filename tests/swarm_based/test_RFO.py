import numpy as np
from mealpy import FloatVar
from mealpy.swarm_based import OriginalRFO


def objective_function(solution):
    return np.sum(solution ** 2)


def test_OriginalRFO():
    problem = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=(-10.,) * 5, ub=(10.,) * 5),
        "minmax": "min",
    }

    model = OriginalRFO(epoch=50, pop_size=20)
    result = model.solve(problem)

    assert result.solution is not None
    assert result.target.fitness is not None
