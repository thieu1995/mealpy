import numpy as np
from mealpy import FloatVar
from mealpy.swarm_based.RFO import OriginalRFO


def sphere(x):
    return np.sum(x ** 2)


def test_rfo_runs():
    problem = {
        "obj_func": sphere,
        "bounds": FloatVar(lb=(-5.,) * 5, ub=(5.,) * 5),
        "minmax": "min",
    }

    model = OriginalRFO(epoch=10, pop_size=10)
    result = model.solve(problem)

    assert result.solution is not None

