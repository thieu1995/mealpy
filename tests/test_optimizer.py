#!/usr/bin/env python
# Created by "Thieu" at 00:15, 15/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.optimizer import Optimizer
import numpy as np
import pytest


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def model():
    def fitness_function(solution):
        return np.sum(solution ** 2)

    problem = {
        "fit_func": fitness_function,
        "lb": [-10, -10, -10, -10, -10],
        "ub": [10, 10, 10, 10, 10],
        "minmax": "min",
        "log_to": None,
    }
    model = Optimizer(problem)
    return model


def test_amend_position(model):
    LB = model.problem.lb
    UB = model.problem.ub
    pos = np.array([-15.4, 100, 0.4, -9, 7])
    pos = model.amend_position(pos, LB, UB)
    comparison = (pos == np.array([-10, 10, 0.4, -9, 7]))
    assert comparison.all()


def test_get_target_wrapper(model):
    pos = np.array([1, 2, 0, 2, 1])
    target = model.get_target_wrapper(pos)
    fit = np.sum(pos ** 2)
    assert target[model.ID_FIT] == fit


def test_create_solution(model):
    position, target = model.create_solution(model.problem.lb, model.problem.ub)
    fitness, objs = target
    assert len(position) == model.problem.n_dims
    assert len(objs) == model.problem.n_objs
    assert isinstance(position, np.ndarray)
    assert isinstance(target, list)
    assert isinstance(objs, list)


def test_create_population(model):
    n_agent = 10
    pop = model.create_population(pop_size=n_agent)
    idx_rand = np.random.choice(range(0, n_agent))
    agent = pop[idx_rand]
    assert type(pop) == list
    assert len(pop) == n_agent
    assert len(agent) == 2
    assert len(agent[model.ID_POS]) == model.problem.n_dims
    assert len(agent[model.ID_TAR]) == 2
    assert len(agent[model.ID_TAR][model.ID_OBJ]) == model.problem.n_objs


def test_update_target_wrapper_population(model):
    pop = [
        [np.array([1, 0, 2, 0, 1]), None],
        [np.array([0, 0, 3, 4, 5]), None],
        [np.array([-1, -2, 3, 4, 0]), None],
        [np.array([5, -4, 0, 1, -1]), None]
    ]
    list_targets = [model.get_target_wrapper(agent[model.ID_POS]) for agent in pop]
    pop = model.update_target_wrapper_population(pop)
    for idx, agent in enumerate(pop):
        assert agent[model.ID_TAR] is not None
        assert agent[model.ID_TAR][model.ID_FIT] == list_targets[idx][model.ID_FIT]


def test_get_better_solution(model):
    pos_a = np.array([1, -1, 0, 3, 2])
    pos_b = np.array([0, 0, 0, 0, 1])
    tar_a = model.get_target_wrapper(pos_a)
    tar_b = model.get_target_wrapper(pos_b)
    agent_a = [pos_a, tar_a]
    agent_b = [pos_b, tar_b]
    minmax = model.problem.minmax
    better = model.get_better_solution(agent_a, agent_b)
    if minmax == "min":
        assert tar_b[model.ID_FIT] == better[model.ID_TAR][model.ID_FIT]
    else:
        assert tar_a[model.ID_FIT] == better[model.ID_TAR][model.ID_FIT]


def test_compare_agent(model):
    pos_a = np.array([1, -1, 0, 3, 2])
    pos_b = np.array([0, 0, 0, 0, 1])
    tar_a = model.get_target_wrapper(pos_a)
    tar_b = model.get_target_wrapper(pos_b)
    agent_a = [pos_a, tar_a]
    agent_b = [pos_b, tar_b]
    minmax = model.problem.minmax
    if minmax == "min":
        flag = model.compare_agent(agent_a, agent_b)
        assert flag is False
    else:
        flag = model.compare_agent(agent_a, agent_b)
        assert flag is True


def test_get_special_solutions(model):
    pop = [
        [np.array([1, 2, 3, 4, 5]), [55, [55]]],
        [np.array([0, 1, 2, 3, 4]), [30, [30]]],
        [np.array([0, 0, 1, 2, 3]), [14, [14]]],
        [np.array([0, 0, 0, 1, 2]), [5, [5]]],
        [np.array([0, 0, 0, 0, 2]), [4, [4]]],
        [np.array([0, 0, 0, 0, 1]), [1, [1]]]
    ]
    minmax = model.problem.minmax
    if minmax == "min":
        pop_sorted, k_best, k_worst = model.get_special_solutions(pop, best=2, worst=1)
        assert len(pop_sorted) == len(pop)
        assert len(k_best) == 2
        assert len(k_worst) == 1
        best_1, best_2 = k_best
        assert best_1[model.ID_TAR][model.ID_FIT] == 1
        assert best_2[model.ID_TAR][model.ID_FIT] == 4
        worst = k_worst[0]
        assert worst[model.ID_TAR][model.ID_FIT] == 55

        _, _, k_worst = model.get_special_solutions(pop, worst=3)
        assert len(k_worst) == 3
        assert k_worst[0][model.ID_TAR][model.ID_FIT] == 55
        assert k_worst[-1][model.ID_TAR][model.ID_FIT] == 14
    else:
        _, k_best, k_worst = model.get_special_solutions(pop, best=3, worst=4)
        least_best = k_best[-1]
        least_worst = k_worst[-1]
        assert least_worst[model.ID_TAR][model.ID_FIT] == least_best[model.ID_TAR][model.ID_FIT]
        assert k_best[0][model.ID_TAR][model.ID_FIT] == 55
        assert k_worst[0][model.ID_TAR][model.ID_FIT] == 1


def test_get_special_fitness(model):
    pop = [
        [np.array([1, 2, 3, 4, 5]), [55, [55]]],
        [np.array([0, 1, 2, 3, 4]), [30, [30]]],
        [np.array([0, 0, 1, 2, 3]), [14, [14]]],
        [np.array([0, 0, 0, 1, 2]), [5, [5]]],
        [np.array([0, 0, 0, 0, 2]), [4, [4]]],
        [np.array([0, 0, 0, 0, 1]), [1, [1]]]
    ]
    fit_sum = 55 + 30 + 14 + 5 + 4 + 1
    minmax = model.problem.minmax
    if minmax == "min":
        total_fit, best_fit, worst_fit = model.get_special_fitness(pop)
        assert total_fit == fit_sum
        assert best_fit == 1
        assert worst_fit == 55
    else:
        total_fit, best_fit, worst_fit = model.get_special_fitness(pop)
        assert total_fit == fit_sum
        assert best_fit == 55
        assert worst_fit == 1


def test_greedy_selection_population(model):
    pop_old = [
        [np.array([1, 2, 3, 4, 5]), [55, [55]]],
        [np.array([0, 1, 2, 3, 4]), [30, [30]]],
        [np.array([0, 0, 1, 2, 3]), [14, [14]]],
        [np.array([0, 0, 0, 1, 2]), [5, [5]]],
    ]
    pop_new = [
        [np.array([1, 2, 3, 4, 4]), [46, [46]]],
        [np.array([0, 1, 2, 3, 3]), [23, [23]]],
        [np.array([0, 0, 1, 2, 4]), [21, [21]]],
        [np.array([0, 0, 0, 1, 3]), [10, [10]]],
    ]
    pop_child = model.greedy_selection_population(pop_old, pop_new)
    list_better_fits_min = [46, 23, 14, 5]
    list_better_fits_max = [55, 30, 21, 10]

    if model.problem.minmax == "min":
        for idx, agent in enumerate(pop_child):
            assert agent[model.ID_TAR][model.ID_FIT] == list_better_fits_min[idx]
    else:
        for idx, agent in enumerate(pop_child):
            assert agent[model.ID_TAR][model.ID_FIT] == list_better_fits_max[idx]


def test_get_sorted_strim_population(model):
    pop = [
        [np.array([1, 2, 3, 4, 5]), [55, [55]]],
        [np.array([0, 0, 0, 0, 1]), [1, [1]]],
        [np.array([0, 0, 1, 2, 3]), [14, [14]]],
        [np.array([0, 0, 0, 1, 2]), [5, [5]]],
        [np.array([0, 1, 2, 3, 4]), [30, [30]]],
        [np.array([0, 0, 0, 0, 2]), [4, [4]]],
    ]
    pop_size = 4
    pop_new = model.get_sorted_strim_population(pop, pop_size)
    assert len(pop_new) == pop_size
    if model.problem.minmax == "min":
        assert pop_new[0][model.ID_TAR][model.ID_FIT] == 1
        assert pop_new[-1][model.ID_TAR][model.ID_FIT] == 14
    else:
        assert pop_new[0][model.ID_TAR][model.ID_FIT] == 55
        assert pop_new[-1][model.ID_TAR][model.ID_FIT] == 5


def test_create_opposition_position(model):
    pop = [
        [np.array([1, 2, 3, 4, 5]), [55, [55]]],
        [np.array([0, 0, 0, 0, 1]), [1, [1]]],
        [np.array([0, 0, 1, 2, 3]), [14, [14]]],
        [np.array([0, 0, 0, 1, 2]), [5, [5]]],
        [np.array([0, 1, 2, 3, 4]), [30, [30]]],
        [np.array([0, 0, 0, 0, 2]), [4, [4]]],
    ]
    g_best = [np.array([0, 0, 0, 0, 1]) , [1, [1]]]
    pos_opposite = model.create_opposition_position(pop[0], g_best)
    assert isinstance(pos_opposite, np.ndarray)
    assert len(pos_opposite) == model.problem.n_dims


def test_crossover_arithmetic(model):
    pop = [
        [np.array([1, 2, 3, 4, 5]), [55, [55]]],
        [np.array([0, 0, 0, 0, 1]), [1, [1]]],
        [np.array([0, 0, 1, 2, 3]), [14, [14]]],
        [np.array([0, 0, 0, 1, 2]), [5, [5]]],
        [np.array([0, 1, 2, 3, 4]), [30, [30]]],
        [np.array([0, 0, 0, 0, 2]), [4, [4]]],
    ]
    idx_dad, idx_mom = np.random.choice(range(0, len(pop)), 2, replace=False)
    pos_dad, pos_mom = pop[idx_dad][model.ID_POS], pop[idx_mom][model.ID_POS]
    pos_child1, pos_child2 = model.crossover_arithmetic(pos_dad, pos_mom)

    assert len(pos_child1) == len(pos_child2)
    assert isinstance(pos_child1, np.ndarray)
    assert len(pos_child2) == model.problem.n_dims
