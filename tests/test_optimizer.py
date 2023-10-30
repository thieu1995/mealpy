#!/usr/bin/env python
# Created by "Thieu" at 00:15, 15/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import Problem, Optimizer, FloatVar
import numpy as np
import pytest


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def model():
    def objective_function(solution):
        return np.sum(solution ** 2)

    problem = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-10, -15, -4, -2, -8], ub=[10, 15, 12, 8, 20]),
        "minmax": "min",
        "log_to": None,
    }
    model = Optimizer()
    model.problem = Problem(**problem)
    return model


def test_generate_agent(model):
    pos = np.array([-1, 0, 2, 1, 0, 3])
    agent = model.generate_agent(pos)
    assert np.all(agent.solution == pos)


def test_correct_solution(model):
    pos = np.array([-15.4, 100, 0.4, -9, 7])
    pos = model.correct_solution(pos)
    out = np.clip(pos, model.problem.lb, model.problem.ub)
    assert np.all(pos == out)


def test_get_target(model):
    pos = np.array([1, 2, 0, 2, 1])
    target = model.get_target(pos)
    fit = np.sum(pos ** 2)
    assert target.fitness == fit


def test_create_solution(model):
    pos = np.array([1, 2, 0, 2, 1])
    agent = model.generate_agent()
    assert len(agent.solution) == model.problem.n_dims
    assert len(agent.target.objectives) == model.problem.n_objs
    assert isinstance(pos, np.ndarray)


def test_generate_population(model):
    n_agent = 10
    pop = model.generate_population(pop_size=n_agent)
    idx_rand = np.random.choice(range(0, n_agent))
    agent = pop[idx_rand]
    assert type(pop) is list
    assert len(pop) == n_agent
    assert len(agent.solution) == model.problem.n_dims


def test_update_target_for_population(model):
    pop = model.generate_population(5)
    list_targets = [model.get_target(agent.solution) for agent in pop]
    model.mode = "thread"
    pop = model.update_target_for_population(pop)
    for idx, agent in enumerate(pop):
        assert agent.target is not None
        assert agent.target.fitness == list_targets[idx].fitness


def test_get_better_solution(model):
    pos_a = np.array([1, -1, 0, 3, 2])
    pos_b = np.array([0, 0, 0, 0, 1])
    agent_a = model.generate_agent(pos_a)
    agent_b = model.generate_agent(pos_b)
    minmax = model.problem.minmax
    better = model.get_better_agent(agent_a, agent_b, model.problem.minmax)
    if minmax == "min":
        assert agent_b.target.fitness == better.target.fitness
    else:
        assert agent_a.target.fitness == better.target.fitness


def test_compare_agent(model):
    pos_a = np.array([1, -1, 0, 3, 2])
    pos_b = np.array([0, 0, 0, 0, 1])
    agent_a = model.generate_agent(pos_a)
    agent_b = model.generate_agent(pos_b)
    if model.problem.minmax == "min":
        flag = model.compare_target(agent_a.target, agent_b.target, model.problem.minmax)
        assert flag is False
    else:
        flag = model.compare_target(agent_a.target, agent_b.target, model.problem.minmax)
        assert flag is True
