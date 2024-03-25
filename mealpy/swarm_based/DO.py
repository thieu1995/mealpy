#!/usr/bin/env python
# Created by "Thieu" at 04:43, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalDO(Optimizer):
    """
    The original version of: Dragonfly Optimization (DO)

    Links:
        1. https://link.springer.com/article/10.1007/s00521-015-1920-1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, DO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = DO.OriginalDO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2016. Dragonfly algorithm: a new meta-heuristic optimization technique for solving single-objective,
    discrete, and multi-objective problems. Neural computing and applications, 27(4), pp.1053-1073.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        self.pop_delta = self.generate_population(self.pop_size)
        # Initial radius of dragonflies' neighborhoods
        self.radius = (self.problem.ub - self.problem.lb) / 10
        self.delta_max = (self.problem.ub - self.problem.lb) / 10

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        _, (self.g_best, ), (self.g_worst, ) = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)

        r = (self.problem.ub - self.problem.lb) / 4 + ((self.problem.ub - self.problem.lb) * (2 * epoch / self.epoch))
        w = 0.9 - epoch * ((0.9 - 0.4) / self.epoch)
        my_c = 0.1 - epoch * ((0.1 - 0) / (self.epoch / 2))
        my_c = 0 if my_c < 0 else my_c

        s = 2 * self.generator.random() * my_c  # Seperation weight
        a = 2 * self.generator.random() * my_c  # Alignment weight
        c = 2 * self.generator.random() * my_c  # Cohesion weight
        f = 2 * self.generator.random()  # Food attraction weight
        e = my_c  # Enemy distraction weight

        pop_new = []
        pop_delta_new = []
        for idx in range(0, self.pop_size):
            pos_neighbours = []
            pos_neighbours_delta = []
            neighbours_num = 0
            # Find the neighbouring solutions
            for j in range(0, self.pop_size):
                dist = np.abs(self.pop[idx].solution - self.pop[j].solution)
                if np.all(dist <= r) and np.all(dist != 0):
                    neighbours_num += 1
                    pos_neighbours.append(self.pop[j].solution)
                    pos_neighbours_delta.append(self.pop_delta[j].solution)
            pos_neighbours = np.array(pos_neighbours)
            pos_neighbours_delta = np.array(pos_neighbours_delta)

            # Separation: Eq 3.1, Alignment: Eq 3.2, Cohesion: Eq 3.3
            if neighbours_num > 1:
                S = np.sum(pos_neighbours, axis=0) - neighbours_num * self.pop[idx].solution
                A = np.sum(pos_neighbours_delta, axis=0) / neighbours_num
                C_temp = np.sum(pos_neighbours, axis=0) / neighbours_num
            else:
                S = np.zeros(self.problem.n_dims)
                A = self.pop_delta[idx].solution.copy()
                C_temp = self.pop[idx].solution.copy()
            C = C_temp - self.pop[idx].solution

            # Attraction to food: Eq 3.4
            dist_to_food = np.abs(self.pop[idx].solution - self.g_best.solution)
            if np.all(dist_to_food <= r):
                F = self.g_best.solution - self.pop[idx].solution
            else:
                F = np.zeros(self.problem.n_dims)

            # Distraction from enemy: Eq 3.5
            dist_to_enemy = np.abs(self.pop[idx].solution - self.g_worst.solution)
            if np.all(dist_to_enemy <= r):
                enemy = self.g_worst.solution + self.pop[idx].solution
            else:
                enemy = np.zeros(self.problem.n_dims)

            pos_new = self.pop[idx].solution.copy().astype(float)
            pos_delta_new = self.pop_delta[idx].solution.copy().astype(float)
            if np.any(dist_to_food > r):
                if neighbours_num > 1:
                    temp = w * self.pop_delta[idx].solution + self.generator.uniform(0, 1, self.problem.n_dims) * A + \
                           self.generator.uniform(0, 1, self.problem.n_dims) * C + self.generator.uniform(0, 1, self.problem.n_dims) * S
                    temp = np.clip(temp, -1 * self.delta_max, self.delta_max)
                    pos_delta_new = temp.copy()
                    pos_new += temp
                else:  # Eq. 3.8
                    pos_new += self.get_levy_flight_step(beta=1.5, multiplier=0.01, case=-1) * self.pop[idx].solution
                    pos_delta_new = np.zeros(self.problem.n_dims)
            else:
                # Eq. 3.6
                temp = (a * A + c * C + s * S + f * F + e * enemy) + w * self.pop_delta[idx].solution
                temp = np.clip(temp, -1 * self.delta_max, self.delta_max)
                pos_delta_new = temp
                pos_new += temp

            # Amend solution
            pos_new = self.correct_solution(pos_new)
            pos_delta_new = self.correct_solution(pos_delta_new)
            agent = self.generate_empty_agent(pos_new)
            agent_delta = self.generate_empty_agent(pos_delta_new)
            pop_new.append(agent)
            pop_delta_new.append(agent_delta)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                agent_delta.target = self.get_target(pos_delta_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
                self.pop_delta[idx] = self.get_better_agent(agent_delta, self.pop_delta[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            pop_delta_new = self.update_target_for_population(pop_delta_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
            self.pop_delta = self.greedy_selection_population(self.pop_delta, pop_delta_new, self.problem.minmax)
