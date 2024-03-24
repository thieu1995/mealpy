#!/usr/bin/env python
# Created by "Thieu" at 11:16, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class DevSARO(Optimizer):
    """
    The developed version: Search And Rescue Optimization (SARO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + se (float): [0.3, 0.8], social effect, default = 0.5
        + mu (int): maximum unsuccessful search number, belongs to range: [2, 2+int(self.pop_size/2)], default = 15

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SARO
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
    >>> model = SARO.DevSARO(epoch=1000, pop_size=50, se = 0.5, mu = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, se: float = 0.5, mu: int = 15, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            se (float): social effect, default = 0.5
            mu (int): maximum unsuccessful search number, default = 15
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.se = self.validator.check_float("se", se, (0, 1.0))
        self.mu = self.validator.check_int("mu", mu, [2, 2+int(self.pop_size/2)])
        self.set_parameters(["epoch", "pop_size", "se", "mu"])
        self.sort_flag = True

    def initialize_variables(self):
        self.dyn_USN = np.zeros(self.pop_size)

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(2 * self.pop_size)
        else:
            self.pop = self.pop + self.generate_population(self.pop_size)

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        rand_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, rand_pos)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_x = [agent.copy() for agent in self.pop[:self.pop_size]]
        pop_m = [agent.copy() for agent in self.pop[self.pop_size:]]
        pop_new = []
        for idx in range(self.pop_size):
            ## Social Phase
            k = self.generator.choice(list(set(range(0, 2 * self.pop_size)) - {idx}))
            sd = pop_x[idx].solution - self.pop[k].solution
            #### Remove third loop here, also using random flight back when out of bound
            pos_new_1 = self.pop[k].solution + self.generator.uniform() * sd
            pos_new_2 = pop_x[idx].solution + self.generator.uniform() * sd
            condition = np.logical_and(self.generator.uniform(0, 1, self.problem.n_dims) < self.se, self.pop[k].target.fitness < pop_x[idx].target.fitness)
            pos_new = np.where(condition, pos_new_1, pos_new_2)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx in range(self.pop_size):
            if self.compare_target(pop_new[idx].target, pop_x[idx].target, self.problem.minmax):
                pop_m[self.generator.integers(0, self.pop_size)] = pop_x[idx].copy()
                pop_x[idx] = pop_new[idx].copy()
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1
        pop = pop_x.copy() + pop_m.copy()
        pop_new = []
        for idx in range(self.pop_size):
            ## Individual phase
            k1, k2 = self.generator.choice(list(set(range(0, 2 * self.pop_size)) - {idx}), 2, replace=False)
            #### Remove third loop here, and flight back strategy now be a random
            pos_new = self.g_best.solution + self.generator.uniform() * (pop[k1].solution - pop[k2].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_target(pop_new[idx].target, pop_x[idx].target, self.problem.minmax):
                pop_m[self.generator.integers(0, self.pop_size)] = pop_x[idx].copy()
                pop_x[idx] = pop_new[idx].copy()
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1
            if self.dyn_USN[idx] > self.mu:
                pop_x[idx] = self.generate_agent()
                self.dyn_USN[idx] = 0
        self.pop = pop_x + pop_m


class OriginalSARO(DevSARO):
    """
    The original version of: Search And Rescue Optimization (SARO)

    Links:
       1. https://doi.org/10.1155/2019/2482543

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + se (float): [0.3, 0.8], social effect, default = 0.5
        + mu (int): [10, 20], maximum unsuccessful search number, default = 15

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SARO
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
    >>> model = SARO.OriginalSARO(epoch=1000, pop_size=50, se = 0.5, mu = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Shabani, A., Asgarian, B., Gharebaghi, S.A., Salido, M.A. and Giret, A., 2019. A new optimization
    algorithm based on search and rescue operations. Mathematical Problems in Engineering, 2019.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, se: float = 0.5, mu: int = 15, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            se (float): social effect, default = 0.5
            mu (int): maximum unsuccessful search number, default = 15
        """
        super().__init__(epoch, pop_size, se, mu, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_x = [agent.copy() for agent in self.pop[:self.pop_size]]
        pop_m = [agent.copy() for agent in self.pop[self.pop_size:]]
        pop_new = []
        for idx in range(self.pop_size):
            ## Social Phase
            k = self.generator.choice(list(set(range(0, 2 * self.pop_size)) - {idx}))
            sd = pop_x[idx].solution - self.pop[k].solution
            j_rand = self.generator.integers(0, self.problem.n_dims)
            r1 = self.generator.uniform(-1, 1)

            pos_new = pop_x[idx].solution.copy()
            for j in range(0, self.problem.n_dims):
                if self.generator.uniform() < self.se or j == j_rand:
                    if self.compare_target(self.pop[k].target, pop_x[idx].target, self.problem.minmax):
                        pos_new[j] = self.pop[k].solution[j] + r1 * sd[j]
                    else:
                        pos_new[j] = pop_x[idx].solution[j] + r1 * sd[j]
                if pos_new[j] < self.problem.lb[j]:
                    pos_new[j] = (pop_x[idx].solution[j] + self.problem.lb[j]) / 2
                if pos_new[j] > self.problem.ub[j]:
                    pos_new[j] = (pop_x[idx].solution[j] + self.problem.ub[j]) / 2
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_target(pop_new[idx].target, pop_x[idx].target, self.problem.minmax):
                pop_m[self.generator.integers(0, self.pop_size)] = pop_x[idx].copy()
                pop_x[idx] = pop_new[idx].copy()
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

        ## Individual phase
        pop = pop_x.copy() + pop_m.copy()
        pop_new = []
        for idx in range(0, self.pop_size):
            k, m = self.generator.choice(list(set(range(0, 2 * self.pop_size)) - {idx}), 2, replace=False)
            pos_new = pop_x[idx].solution + self.generator.uniform() * (pop[k].solution - pop[m].solution)
            for j in range(0, self.problem.n_dims):
                if pos_new[j] < self.problem.lb[j]:
                    pos_new[j] = (pop_x[idx].solution[j] + self.problem.lb[j]) / 2
                if pos_new[j] > self.problem.ub[j]:
                    pos_new[j] = (pop_x[idx].solution[j] + self.problem.ub[j]) / 2
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_target(pop_new[idx].target, pop_x[idx].target, self.problem.minmax):
                pop_m[self.generator.integers(0, self.pop_size)] = pop_x[idx]
                pop_x[idx] = pop_new[idx].copy()
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

            if self.dyn_USN[idx] > self.mu:
                pop_x[idx] = self.generate_agent()
                self.dyn_USN[idx] = 0
        self.pop = pop_x + pop_m
