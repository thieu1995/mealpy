#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalAGTO(Optimizer):
    """
    The original version of: Artificial Gorilla Troops Optimization (AGTO)

    Links:
        1. https://doi.org/10.1002/int.22535
        2. https://www.mathworks.com/matlabcentral/fileexchange/95953-artificial-gorilla-troops-optimizer

    Notes (parameters):
        1. p1 (float): the probability of transition in exploration phase (p in the paper), default = 0.03
        2. p2 (float): the probability of transition in exploitation phase (w in the paper), default = 0.8
        3. beta (float): coefficient in updating equation, should be in [-5.0, 5.0], default = 3.0

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, AGTO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = AGTO.OriginalAGTO(epoch=1000, pop_size=50, p1=0.03, p2=0.8, beta=3.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdollahzadeh, B., Soleimanian Gharehchopogh, F., & Mirjalili, S. (2021). Artificial gorilla troops optimizer: a new
    nature‐inspired metaheuristic algorithm for global optimization problems. International Journal of Intelligent Systems, 36(10), 5887-5958.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, p1: float = 0.03, p2: float = 0.8, beta: float = 3.0, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p1 = self.validator.check_float("p1", p1, (0, 1))      # p in the paper
        self.p2 = self.validator.check_float("p2", p2, (0, 1))      # w in the paper
        self.beta = self.validator.check_float("beta", beta, [-10.0, 10.0])
        self.set_parameters(["epoch", "pop_size", "p1", "p2", "beta"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        a = (np.cos(2*self.generator.random())+1) * (1 - epoch/self.epoch)
        c = a * (2 * self.generator.random() - 1)
        ## Exploration
        pop_new = []
        for idx in range(0, self.pop_size):
            if self.generator.random() < self.p1:
                pos_new = self.problem.generate_solution()
            else:
                if self.generator.random() >= 0.5:
                    z = self.generator.uniform(-a, a, self.problem.n_dims)
                    rand_idx = self.generator.integers(0, self.pop_size)
                    pos_new = (self.generator.random() - a) * self.pop[rand_idx].solution + c * z * self.pop[idx].solution
                else:
                    id1, id2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                    pos_new = self.pop[idx].solution - c*(c*self.pop[idx].solution - self.pop[id1].solution) + \
                        self.generator.random() * (self.pop[idx].solution - self.pop[id2].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        _, self.g_best = self.update_global_best_agent(self.pop, save=False)

        pos_list = np.array([agent.solution for agent in self.pop])
        ## Exploitation
        pop_new = []
        for idx in range(0, self.pop_size):
            if a >= self.p2:
                g = 2 ** c
                delta = (np.abs(np.mean(pos_list, axis=0)) ** g) ** (1.0 / g)
                pos_new = c*delta*(self.pop[idx].solution - self.g_best.solution) + self.pop[idx].solution
            else:
                if self.generator.random() >= 0.5:
                    h = self.generator.normal(0, 1, self.problem.n_dims)
                else:
                    h = self.generator.normal(0, 1)
                r1 = self.generator.random()
                pos_new = self.g_best.solution - (2*r1-1)*(self.g_best.solution - self.pop[idx].solution) * (self.beta * h)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class MGTO(Optimizer):
    """
    The original version of: Modified Gorilla Troops Optimization (mGTO)

    Notes (parameters):
        1. pp (float): the probability of transition in exploration phase (p in the paper), default = 0.03

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, AGTO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = AGTO.MGTO(epoch=1000, pop_size=50, pp=0.03)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mostafa, R. R., Gaheen, M. A., Abd ElAziz, M., Al-Betar, M. A., & Ewees, A. A. (2023). An improved gorilla
    troops optimizer for global optimization problems and feature selection. Knowledge-Based Systems, 110462.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, pp: float = 0.03, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pp (float): the probability of transition in exploration phase (p in the paper), default = 0.03
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pp = self.validator.check_float("p1", pp, (0, 1))      # p in the paper
        self.set_parameters(["epoch", "pop_size", "pp"])
        self.sort_flag = False

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        random_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, random_pos)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        F = 1 + np.cos(2 * self.generator.random())
        C = F * (1 - epoch / self.epoch)
        L = C * self.generator.choice([-1, 1])

        ## Elite opposition-based learning
        pos_list = np.array([agent.solution for agent in self.pop])
        d_lb, d_ub = np.min(pos_list, axis=0), np.max(pos_list, axis=0)
        pos_list = d_lb + d_ub - pos_list
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.correct_solution(pos_list[idx])
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        self.pop = pop_new
        _, self.g_best = self.update_global_best_agent(self.pop, save=False)

        ## Exploration
        pop_new = []
        for idx in range(0, self.pop_size):
            if self.generator.random() < self.pp:
                pos_new = self.problem.generate_solution()
            else:
                if self.generator.random() >= 0.5:
                    rand_idx = self.generator.integers(0, self.pop_size)
                    pos_new = (self.generator.random() - C) * self.pop[rand_idx].solution + L * self.generator.uniform(-C, C) * self.pop[idx].solution
                else:
                    id1, id2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                    pos_new = self.pop[idx].solution - L*(L*self.pop[idx].solution - self.pop[id1].solution) + \
                        self.generator.random() * (self.pop[idx].solution - self.pop[id2].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        _, self.g_best = self.update_global_best_agent(self.pop, save=False)

        pos_list = np.array([agent.solution for agent in self.pop])
        ## Exploitation
        pop_new = []
        for idx in range(0, self.pop_size):
            if np.abs(C) >= 1:
                g = self.generator.choice([-0.5, 2.])
                M = (np.abs(np.mean(pos_list, axis=0)) ** g) ** (1.0 / g)
                # print(M)
                p = self.generator.uniform(0, 1, self.problem.n_dims)
                pos_new = L * M * (self.pop[idx].solution - self.g_best.solution) * (0.01 * np.tan(np.pi*( p - 0.5)))
            else:
                Q = 2 * self.generator.random() - 1
                v = self.generator.uniform(0, 1)
                pos_new = self.g_best.solution - Q * (self.g_best.solution - self.pop[idx].solution) * np.tan(v * np.pi/2)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
