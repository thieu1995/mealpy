#!/usr/bin/env python
# Created by "Thieu" at 15:05, 03/06/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from math import gamma
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalSLO(Optimizer):
    """
    The original version of: Sea Lion Optimization Algorithm (SLO)

    Links:
        1. https://www.researchgate.net/publication/333516932_Sea_Lion_Optimization_Algorithm
        2. https://doi.org/10.14569/IJACSA.2019.0100548

    Notes
    ~~~~~
    + There are some unclear equations and parameters in the original paper 

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.SLO import OriginalSLO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> model = OriginalSLO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Masadeh, R., Mahafzah, B.A. and Sharieh, A., 2019. Sea lion optimization algorithm. Sea, 10(5), p.388.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def amend_position(self, position=None, lb=None, ub=None):
        """
        Args:
            position: vector position (location) of the solution.
            lb: list of lower bound values
            ub: list of upper bound values

        Returns:
            Amended position (make the position is in bound)
        """
        condition = np.logical_and(lb <= position, position <= ub)
        pos_rand = np.random.uniform(lb, ub)
        return np.where(condition, position, pos_rand)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        c = 2 - 2 * epoch / self.epoch
        t0 = np.random.rand()
        v1 = np.sin(2 * np.pi * t0)
        v2 = np.sin(2 * np.pi * (1 - t0))
        SP_leader = np.abs(v1 * (1 + v2) / v2)  # In the paper this is not clear how to calculate

        pop_new = []
        for idx in range(0, self.pop_size):
            if SP_leader < 0.25:
                if c < 1:
                    pos_new = self.g_best[self.ID_POS] - c * np.abs(2 * np.random.rand() *
                                                                    self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                else:
                    ri = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))  # random index
                    pos_new = self.pop[ri][self.ID_POS] - c * np.abs(2 * np.random.rand() *
                                                                     self.pop[ri][self.ID_POS] - self.pop[idx][self.ID_POS])
            else:
                pos_new = np.abs(self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) * \
                          np.cos(2 * np.pi * np.random.uniform(-1, 1)) + self.g_best[self.ID_POS]
            # In the paper doesn't check also doesn't update old solution at this point
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)


class ModifiedSLO(Optimizer):
    """
    The original version of: Modified Sea Lion Optimization (M-SLO)

    Notes
    ~~~~~
    + Local best idea in PSO is inspired 
    + Levy-flight technique is used 
    + Shrink encircling idea is used 

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.SLO import ModifiedSLO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> model = ModifiedSLO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    ID_LOC_POS = 2
    ID_LOC_FIT = 3

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, [target, [obj1, obj2, ...]], local_pos, local_fit]
        """
        ## Increase exploration at the first initial population using opposition-based learning.
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        local_pos = lb + ub - position
        local_pos = self.amend_position(local_pos, lb, ub)
        local_target = self.get_target_wrapper(local_pos)
        if self.compare_agent([None, target], [None, local_target]):
            return [local_pos, local_target, position, target]
        else:
            return [position, target, local_pos, local_target]

    def shrink_encircling_levy__(self, current_pos, epoch, dist, c, beta=1):
        up = gamma(1 + beta) * np.sin(np.pi * beta / 2)
        down = (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2))
        xich_ma_1 = np.power(up / down, 1 / beta)
        xich_ma_2 = 1
        a = np.random.normal(0, xich_ma_1, 1)
        b = np.random.normal(0, xich_ma_2, 1)
        LB = 0.01 * a / (np.power(np.abs(b), 1 / beta)) * dist * c
        D = np.random.uniform(self.problem.lb, self.problem.ub)
        levy = LB * D
        return (current_pos - np.sqrt(epoch + 1) * np.sign(np.random.random(1) - 0.5)) * levy

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """

        c = 2 - 2 * epoch / self.epoch
        if c > 1:
            pa = 0.3  # At the beginning of the process, the probability for shrinking encircling is small
        else:
            pa = 0.7  # But at the end of the process, it become larger. Because sea lion are shrinking encircling prey
        SP_leader = np.random.uniform(0, 1)

        pop_new = []
        for idx in range(0, self.pop_size):
            agent = deepcopy(self.pop[idx])
            if SP_leader >= 0.6:
                pos_new = np.cos(2 * np.pi * np.random.normal(0, 1)) * \
                          np.abs(self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + self.g_best[self.ID_POS]
            else:
                if np.random.uniform() < pa:
                    dist1 = np.random.uniform() * np.abs(2 * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = self.shrink_encircling_levy__(self.pop[idx][self.ID_POS], epoch, dist1, c)
                else:
                    rand_SL = self.pop[np.random.randint(0, self.pop_size)][self.ID_LOC_POS]
                    rand_SL = 2 * self.g_best[self.ID_POS] - rand_SL
                    pos_new = rand_SL - c * np.abs(np.random.uniform() * rand_SL - self.pop[idx][self.ID_POS])
            agent[self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(agent[self.ID_POS])
        pop_new = self.update_target_wrapper_population(pop_new)

        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.pop[idx] = deepcopy(pop_new[idx])
                if self.compare_agent(pop_new[idx], [None, self.pop[idx][self.ID_LOC_FIT]]):
                    self.pop[idx][self.ID_LOC_POS] = deepcopy(pop_new[idx][self.ID_POS])
                    self.pop[idx][self.ID_LOC_FIT] = deepcopy(pop_new[idx][self.ID_TAR])


class ImprovedSLO(ModifiedSLO):
    """
    The original version: Improved Sea Lion Optimization (ImprovedSLO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): Local coefficient same as PSO, default = 1.2
        + c2 (float): Global coefficient same as PSO, default = 1.2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.SLO import ImprovedSLO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> c1 = 1.2
    >>> c2 = 1.5
    >>> model = ImprovedSLO(epoch, pop_size, c1, c2)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, c1=1.2, c2=1.2, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c1 (float): Local coefficient same as PSO, default = 1.2
            c2 (float): Global coefficient same as PSO, default = 1.2
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.set_parameters(["epoch", "pop_size", "c1", "c2"])
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        c = 2 - 2 * epoch / self.epoch
        t0 = np.random.rand()
        v1 = np.sin(2 * np.pi * t0)
        v2 = np.sin(2 * np.pi * (1 - t0))
        SP_leader = np.abs(v1 * (1 + v2) / v2)

        pop_new = []
        for idx in range(0, self.pop_size):
            agent = deepcopy(self.pop[idx])
            if SP_leader < 0.5:
                if c < 1:  # Exploitation improved by historical movement + global best affect
                    # pos_new = g_best[self.ID_POS] - c * np.abs(2 * rand() * g_best[self.ID_POS] - pop[i][self.ID_POS])
                    dif1 = np.abs(2 * np.random.rand() * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    dif2 = np.abs(2 * np.random.rand() * self.pop[idx][self.ID_LOC_POS] - self.pop[idx][self.ID_POS])
                    pos_new = self.c1 * np.random.rand() * (self.pop[idx][self.ID_POS] - dif1) + \
                              self.c2 * np.random.rand() * (self.pop[idx][self.ID_POS] - dif2)
                else:  # Exploration improved by opposition-based learning
                    # Create a new solution by equation below
                    # Then create an opposition solution of above solution
                    # Compare both of them and keep the good one (Searching at both direction)
                    pos_new = self.g_best[self.ID_POS] + c * np.random.normal(0, 1, self.problem.n_dims) * \
                              (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                    target_new = self.get_target_wrapper(pos_new)
                    pos_new_oppo = self.problem.lb + self.problem.ub - self.g_best[self.ID_POS] + \
                                   np.random.rand() * (self.g_best[self.ID_POS] - pos_new)
                    target_new_oppo = self.get_target_wrapper(self.amend_position(pos_new_oppo, self.problem.lb, self.problem.ub))
                    if self.compare_agent([pos_new_oppo, target_new_oppo], [pos_new, target_new]):
                        pos_new = pos_new_oppo
            else:  # Exploitation
                pos_new = self.g_best[self.ID_POS] + np.cos(2 * np.pi * np.random.uniform(-1, 1)) * \
                          np.abs(self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            agent[self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(agent[self.ID_POS])
        pop_new = self.update_target_wrapper_population(pop_new)

        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.pop[idx] = deepcopy(pop_new[idx])
                if self.compare_agent(pop_new[idx], [None, self.pop[idx][self.ID_LOC_FIT]]):
                    self.pop[idx][self.ID_LOC_POS] = deepcopy(pop_new[idx][self.ID_POS])
                    self.pop[idx][self.ID_LOC_FIT] = deepcopy(pop_new[idx][self.ID_TAR])
