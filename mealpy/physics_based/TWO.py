#!/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalTWO(Optimizer):
    """
    The original version of: Tug of War Optimization (TWO)

    Links:
        1. https://www.researchgate.net/publication/332088054_Tug_of_War_Optimization_Algorithm

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.TWO import OriginalTWO
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
    >>> model = OriginalTWO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Kaveh, A., 2017. Tug of war optimization. In Advances in metaheuristic algorithms for
    optimal design of structures (pp. 451-487). Springer, Cham.
    """

    ID_POS = 0
    ID_TAR = 1
    ID_WEIGHT = 2

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
        self.sort_flag = False
        self.muy_s = 1
        self.muy_k = 1
        self.delta_t = 1
        self.alpha = 0.99
        self.beta = 0.1

    def initialization(self):
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)
        self.pop = self.update_weight__(self.pop)

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, target, weight]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        weight = 0.0
        return [position, target, weight]

    def update_weight__(self, teams):
        list_fits = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in teams])
        maxx, minn = np.max(list_fits), np.min(list_fits)
        if maxx == minn:
            list_fits = np.random.uniform(0.0, 1.0, self.pop_size)
        list_weights = np.exp(-(list_fits - maxx) / (maxx - minn))
        list_weights = list_weights/np.sum(list_weights) + 0.1
        for idx in range(self.pop_size):
            teams[idx][self.ID_WEIGHT] = list_weights[idx]
        return teams

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = deepcopy(self.pop)
        for idx in range(self.pop_size):
            pos_new = pop_new[idx][self.ID_POS].astype(float)
            for j in range(self.pop_size):
                if self.pop[idx][self.ID_WEIGHT] < self.pop[j][self.ID_WEIGHT]:
                    force = max(self.pop[idx][self.ID_WEIGHT] * self.muy_s, self.pop[j][self.ID_WEIGHT] * self.muy_s)
                    resultant_force = force - self.pop[idx][self.ID_WEIGHT] * self.muy_k
                    g = self.pop[j][self.ID_POS] - self.pop[idx][self.ID_POS]
                    acceleration = resultant_force * g / (self.pop[idx][self.ID_WEIGHT] * self.muy_k)
                    delta_x = 0.5 * acceleration + np.power(self.alpha, epoch + 1) * self.beta * \
                              (self.problem.ub - self.problem.lb) * np.random.normal(0, 1, self.problem.n_dims)
                    pos_new += delta_x
            pop_new[idx][self.ID_POS] = pos_new
        for idx in range(self.pop_size):
            pos_new = pop_new[idx][self.ID_POS].astype(float)
            for j in range(self.problem.n_dims):
                if pos_new[j] < self.problem.lb[j] or pos_new[j] > self.problem.ub[j]:
                    if np.random.random() <= 0.5:
                        pos_new[j] = self.g_best[self.ID_POS][j] + np.random.randn() / (epoch + 1) * \
                                                     (self.g_best[self.ID_POS][j] - pos_new[j])
                        if pos_new[j] < self.problem.lb[j] or pos_new[j] > self.problem.ub[j]:
                            pos_new[j] = self.pop[idx][self.ID_POS][j]
                    else:
                        if pos_new[j] < self.problem.lb[j]:
                            pos_new[j] = self.problem.lb[j]
                        if pos_new[j] > self.problem.ub[j]:
                            pos_new[j] = self.problem.ub[j]
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new[idx][self.ID_POS] = pos_new
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[idx][self.ID_TAR] = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(pop_new[idx], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
        self.pop = self.update_weight__(self.pop)


class OppoTWO(OriginalTWO):
    """
    The opossition-based learning version: Tug of War Optimization (OTWO)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.TWO import OppoTWO
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
    >>> model = OppoTWO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)

    def initialization(self):
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)
        list_idx = np.random.choice(range(0, self.pop_size), int(self.pop_size/2), replace=False)
        pop_temp = [self.pop[list_idx[idx]] for idx in range(0, int(self.pop_size/2))]
        pop_oppo = []
        for i in range(len(pop_temp)):
            pos_opposite = self.problem.ub + self.problem.lb - pop_temp[i][self.ID_POS]
            pos_opposite = self.amend_position(pos_opposite, self.problem.lb, self.problem.ub)
            pop_oppo.append([pos_opposite, None, 0.0])
            if self.mode not in self.AVAILABLE_MODES:
                pop_oppo[-1][self.ID_TAR] = self.get_target_wrapper(pos_opposite)
        pop_oppo = self.update_target_wrapper_population(pop_oppo)
        self.pop = pop_temp + pop_oppo
        self.pop = self.update_weight__(self.pop)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Apply force of others solution on each individual solution
        pop_new = deepcopy(self.pop)
        for idx in range(self.pop_size):
            pos_new = pop_new[idx][self.ID_POS].astype(float)
            for j in range(self.pop_size):
                if self.pop[idx][self.ID_WEIGHT] < self.pop[j][self.ID_WEIGHT]:
                    force = max(self.pop[idx][self.ID_WEIGHT] * self.muy_s, self.pop[j][self.ID_WEIGHT] * self.muy_s)
                    resultant_force = force - self.pop[idx][self.ID_WEIGHT] * self.muy_k
                    g = self.pop[j][self.ID_POS] - self.pop[idx][self.ID_POS]
                    temp = (self.pop[idx][self.ID_WEIGHT] * self.muy_k)
                    acceleration = resultant_force * g / temp
                    delta_x = 1 / 2 * acceleration + np.power(self.alpha, epoch + 1) * self.beta * \
                              (self.problem.ub - self.problem.lb) * np.random.normal(0, 1, self.problem.n_dims)
                    pos_new += delta_x
            self.pop[idx][self.ID_POS] = pos_new

        ## Amend solution and update fitness value
        for idx in range(self.pop_size):
            pos_new = self.g_best[self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) / (epoch + 1) * \
                      (self.g_best[self.ID_POS] - pop_new[idx][self.ID_POS])
            conditions = np.logical_or(pop_new[idx][self.ID_POS] < self.problem.lb, pop_new[idx][self.ID_POS] > self.problem.ub)
            conditions = np.logical_and(conditions, np.random.random(self.problem.n_dims) < 0.5)
            pos_new = np.where(conditions, pos_new, self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new[idx][self.ID_POS] = pos_new
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[idx][self.ID_TAR] = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(pop_new[idx], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        ## Opposition-based here
        pop = []
        for idx in range(self.pop_size):
            C_op = self.create_opposition_position(self.pop[idx][self.ID_POS], self.g_best[self.ID_POS])
            pos_new = self.amend_position(C_op, self.problem.lb, self.problem.ub)
            pop.append([pos_new, None, 0.0])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target, 0.0], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_wrapper_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop)
        self.pop = self.update_weight__(self.pop)


class LevyTWO(OriginalTWO):
    """
    The Levy-flight version of: Tug of War Optimization (LevyTWO)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.TWO import LevyTWO
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
    >>> model = LevyTWO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = deepcopy(self.pop)
        for i in range(self.pop_size):
            pos_new = self.pop[i][self.ID_POS].astype(float)
            for k in range(self.pop_size):
                if self.pop[i][self.ID_WEIGHT] < self.pop[k][self.ID_WEIGHT]:
                    force = max(self.pop[i][self.ID_WEIGHT] * self.muy_s, self.pop[k][self.ID_WEIGHT] * self.muy_s)
                    resultant_force = force - self.pop[i][self.ID_WEIGHT] * self.muy_k
                    g = self.pop[k][self.ID_POS] - self.pop[i][self.ID_POS]
                    acceleration = resultant_force * g / (self.pop[i][self.ID_WEIGHT] * self.muy_k)
                    delta_x = 1 / 2 * acceleration + np.power(self.alpha, epoch + 1) * self.beta * \
                              (self.problem.ub - self.problem.lb) * np.random.normal(0, 1, self.problem.n_dims)
                    pos_new +=delta_x
            pop_new[i][self.ID_POS] = pos_new
        for i in range(self.pop_size):
            pos_new = self.pop[i][self.ID_POS].astype(float)
            for j in range(self.problem.n_dims):
                if pos_new[j] < self.problem.lb[j] or pos_new[j] > self.problem.ub[j]:
                    if np.random.random() <= 0.5:
                        pos_new[j] = self.g_best[self.ID_POS][j] + np.random.randn() / (epoch + 1) * \
                                                     (self.g_best[self.ID_POS][j] - pos_new[j])
                        if pos_new[j] < self.problem.lb[j] or pos_new[j] > self.problem.ub[j]:
                            pos_new[j] = self.pop[i][self.ID_POS][j]
                    else:
                        if pos_new[j] < self.problem.lb[j]:
                            pos_new[j] = self.problem.lb[j]
                        if pos_new[j] > self.problem.ub[j]:
                            pos_new[j] = self.problem.ub[j]
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new[i][self.ID_POS] = pos_new
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[i][self.ID_TAR] = self.get_target_wrapper(pos_new)
                self.pop[i] = self.get_better_solution(pop_new[i], self.pop[i])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        ### Apply levy-flight here
        for i in range(self.pop_size):
            ## Chance for each agent to update using levy is 50%
            if np.random.rand() < 0.5:
                levy_step = self.get_levy_flight_step(beta=1.0, multiplier=0.1, size=self.problem.n_dims, case=-1)
                pos_new = pop_new[i][self.ID_POS] + levy_step
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                target = self.get_target_wrapper(pos_new)
                if self.compare_agent([pos_new, target, 0.0], pop_new[i]):
                    pop_new[i] = [pos_new, target, 0.0]
        self.pop = self.update_weight__(pop_new)


class EnhancedTWO(OppoTWO, LevyTWO):
    """
    The original version of: Enhenced Tug of War Optimization (ETWO)

    Links:
        1. https://doi.org/10.1016/j.procs.2020.03.063

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.TWO import EnhancedTWO
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
    >>> model = EnhancedTWO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, T., Hoang, B., Nguyen, G. and Nguyen, B.M., 2020. A new workload prediction model using
    extreme learning machine and enhanced tug of war optimization. Procedia Computer Science, 170, pp.362-369.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)

    def initialization(self):
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)
        pop_oppo = deepcopy(self.pop)
        for i in range(self.pop_size):
            pos_opposite = self.problem.ub + self.problem.lb - self.pop[i][self.ID_POS]
            pos_new = self.amend_position(pos_opposite, self.problem.lb, self.problem.ub)
            pop_oppo[i][self.ID_POS] = pos_new
            if self.mode not in self.AVAILABLE_MODES:
                pop_oppo[i][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_oppo = self.update_target_wrapper_population(pop_oppo)
        self.pop = self.get_sorted_strim_population(self.pop + pop_oppo, self.pop_size)
        self.pop = self.update_weight__(self.pop)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = deepcopy(self.pop)
        for i in range(self.pop_size):
            pos_new = self.pop[i][self.ID_POS].astype(float)
            for k in range(self.pop_size):
                if self.pop[i][self.ID_WEIGHT] < self.pop[k][self.ID_WEIGHT]:
                    force = max(self.pop[i][self.ID_WEIGHT] * self.muy_s, self.pop[k][self.ID_WEIGHT] * self.muy_s)
                    resultant_force = force - self.pop[i][self.ID_WEIGHT] * self.muy_k
                    g = self.pop[k][self.ID_POS] - self.pop[i][self.ID_POS]
                    acceleration = resultant_force * g / (self.pop[i][self.ID_WEIGHT] * self.muy_k)
                    delta_x = 1 / 2 * acceleration + np.power(self.alpha, epoch + 1) * self.beta * \
                              (self.problem.ub - self.problem.lb) * np.random.normal(0, 1, self.problem.n_dims)
                    pos_new += delta_x
            pop_new[i][self.ID_POS] = pos_new
        for i in range(self.pop_size):
            pos_new = self.pop[i][self.ID_POS].astype(float)
            for j in range(self.problem.n_dims):
                if pos_new[j] < self.problem.lb[j] or pos_new[j] > self.problem.ub[j]:
                    if np.random.random() <= 0.5:
                        pos_new[j] = self.g_best[self.ID_POS][j] + np.random.randn() / (epoch + 1) * \
                                                     (self.g_best[self.ID_POS][j] - pos_new[j])
                        if pos_new[j] < self.problem.lb[j] or pos_new[j] > self.problem.ub[j]:
                            pos_new[j] = self.pop[i][self.ID_POS][j]
                    else:
                        if pos_new[j] < self.problem.lb[j]:
                            pos_new[j] = self.problem.lb[j]
                        if pos_new[j] > self.problem.ub[j]:
                            pos_new[j] = self.problem.ub[j]
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new[i][self.ID_POS] = pos_new
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[i][self.ID_TAR] = self.get_target_wrapper(pos_new)
                self.pop[i] = self.get_better_solution(pop_new[i], self.pop[i])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        for i in range(self.pop_size):
            C_op = self.create_opposition_position(pop_new[i][self.ID_POS], self.g_best[self.ID_POS])
            C_op = self.amend_position(C_op, self.problem.lb, self.problem.ub)
            target_op = self.get_target_wrapper(C_op)
            if self.compare_agent([C_op, target_op], pop_new[i]):
                pop_new[i] = [C_op, target_op, 0.0]
            else:
                levy_step = self.get_levy_flight_step(beta=1.0, multiplier=1.0, size=self.problem.n_dims, case=-1)
                pos_new = pop_new[i][self.ID_POS] + 1.0 / np.sqrt(epoch + 1) * levy_step
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                target = self.get_target_wrapper(pos_new)
                if self.compare_agent([pos_new, target], pop_new[i]):
                    pop_new[i] = [pos_new, target, 0.0]
        self.pop = self.update_weight__(pop_new)
