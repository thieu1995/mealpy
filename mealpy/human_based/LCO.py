#!/usr/bin/env python
# Created by "Thieu" at 11:16, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalLCO(Optimizer):
    """
    The original version of: Life Choice-based Optimization (LCO)

    Links:
        1. https://doi.org/10.1007/s00500-019-04443-z

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r1 (float): [1.5, 4], coefficient factor, default = 2.35

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.LCO import OriginalLCO
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
    >>> r1 = 2.35
    >>> model = OriginalLCO(epoch, pop_size, r1)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Khatri, A., Gaba, A., Rana, K.P.S. and Kumar, V., 2020. A novel life choice-based optimizer. Soft Computing, 24(12), pp.9121-9141.
    """

    def __init__(self, epoch=10000, pop_size=100, r1=2.35, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r1 (float): coefficient factor
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.r1 = self.validator.check_float("r1", r1, [1.0, 3.0])
        self.set_parameters(["epoch", "pop_size", "r1"])
        self.n_agents = int(np.ceil(np.sqrt(self.pop_size)))
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            prob = np.random.rand()
            if prob > 0.875:  # Update using Eq. 1, update from n best position
                temp = np.array([np.random.rand() * self.pop[j][self.ID_POS] for j in range(0, self.n_agents)])
                temp = np.mean(temp, axis=0)
            elif prob < 0.7:  # Update using Eq. 2-6
                f1 = 1 - epoch / self.epoch
                f2 = 1 - f1
                prev_pos = self.g_best[self.ID_POS] if idx == 0 else self.pop[idx-1][self.ID_POS]
                best_diff = f1 * self.r1 * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                better_diff = f2 * self.r1 * (prev_pos - self.pop[idx][self.ID_POS])
                temp = self.pop[idx][self.ID_POS] + np.random.rand() * better_diff + np.random.rand() * best_diff
            else:
                temp = self.problem.ub - (self.pop[idx][self.ID_POS] - self.problem.lb) * np.random.rand()
            pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(pop_new, self.pop)


class BaseLCO(OriginalLCO):
    """
    The developed version: Life Choice-based Optimization (LCO)

    Notes
    ~~~~~
    The flow is changed with if else statement.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r1 (float): [1.5, 4], coefficient factor, default = 2.35

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.LCO import BaseLCO
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
    >>> r1 = 2.35
    >>> model = BaseLCO(epoch, pop_size, r1)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, r1=2.35, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r1 (float): coefficient factor
        """
        super().__init__(epoch, pop_size, r1, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # epoch: current chance, self.epoch: number of chances
        pop_new = []
        for idx in range(0, self.pop_size):
            prob = np.random.rand()
            if prob > 0.875:  # Update using Eq. 1, update from n best position
                temp = np.array([np.random.rand() * self.pop[j][self.ID_POS] for j in range(0, self.n_agents)])
                temp = np.mean(temp, axis=0)
            elif prob < 0.7:  # Update using Eq. 2-6
                f = (epoch + 1) / self.epoch
                if idx != 0:
                    better_diff = f * self.r1 * (self.pop[idx - 1][self.ID_POS] - self.pop[idx][self.ID_POS])
                else:
                    better_diff = f * self.r1 * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                best_diff = (1 - f) * self.r1 * (self.pop[0][self.ID_POS] - self.pop[idx][self.ID_POS])
                temp = self.pop[idx][self.ID_POS] + np.random.rand() * better_diff + np.random.rand() * best_diff
            else:
                temp = self.generate_position(self.problem.lb, self.problem.ub)
            pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(pop_new, self.pop)


class ImprovedLCO(Optimizer):
    """
    The improved version: Life Choice-based Optimization (ILCO)

    Notes
    ~~~~~
    + The flow of the original LCO is kept.
    + Gaussian distribution and mutation mechanism are added
    + R1 parameter is removed

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.LCO import BaseLCO
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
    >>> model = BaseLCO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
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
        self.pop_len = int(self.pop_size / 2)
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # epoch: current chance, self.epoch: number of chances
        pop_new = []
        for idx in range(0, self.pop_size):
            rand = np.random.random()
            if rand > 0.875:  # Update using Eq. 1, update from n best position
                n = int(np.ceil(np.sqrt(self.pop_size)))
                pos_new = np.array([np.random.rand() * self.pop[j][self.ID_POS] for j in range(0, n)])
                pos_new = np.mean(pos_new, axis=0)
            elif rand < 0.7:  # Update using Eq. 2-6
                f = (epoch + 1) / self.epoch
                if idx != 0:
                    better_diff = f * np.random.rand() * (self.pop[idx - 1][self.ID_POS] - self.pop[idx][self.ID_POS])
                else:
                    better_diff = f * np.random.rand() * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                best_diff = (1 - f) * np.random.rand() * (self.pop[0][self.ID_POS] - self.pop[idx][self.ID_POS])
                pos_new = self.pop[idx][self.ID_POS] + better_diff + best_diff
            else:
                pos_new = self.problem.ub - (self.pop[idx][self.ID_POS] - self.problem.lb) * np.random.rand()
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(pop_new, self.pop)

        ## Sort the updated population based on fitness
        pop, local_best = self.get_global_best_solution(self.pop)
        pop_s1, pop_s2 = pop[:self.pop_len], pop[self.pop_len:]

        ## Mutation scheme
        pop_child1 = []
        for idx in range(0, self.pop_len):
            pos_new = pop_s1[idx][self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * pop_s1[idx][self.ID_POS]
            # np.random.rand() * ((epoch+1) / self.epoch) * np.random.normal(0, 1, self.problem.n_dims)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_child1.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                pop_s1[idx] = self.get_better_solution([pos_new, target], pop_s1[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_child1 = self.update_target_wrapper_population(pop_child1)
            pop_s1 = self.greedy_selection_population(pop_s1, pop_child1)

        ## Search Mechanism
        pos_s1_list = [item[self.ID_POS] for item in pop_s1]
        pos_s1_mean = np.mean(pos_s1_list, axis=0)
        pop_child2 = []
        for idx in range(0, self.pop_len):
            pos_new = local_best[self.ID_POS] + np.random.uniform(0, 1) * pos_s1_mean * ((epoch+1) / self.epoch)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_child2.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                pop_s2[idx] = self.get_better_solution(pop_s2[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_child2 = self.update_target_wrapper_population(pop_s2)
            pop_s2 = self.greedy_selection_population(pop_s2, pop_child2)
        ## Construct a new population
        self.pop = pop_s1 + pop_s2
