#!/usr/bin/env python
# Created by "Thieu" at 20:22, 12/06/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseSMA(Optimizer):
    """
    The developed version: Slime Mould Algorithm (SMA)

    Notes
    ~~~~~
    + Selected 2 unique and random solution to create new solution (not to create variable)
    + Check bound and compare old position with new position to get the best one

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_t (float): (0, 1.0) -> better [0.01, 0.1], probability threshold (z in the paper)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.SMA import BaseSMA
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
    >>> p_t = 0.03
    >>> model = BaseSMA(epoch, pop_size, p_t)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    ID_WEI = 2

    def __init__(self, epoch=10000, pop_size=100, p_t=0.03, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_t (float): probability threshold (z in the paper), default = 0.03
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.p_t = self.validator.check_float("p_t", p_t, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "p_t"])
        self.sort_flag = True

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: [position, target, weight]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        weight = np.zeros(len(lb))
        return [position, target, weight]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # plus eps to avoid denominator zero
        s = self.g_best[self.ID_TAR][self.ID_FIT] - self.pop[-1][self.ID_TAR][self.ID_FIT] + self.EPSILON

        # calculate the fitness weight of each slime mold
        for i in range(0, self.pop_size):
            # Eq.(2.5)
            if i <= int(self.pop_size / 2):
                self.pop[i][self.ID_WEI] = 1 + np.random.uniform(0, 1, self.problem.n_dims) * \
                    np.log10((self.g_best[self.ID_TAR][self.ID_FIT] - self.pop[i][self.ID_TAR][self.ID_FIT]) / s + 1)
            else:
                self.pop[i][self.ID_WEI] = 1 - np.random.uniform(0, 1, self.problem.n_dims) * \
                    np.log10((self.g_best[self.ID_TAR][self.ID_FIT] - self.pop[i][self.ID_TAR][self.ID_FIT]) / s + 1)

        a = np.arctanh(-((epoch + 1) / (self.epoch+1)) + 1)  # Eq.(2.4)
        b = 1 - (epoch + 1) / (self.epoch+1)

        pop_new = []
        for idx in range(0, self.pop_size):
            # Update the Position of search agent
            if np.random.uniform() < self.p_t:  # Eq.(2.7)
                pos_new = self.generate_position(self.problem.lb, self.problem.ub)
            else:
                p = np.tanh(np.abs(self.pop[idx][self.ID_TAR][self.ID_FIT] - self.g_best[self.ID_TAR][self.ID_FIT]))  # Eq.(2.2)
                vb = np.random.uniform(-a, a, self.problem.n_dims)  # Eq.(2.3)
                vc = np.random.uniform(-b, b, self.problem.n_dims)

                # two positions randomly selected from population, apply for the whole problem size instead of 1 variable
                id_a, id_b = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)

                pos_1 = self.g_best[self.ID_POS] + vb * (self.pop[idx][self.ID_WEI] * self.pop[id_a][self.ID_POS] - self.pop[id_b][self.ID_POS])
                pos_2 = vc * self.pop[idx][self.ID_POS]
                condition = np.random.random(self.problem.n_dims) < p
                pos_new = np.where(condition, pos_1, pos_2)
            # Check bound and re-calculate fitness after each individual move
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None, np.zeros(self.problem.n_dims)])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target, np.zeros(self.problem.n_dims)], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)


class OriginalSMA(BaseSMA):
    """
    The original version of: Slime Mould Algorithm (SMA)

    Links:
        1. https://doi.org/10.1016/j.future.2020.03.055
        2. https://www.researchgate.net/publication/340431861_Slime_mould_algorithm_A_new_method_for_stochastic_optimization

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_t (float): (0, 1.0) -> better [0.01, 0.1], probability threshold (z in the paper)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.SMA import OriginalSMA
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
    >>> p_t = 0.03
    >>> model = OriginalSMA( epoch, pop_size, p_t)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Li, S., Chen, H., Wang, M., Heidari, A.A. and Mirjalili, S., 2020. Slime mould algorithm: A new method for
    stochastic optimization. Future Generation Computer Systems, 111, pp.300-323.
    """

    ID_WEI = 2

    def __init__(self, epoch=10000, pop_size=100, p_t=0.03, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 1000
            pop_size (int): number of population size, default = 100
            p_t (float): probability threshold (z in the paper), default = 0.03
        """
        super().__init__(epoch, pop_size, p_t, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """

        # plus eps to avoid denominator zero
        s = self.g_best[self.ID_TAR][self.ID_FIT] - self.pop[-1][self.ID_TAR][self.ID_FIT] + self.EPSILON

        # calculate the fitness weight of each slime mold
        for i in range(0, self.pop_size):
            # Eq.(2.5)
            if i <= int(self.pop_size / 2):
                self.pop[i][self.ID_WEI] = 1 + np.random.uniform(0, 1, self.problem.n_dims) * \
                    np.log10((self.g_best[self.ID_TAR][self.ID_FIT] - self.pop[i][self.ID_TAR][self.ID_FIT]) / s + 1)
            else:
                self.pop[i][self.ID_WEI] = 1 - np.random.uniform(0, 1, self.problem.n_dims) * \
                    np.log10((self.g_best[self.ID_TAR][self.ID_FIT] - self.pop[i][self.ID_TAR][self.ID_FIT]) / s + 1)

        a = np.arctanh(-((epoch + 1) / (self.epoch+1)) + 1)  # Eq.(2.4)
        b = 1 - (epoch + 1) / (self.epoch+1)

        pop_new = []
        for idx in range(0, self.pop_size):
            # Update the Position of search agent
            current_agent = deepcopy(self.pop[idx])
            if np.random.uniform() < self.p_t:  # Eq.(2.7)
                current_agent[self.ID_POS] = np.random.uniform(self.problem.lb, self.problem.ub)
            else:
                p = np.tanh(np.abs(current_agent[self.ID_TAR][self.ID_FIT] - self.g_best[self.ID_TAR][self.ID_FIT]))  # Eq.(2.2)
                vb = np.random.uniform(-a, a, self.problem.n_dims)  # Eq.(2.3)
                vc = np.random.uniform(-b, b, self.problem.n_dims)
                for j in range(0, self.problem.n_dims):
                    # two positions randomly selected from population
                    id_a, id_b = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                    if np.random.uniform() < p:  # Eq.(2.1)
                        current_agent[self.ID_POS][j] = self.g_best[self.ID_POS][j] + \
                            vb[j] * (current_agent[self.ID_WEI][j] * self.pop[id_a][self.ID_POS][j] - self.pop[id_b][self.ID_POS][j])
                    else:
                        current_agent[self.ID_POS][j] = vc[j] * current_agent[self.ID_POS][j]
            pos_new = self.amend_position(current_agent[self.ID_POS], self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None, np.zeros(self.problem.n_dims)])
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx] = [pos_new, self.get_target_wrapper(pos_new), np.zeros(self.problem.n_dims)]
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_wrapper_population(pop_new)
