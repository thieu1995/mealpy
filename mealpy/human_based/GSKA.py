#!/usr/bin/env python
# Created by "Thieu" at 16:58, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseGSKA(Optimizer):
    """
    My changed version of: Gaining Sharing Knowledge-based Algorithm (GSKA)

    Notes
    ~~~~~
    + I remove all the third loop, remove 2 parameters
    + Solution represent junior or senior instead of dimension of solution
    + Change some equations for large-scale optimization
    + Apply the ideas of levy-flight and global best
    + Keep the better one after updating process

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.1, 0.5], percent of the best (p in the paper), default = 0.1
        + kr (float): [0.5, 0.9], knowledge ratio, default = 0.7

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.GSKA import BaseGSKA
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
    >>> pb = 0.1
    >>> kr = 0.9
    >>> model = BaseGSKA(problem_dict1, epoch, pop_size, pb, kr)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, problem, epoch=10000, pop_size=100, pb=0.1, kr=0.7, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, n: pop_size, m: clusters
            pb (float): percent of the best   0.1%, 0.8%, 0.1% (p in the paper), default = 0.1
            kr (float): knowledge ratio, default = 0.7
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.pb = self.validator.check_float("pb", pb, (0, 1.0))
        self.kr = self.validator.check_float("kr", kr, (0, 1.0))
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        D = int(np.ceil(self.pop_size * (1 - (epoch + 1) / self.epoch)))
        pop_new = []
        for idx in range(0, self.pop_size):
            # If it is the best it chooses best+2, best+1
            if idx == 0:
                previ, nexti = idx + 2, idx + 1
            # If it is the worse it chooses worst-2, worst-1
            elif idx == self.pop_size - 1:
                previ, nexti = idx - 2, idx - 1
            # Other case it chooses i-1, i+1
            else:
                previ, nexti = idx - 1, idx + 1

            if idx < D:  # senior gaining and sharing
                if np.random.uniform() <= self.kr:
                    rand_idx = np.random.choice(list(set(range(0, self.pop_size)) - {previ, idx, nexti}))
                    if self.compare_agent(self.pop[rand_idx], self.pop[idx]):
                        pos_new = self.pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * \
                                  (self.pop[previ][self.ID_POS] - self.pop[nexti][self.ID_POS] +
                                   self.pop[rand_idx][self.ID_POS] - self.pop[idx][self.ID_POS])
                    else:
                        pos_new = self.g_best[self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * \
                                  (self.pop[rand_idx][self.ID_POS] - self.pop[idx][self.ID_POS])
                else:
                    pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
            else:  # junior gaining and sharing
                if np.random.uniform() <= self.kr:
                    id1 = int(self.pb * self.pop_size)
                    id2 = id1 + int(self.pop_size - 2 * 100 * self.pb)
                    rand_best = np.random.choice(list(set(range(0, id1)) - {idx}))
                    rand_worst = np.random.choice(list(set(range(id2, self.pop_size)) - {idx}))
                    rand_mid = np.random.choice(list(set(range(id1, id2)) - {idx}))
                    if self.compare_agent(self.pop[rand_mid], self.pop[idx]):
                        pos_new = self.pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * \
                                  (self.pop[rand_best][self.ID_POS] - self.pop[rand_worst][self.ID_POS] +
                                   self.pop[rand_mid][self.ID_POS] - self.pop[idx][self.ID_POS])
                    else:
                        pos_new = self.g_best[self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * \
                                  (self.pop[rand_mid][self.ID_POS] - self.pop[idx][self.ID_POS])
                else:
                    pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(pop_new, self.pop)


class OriginalGSKA(Optimizer):
    """
    The original version of: Gaining Sharing Knowledge-based Algorithm (GSKA)

    Links:
        1. https://doi.org/10.1007/s13042-019-01053-x

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.1, 0.5], percent of the best (p in the paper), default = 0.1
        + kf (float): [0.3, 0.8], knowledge factor that controls the total amount of gained and shared knowledge added from others to the current individual during generations, default = 0.5
        + kr (float): [0.5, 0.95], knowledge ratio, default = 0.9
        + kg (int): [3, 20], number of generations effect to D-dimension, default = 5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.GSKA import OriginalGSKA
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
    >>> pb = 0.1
    >>> kf = 0.5
    >>> kr = 0.9
    >>> kg = 5
    >>> model = OriginalGSKA(problem_dict1, epoch, pop_size, pb, kf, kr, kg)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mohamed, A.W., Hadi, A.A. and Mohamed, A.K., 2020. Gaining-sharing knowledge based algorithm for solving
    optimization problems: a novel nature-inspired algorithm. International Journal of Machine Learning and Cybernetics, 11(7), pp.1501-1529.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, pb=0.1, kf=0.5, kr=0.9, kg=5, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, n: pop_size, m: clusters
            pb (float): percent of the best   0.1%, 0.8%, 0.1% (p in the paper), default = 0.1
            kf (float): knowledge factor that controls the total amount of gained and shared knowledge added
                        from others to the current individual during generations, default = 0.5
            kr (float): knowledge ratio, default = 0.9
            kg (int): Number of generations effect to D-dimension, default = 5
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.pb = self.validator.check_float("pb", pb, (0, 1.0))
        self.kf = self.validator.check_float("kf", kf, (0, 1.0))
        self.kr = self.validator.check_float("kr", kr, (0, 1.0))
        self.kg = self.validator.check_int("kg", kg, [1, 1 + int(epoch / 2)])
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        D = int(self.problem.n_dims * (1 - (epoch + 1) / self.epoch) ** self.kg)
        pop_new = []
        for idx in range(0, self.pop_size):
            # If it is the best it chooses best+2, best+1
            if idx == 0:
                previ, nexti = idx + 2, idx + 1
            # If it is the worse it chooses worst-2, worst-1
            elif idx == self.pop_size - 1:
                previ, nexti = idx - 2, idx - 1
            # Other case it chooses i-1, i+1
            else:
                previ, nexti = idx - 1, idx + 1

            # The random individual is for all dimension values
            rand_idx = np.random.choice(list(set(range(0, self.pop_size)) - {previ, idx, nexti}))
            pos_new = deepcopy(self.pop[idx][self.ID_POS])

            for j in range(0, self.problem.n_dims):
                if j < D:  # junior gaining and sharing
                    if np.random.uniform() <= self.kr:
                        if self.compare_agent(self.pop[rand_idx], self.pop[idx]):
                            pos_new[j] = self.pop[idx][self.ID_POS][j] + self.kf * \
                                         (self.pop[previ][self.ID_POS][j] - self.pop[nexti][self.ID_POS][j] +
                                          self.pop[rand_idx][self.ID_POS][j] - self.pop[idx][self.ID_POS][j])
                        else:
                            pos_new[j] = self.pop[idx][self.ID_POS][j] + self.kf * \
                                         (self.pop[previ][self.ID_POS][j] - self.pop[nexti][self.ID_POS][j] +
                                          self.pop[idx][self.ID_POS][j] - self.pop[rand_idx][self.ID_POS][j])
                else:  # senior gaining and sharing
                    if np.random.uniform() <= self.kr:
                        id1 = int(self.pb * self.pop_size)
                        id2 = id1 + int(self.pop_size - 2 * 100 * self.pb)
                        rand_best = np.random.choice(list(set(range(0, id1)) - {idx}))
                        rand_worst = np.random.choice(list(set(range(id2, self.pop_size)) - {idx}))
                        rand_mid = np.random.choice(list(set(range(id1, id2)) - {idx}))
                        if self.compare_agent(self.pop[rand_mid], self.pop[idx]):
                            pos_new[j] = self.pop[idx][self.ID_POS][j] + self.kf * \
                                         (self.pop[rand_best][self.ID_POS][j] - self.pop[rand_worst][self.ID_POS][j] +
                                          self.pop[rand_mid][self.ID_POS][j] - self.pop[idx][self.ID_POS][j])
                        else:
                            pos_new[j] = self.pop[idx][self.ID_POS][j] + self.kf * \
                                         (self.pop[rand_best][self.ID_POS][j] - self.pop[rand_worst][self.ID_POS][j] +
                                          self.pop[idx][self.ID_POS][j] - self.pop[rand_mid][self.ID_POS][j])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(pop_new, self.pop)
