# !/usr/bin/env python
# Created by "Thieu" at 09:17, 09/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from scipy.spatial.distance import cdist
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseBRO(Optimizer):
    """
    The developed version: Battle Royale Optimization (BRO)

    Notes
    ~~~~~
    The flow of algorithm is changed. Thrid loop is removed

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + threshold (int): [2, 5], dead threshold, default=3

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.BRO import BaseBRO
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
    >>> threshold = 3
    >>> model = BaseBRO(epoch, pop_size, threshold)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    ID_DAM = 2

    def __init__(self, epoch=10000, pop_size=100, threshold=3, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            threshold (int): dead threshold, default=3
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.threshold = self.validator.check_float("threshold", threshold, [1, 10])
        self.set_parameters(["epoch", "pop_size", "threshold"])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def initialize_variables(self):
        shrink = np.ceil(np.log10(self.epoch))
        self.dyn_delta = np.round(self.epoch / shrink)
        self.problem.lb_updated = deepcopy(self.problem.lb)
        self.problem.ub_updated = deepcopy(self.problem.ub)

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, target, damage]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        damage = 0
        return [position, target, damage]

    def get_idx_min__(self, data):
        k_zero = np.count_nonzero(data == 0)
        if k_zero == len(data):
            return np.random.choice(range(0, k_zero))
        ## 1st: Partition sorting, not good solution here.
        # return np.argpartition(data, k_zero)[k_zero]
        ## 2nd: Faster
        return np.where(data == np.min(data[data != 0]))[0][0]

    def find_idx_min_distance__(self, target_pos=None, pop=None):
        list_pos = np.array([pop[idx][self.ID_POS] for idx in range(0, self.pop_size)])
        target_pos = np.reshape(target_pos, (1, -1))
        dist_list = cdist(list_pos, target_pos, 'euclidean')
        dist_list = np.reshape(dist_list, (-1))
        return self.get_idx_min__(dist_list)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        for i in range(self.pop_size):
            # Compare ith soldier with nearest one (jth)
            j = self.find_idx_min_distance__(self.pop[i][self.ID_POS], self.pop)
            if self.compare_agent(self.pop[i], self.pop[j]):
                ## Update Winner based on global best solution
                pos_new = self.pop[i][self.ID_POS] + np.random.normal(0, 1) * \
                          np.mean(np.array([self.pop[i][self.ID_POS], self.g_best[self.ID_POS]]), axis=0)
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                target = self.get_target_wrapper(pos_new)
                dam_new = self.pop[i][self.ID_DAM] - 1  ## Substract damaged hurt -1 to go next battle
                self.pop[i] = [pos_new, target, dam_new]
                ## Update Loser
                if self.pop[j][self.ID_DAM] < self.threshold:  ## If loser not dead yet, move it based on general
                    pos_new = np.random.uniform() * (np.maximum(self.pop[j][self.ID_POS], self.g_best[self.ID_POS]) -
                                                     np.minimum(self.pop[j][self.ID_POS], self.g_best[self.ID_POS])) + \
                              np.maximum(self.pop[j][self.ID_POS], self.g_best[self.ID_POS])
                    dam_new = self.pop[j][self.ID_DAM] + 1

                    self.pop[j][self.ID_TAR] = self.get_target_wrapper(self.pop[j][self.ID_POS])
                else:  ## Loser dead and respawn again
                    pos_new = self.generate_position(self.problem.lb_updated, self.problem.ub_updated)
                    dam_new = 0
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                target = self.get_target_wrapper(pos_new)
                self.pop[j] = [pos_new, target, dam_new]
                nfe_epoch += 2
            else:
                ## Update Loser by following position of Winner
                self.pop[i] = deepcopy(self.pop[j])
                ## Update Winner by following position of General to protect the King and General
                pos_new = self.pop[j][self.ID_POS] + np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[j][self.ID_POS])
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                target = self.get_target_wrapper(pos_new)
                dam_new = 0
                self.pop[j] = [pos_new, target, dam_new]
                nfe_epoch += 1
        self.nfe_per_epoch = nfe_epoch
        if epoch >= self.dyn_delta:  # max_epoch = 1000 -> delta = 300, 450, >500,....
            pos_list = np.array([self.pop[idx][self.ID_POS] for idx in range(0, self.pop_size)])
            pos_std = np.std(pos_list, axis=0)
            lb = self.g_best[self.ID_POS] - pos_std
            ub = self.g_best[self.ID_POS] + pos_std
            self.problem.lb_updated = np.clip(lb, self.problem.lb_updated, self.problem.ub_updated)
            self.problem.ub_updated = np.clip(ub, self.problem.lb_updated, self.problem.ub_updated)
            self.dyn_delta += np.round(self.dyn_delta / 2)


class OriginalBRO(BaseBRO):
    """
    The original version of: Battle Royale Optimization (BRO)

    Links:
        1. https://doi.org/10.1007/s00521-020-05004-4

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + threshold (int): [2, 5], dead threshold, default=3

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.BRO import BaseBRO
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
    >>> threshold = 3
    >>> model = BaseBRO(epoch, pop_size, threshold)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Rahkar Farshi, T., 2021. Battle royale optimization algorithm. Neural Computing and Applications, 33(4), pp.1139-1157.
    """

    def __init__(self, epoch=10000, pop_size=100, threshold=3, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            threshold (int): dead threshold, default=3
        """
        super().__init__(epoch, pop_size, threshold, **kwargs)
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for i in range(self.pop_size):
            # Compare ith soldier with nearest one (jth)
            j = self.find_idx_min_distance__(self.pop[i][self.ID_POS], self.pop)
            dam, vic = i, j  ## This error in the algorithm's flow in the paper, But in the matlab code, he changed.
            if self.compare_agent(self.pop[i], self.pop[j]):
                dam, vic = j, i  ## The mistake also here in the paper.
            if self.pop[dam][self.ID_DAM] < self.threshold:
                pos_new = np.random.uniform(0, 1, self.problem.n_dims) * \
                          (np.maximum(self.pop[dam][self.ID_POS], self.g_best[self.ID_POS]) -
                           np.minimum(self.pop[dam][self.ID_POS], self.g_best[self.ID_POS])) + \
                          np.maximum(self.pop[dam][self.ID_POS], self.g_best[self.ID_POS])
                self.pop[dam][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                self.pop[dam][self.ID_TAR] = self.get_target_wrapper(self.pop[dam][self.ID_POS])
                self.pop[dam][self.ID_DAM] += 1
                self.pop[vic][self.ID_DAM] = 0
            else:
                self.pop[dam] = self.create_solution(self.problem.lb_updated, self.problem.ub_updated)
        if epoch >= self.dyn_delta:
            pos_list = np.array([self.pop[idx][self.ID_POS] for idx in range(0, self.pop_size)])
            pos_std = np.std(pos_list, axis=0)
            lb = self.g_best[self.ID_POS] - pos_std
            ub = self.g_best[self.ID_POS] + pos_std

            self.problem.lb_updated = np.clip(lb, self.problem.lb_updated, self.problem.ub_updated)
            self.problem.ub_updated = np.clip(ub, self.problem.lb_updated, self.problem.ub_updated)
            self.dyn_delta += round(self.dyn_delta / 2)
