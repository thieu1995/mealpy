#!/usr/bin/env python
# Created by "Thieu" at 08:37, 17/06/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from scipy.stats import cauchy
from copy import deepcopy


class OriginalSHADE(Optimizer):
    """
    The original version of: Success-History Adaptation Differential Evolution (OriginalSHADE)

    Links:
        1. https://doi.org/10.1109/CEC.2013.6557555

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + miu_f (float): [0.4, 0.6], initial weighting factor, default = 0.5
        + miu_cr (float): [0.4, 0.6], initial cross-over probability, default = 0.5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.SHADE import OriginalSHADE
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
    >>> miu_f = 0.5
    >>> miu_cr = 0.5
    >>> model = OriginalSHADE(epoch, pop_size, miu_f, miu_cr)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Tanabe, R. and Fukunaga, A., 2013, June. Success-history based parameter adaptation for
    differential evolution. In 2013 IEEE congress on evolutionary computation (pp. 71-78). IEEE.
    """

    def __init__(self, epoch=750, pop_size=100, miu_f=0.5, miu_cr=0.5, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_f (float): initial weighting factor, default = 0.5
            miu_cr (float): initial cross-over probability, default = 0.5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        # the initial f, location is changed then that f is good
        self.miu_f = self.validator.check_float("miu_f", miu_f, (0, 1.0))
        # the initial cr,
        self.miu_cr = self.validator.check_float("miu_cr", miu_cr, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "miu_f", "miu_cr"])
        self.sort_flag = False

    def initialize_variables(self):
        self.dyn_miu_f = self.miu_f * np.ones(self.pop_size)  # list the initial f,
        self.dyn_miu_cr = self.miu_cr * np.ones(self.pop_size)  # list the initial cr,
        self.dyn_pop_archive = list()
        self.k_counter = 0

    ### Survivor Selection
    def weighted_lehmer_mean__(self, list_objects, list_weights):
        up = list_weights * list_objects ** 2
        down = list_weights * list_objects
        return np.sum(up) / np.sum(down)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        list_f = list()
        list_cr = list()
        list_f_index = list()
        list_cr_index = list()

        list_f_new = np.ones(self.pop_size)
        list_cr_new = np.ones(self.pop_size)
        pop_old = deepcopy(self.pop)
        pop_sorted = self.get_sorted_strim_population(self.pop)

        pop = []
        for idx in range(0, self.pop_size):
            ## Calculate adaptive parameter cr and f
            idx_rand = np.random.randint(0, self.pop_size)
            cr = np.random.normal(self.dyn_miu_cr[idx_rand], 0.1)
            cr = np.clip(cr, 0, 1)
            while True:
                f = cauchy.rvs(self.dyn_miu_f[idx_rand], 0.1)
                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break
            list_cr_new[idx] = cr
            list_f_new[idx] = f
            p = np.random.uniform(2 / self.pop_size, 0.2)
            top = int(self.pop_size * p)
            x_best = pop_sorted[np.random.randint(0, top)]
            x_r1 = self.pop[np.random.choice(list(set(range(0, self.pop_size)) - {idx}))]
            new_pop = self.pop + self.dyn_pop_archive
            while True:
                x_r2 = new_pop[np.random.randint(0, len(new_pop))]
                if np.any(x_r2[self.ID_POS] - x_r1[self.ID_POS]) and np.any(x_r2[self.ID_POS] - self.pop[idx][self.ID_POS]):
                    break
            x_new = self.pop[idx][self.ID_POS] + f * (x_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + f * (x_r1[self.ID_POS] - x_r2[self.ID_POS])
            condition = np.random.random(self.problem.n_dims) < cr
            pos_new = np.where(condition, x_new, self.pop[idx][self.ID_POS])
            j_rand = np.random.randint(0, self.problem.n_dims)
            pos_new[j_rand] = x_new[j_rand]
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop = self.update_target_wrapper_population(pop)

        for i in range(0, self.pop_size):
            if self.compare_agent(pop[i], self.pop[i]):
                list_cr.append(list_cr_new[i])
                list_f.append(list_f_new[i])
                list_f_index.append(i)
                list_cr_index.append(i)
                self.pop[i] = deepcopy(pop[i])
                self.dyn_pop_archive.append(deepcopy(pop[i]))

        # Randomly remove solution
        temp = len(self.dyn_pop_archive) - self.pop_size
        if temp > 0:
            idx_list = np.random.choice(range(0, len(self.dyn_pop_archive)), temp, replace=False)
            archive_pop_new = []
            for idx, solution in enumerate(self.dyn_pop_archive):
                if idx not in idx_list:
                    archive_pop_new.append(solution)
            self.dyn_pop_archive = deepcopy(archive_pop_new)

        # Update miu_cr and miu_f
        if len(list_f) != 0 and len(list_cr) != 0:
            # Eq.13, 14, 10
            list_fit_old = np.ones(len(list_cr_index))
            list_fit_new = np.ones(len(list_cr_index))
            idx_increase = 0
            for i in range(0, self.pop_size):
                if i in list_cr_index:
                    list_fit_old[idx_increase] = pop_old[i][self.ID_TAR][self.ID_FIT]
                    list_fit_new[idx_increase] = self.pop[i][self.ID_TAR][self.ID_FIT]
                    idx_increase += 1
            temp = np.sum(np.abs(list_fit_new - list_fit_old))
            if temp == 0:
                list_weights = 1.0 / len(list_fit_new) * np.ones(len(list_fit_new))
            else:
                list_weights = np.abs(list_fit_new - list_fit_old) / temp
            self.dyn_miu_cr[self.k_counter] = np.sum(list_weights * np.array(list_cr))
            self.dyn_miu_f[self.k_counter] = self.weighted_lehmer_mean__(np.array(list_f), list_weights)
            self.k_counter += 1
            if self.k_counter >= self.pop_size:
                self.k_counter = 0


class L_SHADE(Optimizer):
    """
    The original version of: Linear Population Size Reduction Success-History Adaptation Differential Evolution (LSHADE)

    Links:
        1. https://metahack.org/CEC2014-Tanabe-Fukunaga.pdf

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + miu_f (float): [0.4, 0.6], initial weighting factor, default = 0.5
        + miu_cr (float): [0.4, 0.6], initial cross-over probability, default = 0.5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.SHADE import L_SHADE
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
    >>> miu_f = 0.5
    >>> miu_cr = 0.5
    >>> model = L_SHADE(epoch, pop_size, miu_f, miu_cr)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Tanabe, R. and Fukunaga, A.S., 2014, July. Improving the search performance of SHADE using
    linear population size reduction. In 2014 IEEE congress on evolutionary computation (CEC) (pp. 1658-1665). IEEE.
    """

    def __init__(self, epoch=750, pop_size=100, miu_f=0.5, miu_cr=0.5, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_f (float): initial weighting factor, default = 0.5
            miu_cr (float): initial cross-over probability, default = 0.5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.miu_f = self.validator.check_float("miu_f", miu_f, (0, 1.0))
        self.miu_cr = self.validator.check_float("miu_cr", miu_cr, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "miu_f", "miu_cr"])
        self.sort_flag = False

    def initialize_variables(self):
        # Dynamic variable
        self.dyn_miu_f = self.miu_f * np.ones(self.pop_size)  # list the initial f,
        self.dyn_miu_cr = self.miu_cr * np.ones(self.pop_size)  # list the initial cr,
        self.dyn_pop_archive = list()
        self.dyn_pop_size = self.pop_size
        self.k_counter = 0
        self.n_min = int(self.pop_size / 5)

    ### Survivor Selection
    def weighted_lehmer_mean__(self, list_objects, list_weights):
        up = np.sum(list_weights * list_objects ** 2)
        down = np.sum(list_weights * list_objects)
        return up / down if down != 0 else 0.5

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        list_f = list()
        list_cr = list()
        list_f_index = list()
        list_cr_index = list()

        list_f_new = np.ones(self.pop_size)
        list_cr_new = np.ones(self.pop_size)
        pop_old = deepcopy(self.pop)
        pop_sorted = self.get_sorted_strim_population(self.pop)

        pop = []
        for idx in range(0, self.pop_size):
            ## Calculate adaptive parameter cr and f
            idx_rand = np.random.randint(0, self.pop_size)
            cr = np.random.normal(self.dyn_miu_cr[idx_rand], 0.1)
            cr = np.clip(cr, 0, 1)
            while True:
                f = cauchy.rvs(self.dyn_miu_f[idx_rand], 0.1)
                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break
            list_cr_new[idx] = cr
            list_f_new[idx] = f
            p = np.random.uniform(0.15, 0.2)
            top = int(np.ceil(self.dyn_pop_size * p))
            x_best = pop_sorted[np.random.randint(0, top)]
            x_r1 = self.pop[np.random.choice(list(set(range(0, self.dyn_pop_size)) - {idx}))]
            new_pop = self.pop + self.dyn_pop_archive
            while True:
                x_r2 = new_pop[np.random.randint(0, len(new_pop))]
                if np.any(x_r2[self.ID_POS] - x_r1[self.ID_POS]) and np.any(x_r2[self.ID_POS] - self.pop[idx][self.ID_POS]):
                    break
            x_new = self.pop[idx][self.ID_POS] + f * (x_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + f * (x_r1[self.ID_POS] - x_r2[self.ID_POS])
            pos_new = np.where(np.random.random(self.problem.n_dims) < cr, x_new, self.pop[idx][self.ID_POS])
            j_rand = np.random.randint(0, self.problem.n_dims)
            pos_new[j_rand] = x_new[j_rand]
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop = self.update_target_wrapper_population(pop)

        for i in range(0, self.pop_size):
            if self.compare_agent(pop[i], self.pop[i]):
                list_cr.append(list_cr_new[i])
                list_f.append(list_f_new[i])
                list_f_index.append(i)
                list_cr_index.append(i)
                self.pop[i] = deepcopy(pop[i])
                self.dyn_pop_archive.append(deepcopy(self.pop[i]))

        # Randomly remove solution
        temp = len(self.dyn_pop_archive) - self.pop_size
        if temp > 0:
            idx_list = np.random.choice(range(0, len(self.dyn_pop_archive)), temp, replace=False)
            archive_pop_new = []
            for idx, solution in enumerate(self.dyn_pop_archive):
                if idx not in idx_list:
                    archive_pop_new.append(solution)
            self.dyn_pop_archive = deepcopy(archive_pop_new)

        # Update miu_cr and miu_f
        if len(list_f) != 0 and len(list_cr) != 0:
            # Eq.13, 14, 10
            list_fit_old = np.ones(len(list_cr_index))
            list_fit_new = np.ones(len(list_cr_index))
            idx_increase = 0
            for i in range(0, self.dyn_pop_size):
                if i in list_cr_index:
                    list_fit_old[idx_increase] = pop_old[i][self.ID_TAR][self.ID_FIT]
                    list_fit_new[idx_increase] = self.pop[i][self.ID_TAR][self.ID_FIT]
                    idx_increase += 1
            total_fit = np.sum(np.abs(list_fit_new - list_fit_old))
            list_weights = 0 if total_fit == 0 else np.abs(list_fit_new - list_fit_old) / total_fit
            self.dyn_miu_cr[self.k_counter] = np.sum(list_weights * np.array(list_cr))
            self.dyn_miu_f[self.k_counter] = self.weighted_lehmer_mean__(np.array(list_f), list_weights)
            self.k_counter += 1
            if self.k_counter >= self.dyn_pop_size:
                self.k_counter = 0

        # Linear Population Size Reduction
        self.dyn_pop_size = round(self.pop_size + epoch * ((self.n_min - self.pop_size) / self.epoch))
