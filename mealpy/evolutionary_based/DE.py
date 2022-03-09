# !/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from scipy.stats import cauchy
from copy import deepcopy


class BaseDE(Optimizer):
    """
    The original version of: Differential Evolution (DE)

    Links:
        1. https://doi.org/10.1016/j.swevo.2018.10.006

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + wf (float): [0.5, 0.95], weighting factor, default = 0.8
        + cr (float): [0.5, 0.95], crossover rate, default = 0.9
        + strategy (int): [0, 5], there are lots of variant version of DE algorithm,
            + 0: DE/current-to-rand/1/bin
            + 1: DE/best/1/bin
            + 2: DE/best/2/bin
            + 3: DE/rand/2/bin
            + 4: DE/current-to-best/1/bin
            + 5: DE/current-to-rand/1/bin

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.DE import BaseDE
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> wf = 0.01
    >>> cr = 2
    >>> strategy = 0
    >>> model = BaseDE(problem_dict1, epoch, pop_size, wf, cr, strategy)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mohamed, A.W., Hadi, A.A. and Jambi, K.M., 2019. Novel mutation strategy for enhancing SHADE and
    LSHADE algorithms for global numerical optimization. Swarm and Evolutionary Computation, 50, p.100455.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, wf=0.8, cr=0.9, strategy=0, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wf (float): weighting factor, default = 0.8
            cr (float): crossover rate, default = 0.9
            strategy (int): Different variants of DE, default = 0
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.weighting_factor = wf
        self.crossover_rate = cr
        self.strategy = strategy

    def _mutation__(self, current_pos, new_pos):
        pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.crossover_rate, current_pos, new_pos)
        return self.amend_position(pos_new)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = []
        if self.strategy == 0:
            # Choose 3 random element and different to i
            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                pos_new = self.pop[idx_list[0]][self.ID_POS] + self.weighting_factor * \
                          (self.pop[idx_list[1]][self.ID_POS] - self.pop[idx_list[2]][self.ID_POS])
                pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])
        elif self.strategy == 1:
            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = self.g_best[self.ID_POS] + self.weighting_factor * (self.pop[idx_list[0]][self.ID_POS] - self.pop[idx_list[1]][self.ID_POS])
                pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])
        elif self.strategy == 2:
            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 4, replace=False)
                pos_new = self.g_best[self.ID_POS] + self.weighting_factor * (self.pop[idx_list[0]][self.ID_POS] - self.pop[idx_list[1]][self.ID_POS]) + \
                          self.weighting_factor * (self.pop[idx_list[2]][self.ID_POS] - self.pop[idx_list[3]][self.ID_POS])
                pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])
        elif self.strategy == 3:
            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 5, replace=False)
                pos_new = self.pop[idx_list[0]][self.ID_POS] + self.weighting_factor * \
                          (self.pop[idx_list[1]][self.ID_POS] - self.pop[idx_list[2]][self.ID_POS]) + \
                          self.weighting_factor * (self.pop[idx_list[3]][self.ID_POS] - self.pop[idx_list[4]][self.ID_POS])
                pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])
        elif self.strategy == 4:
            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = self.pop[idx][self.ID_POS] + self.weighting_factor * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                          self.weighting_factor * (self.pop[idx_list[0]][self.ID_POS] - self.pop[idx_list[1]][self.ID_POS])
                pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])
        else:
            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                pos_new = self.pop[idx][self.ID_POS] + self.weighting_factor * (self.pop[idx_list[0]][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                          self.weighting_factor * (self.pop[idx_list[1]][self.ID_POS] - self.pop[idx_list[2]][self.ID_POS])
                pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])
        pop = self.update_fitness_population(pop)

        # create new pop by comparing fitness of corresponding each member in pop and children
        self.pop = self.greedy_selection_population(self.pop, pop)


class JADE(Optimizer):
    """
    The variant version of: Differential Evolution (JADE)

    Links:
        1. https://doi.org/10.1109/TEVC.2009.2014613

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + miu_f (float): [0.4, 0.6], initial adaptive f, default = 0.5
        + miu_cr (float): [0.4, 0.6], initial adaptive cr, default = 0.5
        + pt (float): [0.05, 0.2], The percent of top best agents (p in the paper), default = 0.1
        + ap (float): [0.05, 0.2], The Adaptation Parameter control value of f and cr (c in the paper), default=0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.DE import JADE
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> miu_f = 0.5
    >>> miu_cr = 0.5
    >>> pt = 0.1
    >>> ap = 0.1
    >>> model = JADE(problem_dict1, epoch, pop_size, miu_f, miu_cr, pt, ap)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Zhang, J. and Sanderson, A.C., 2009. JADE: adaptive differential evolution with optional
    external archive. IEEE Transactions on evolutionary computation, 13(5), pp.945-958.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, miu_f=0.5, miu_cr=0.5, pt=0.1, ap=0.1, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_f (float): initial adaptive f, default = 0.5
            miu_cr (float): initial adaptive cr, default = 0.5
            pt (float): The percent of top best agents (p in the paper), default = 0.1
            ap (float): The Adaptation Parameter control value of f and cr (c in the paper), default=0.1
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.miu_f = miu_f  # the initial f, location is changed then that f is good
        self.miu_cr = miu_cr  # the initial cr,
        self.pt = pt  # np.random.uniform(0.05, 0.2) # the x_best is select from the top 100p % solutions
        self.ap = ap  # np.random.uniform(1/20, 1/5) # the adaptation parameter control value of f and cr

        ## Dynamic variable, changing in run time
        self.dyn_miu_cr = self.miu_cr
        self.dyn_miu_f = self.miu_f
        self.dyn_pop_archive = list()

    ### Survivor Selection
    def lehmer_mean(self, list_objects):
        temp = sum(list_objects)
        return 0 if temp == 0 else sum(list_objects ** 2) / temp

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        list_f = list()
        list_cr = list()
        temp_f = list()
        temp_cr = list()

        pop_sorted = self.get_sorted_strim_population(self.pop)
        pop = []
        for idx in range(0, self.pop_size):
            ## Calculate adaptive parameter cr and f
            cr = np.random.normal(self.dyn_miu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            while True:
                f = cauchy.rvs(self.dyn_miu_f, 0.1)
                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break
            temp_f.append(f)
            temp_cr.append(cr)
            top = int(self.pop_size * self.pt)
            x_best = pop_sorted[np.random.randint(0, top)]
            x_r1 = self.pop[np.random.choice(list(set(range(0, self.pop_size)) - {idx}))]
            new_pop = self.pop + self.dyn_pop_archive
            while True:
                x_r2 = new_pop[np.random.randint(0, len(new_pop))]
                if np.any(x_r2[self.ID_POS] - x_r1[self.ID_POS]) and np.any(x_r2[self.ID_POS] - self.pop[idx][self.ID_POS]):
                    break
            x_new = self.pop[idx][self.ID_POS] + f * (x_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + f * (x_r1[self.ID_POS] - x_r2[self.ID_POS])
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < cr, x_new, self.pop[idx][self.ID_POS])
            j_rand = np.random.randint(0, self.problem.n_dims)
            pos_new[j_rand] = x_new[j_rand]
            pos_new = self.amend_position(pos_new)
            pop.append([pos_new, None])
        pop = self.update_fitness_population(pop)

        for idx in range(0, self.pop_size):
            if self.compare_agent(pop[idx], self.pop[idx]):
                self.dyn_pop_archive.append(deepcopy(self.pop[idx]))
                list_cr.append(temp_cr[idx])
                list_f.append(temp_f[idx])
                self.pop[idx] = deepcopy(pop[idx])

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
        if len(list_cr) == 0:
            self.dyn_miu_cr = (1 - self.ap) * self.dyn_miu_cr + self.ap * 0.5
        else:
            self.dyn_miu_cr = (1 - self.ap) * self.dyn_miu_cr + self.ap * np.mean(np.array(list_cr))
        if len(list_f) == 0:
            self.dyn_miu_f = (1 - self.ap) * self.dyn_miu_f + self.ap * 0.5
        else:
            self.dyn_miu_f = (1 - self.ap) * self.dyn_miu_f + self.ap * self.lehmer_mean(np.array(list_f))
        return pop


class SADE(Optimizer):
    """
    The original version of: Self-Adaptive Differential Evolution (SADE)

    Links:
        1. https://doi.org/10.1109/CEC.2005.1554904

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.DE import SADE
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> model = SADE(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Qin, A.K. and Suganthan, P.N., 2005, September. Self-adaptive differential evolution algorithm for
    numerical optimization. In 2005 IEEE congress on evolutionary computation (Vol. 2, pp. 1785-1791). IEEE.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size

        self.loop_probability = 50
        self.loop_cr = 5
        self.ns1 = self.ns2 = self.nf1 = self.nf2 = 0
        self.crm = 0.5
        self.p1 = 0.5

        # Dynamic variable
        self.dyn_list_cr = list()

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = []
        list_probability = []
        list_cr = []
        for idx in range(0, self.pop_size):
            ## Calculate adaptive parameter cr and f
            cr = np.random.normal(self.crm, 0.1)
            cr = np.clip(cr, 0, 1)
            list_cr.append(cr)
            while True:
                f = np.random.normal(0.5, 0.3)
                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break

            id1, id2, id3 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
            if np.random.rand() < self.p1:
                x_new = self.pop[id1][self.ID_POS] + f * (self.pop[id2][self.ID_POS] - self.pop[id3][self.ID_POS])
                pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < cr, x_new, self.pop[idx][self.ID_POS])
                j_rand = np.random.randint(0, self.problem.n_dims)
                pos_new[j_rand] = x_new[j_rand]
                pos_new = self.amend_position(pos_new)
                pop.append([pos_new, None])
                list_probability.append(True)
            else:
                x_new = self.pop[idx][self.ID_POS] + f * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                        f * (self.pop[id1][self.ID_POS] - self.pop[id2][self.ID_POS])
                pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < cr, x_new, self.pop[idx][self.ID_POS])
                j_rand = np.random.randint(0, self.problem.n_dims)
                pos_new[j_rand] = x_new[j_rand]
                pos_new = self.amend_position(pos_new)
                pop.append([pos_new, None])
                list_probability.append(False)
        pop = self.update_fitness_population(pop)

        for idx in range(0, self.pop_size):
            if list_probability[idx]:
                if self.compare_agent(pop[idx], self.pop[idx]):
                    self.ns1 += 1
                    self.pop[idx] = deepcopy(pop[idx])
                else:
                    self.nf1 += 1
            else:
                if self.compare_agent(pop[idx], self.pop[idx]):
                    self.ns2 += 1
                    self.dyn_list_cr.append(list_cr[idx])
                    self.pop[idx] = deepcopy(pop[idx])
                else:
                    self.nf2 += 1

        # Update cr and p1
        if (epoch + 1) / self.loop_cr == 0:
            self.crm = np.mean(self.dyn_list_cr)
            self.dyn_list_cr = list()

        if (epoch + 1) / self.loop_probability == 0:
            self.p1 = self.ns1 * (self.ns2 + self.nf2) / (self.ns2 * (self.ns1 + self.nf1) + self.ns1 * (self.ns2 + self.nf2))
            self.ns1 = self.ns2 = self.nf1 = self.nf2 = 0


class SHADE(Optimizer):
    """
    The variant version of: Success-History Adaptation Differential Evolution (SHADE)

    Links:
        1. https://doi.org/10.1109/CEC.2013.6557555

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + miu_f (float): [0.4, 0.6], initial weighting factor, default = 0.5
        + miu_cr (float): [0.4, 0.6], initial cross-over probability, default = 0.5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.DE import SHADE
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> miu_f = 0.5
    >>> miu_cr = 0.5
    >>> model = SHADE(problem_dict1, epoch, pop_size, miu_f, miu_cr)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Tanabe, R. and Fukunaga, A., 2013, June. Success-history based parameter adaptation for
    differential evolution. In 2013 IEEE congress on evolutionary computation (pp. 71-78). IEEE.
    """

    def __init__(self, problem, epoch=750, pop_size=100, miu_f=0.5, miu_cr=0.5, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_f (float): initial weighting factor, default = 0.5
            miu_cr (float): initial cross-over probability, default = 0.5
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size

        # Dynamic variable
        self.dyn_miu_f = miu_f * np.ones(self.pop_size)  # list the initial f,
        self.dyn_miu_cr = miu_cr * np.ones(self.pop_size)  # list the initial cr,
        self.dyn_pop_archive = list()
        self.k_counter = 0

    ### Survivor Selection
    def weighted_lehmer_mean(self, list_objects, list_weights):
        up = list_weights * list_objects ** 2
        down = list_weights * list_objects
        return sum(up) / sum(down)

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
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < cr, x_new, self.pop[idx][self.ID_POS])
            j_rand = np.random.randint(0, self.problem.n_dims)
            pos_new[j_rand] = x_new[j_rand]
            pos_new = self.amend_position(pos_new)
            pop.append([pos_new, None])
        pop = self.update_fitness_population(pop)

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
            temp = sum(abs(list_fit_new - list_fit_old))
            if temp == 0:
                list_weights = 1.0 / len(list_fit_new) * np.ones(len(list_fit_new))
            else:
                list_weights = abs(list_fit_new - list_fit_old) / temp
            self.dyn_miu_cr[self.k_counter] = sum(list_weights * np.array(list_cr))
            self.dyn_miu_f[self.k_counter] = self.weighted_lehmer_mean(np.array(list_f), list_weights)
            self.k_counter += 1
            if self.k_counter >= self.pop_size:
                self.k_counter = 0


class L_SHADE(Optimizer):
    """
    The original version of: Linear Population Size Reduction Success-History Adaptation Differential Evolution (LSHADE)

    Links:
        1. https://metahack.org/CEC2014-Tanabe-Fukunaga.pdf

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + miu_f (float): [0.4, 0.6], initial weighting factor, default = 0.5
        + miu_cr (float): [0.4, 0.6], initial cross-over probability, default = 0.5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.DE import L_SHADE
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> miu_f = 0.5
    >>> miu_cr = 0.5
    >>> model = L_SHADE(problem_dict1, epoch, pop_size, miu_f, miu_cr)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Tanabe, R. and Fukunaga, A.S., 2014, July. Improving the search performance of SHADE using
    linear population size reduction. In 2014 IEEE congress on evolutionary computation (CEC) (pp. 1658-1665). IEEE.
    """

    def __init__(self, problem, epoch=750, pop_size=100, miu_f=0.5, miu_cr=0.5, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_f (float): initial weighting factor, default = 0.5
            miu_cr (float): initial cross-over probability, default = 0.5
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size

        # Dynamic variable
        self.dyn_miu_f = miu_f * np.ones(self.pop_size)  # list the initial f,
        self.dyn_miu_cr = miu_cr * np.ones(self.pop_size)  # list the initial cr,
        self.dyn_pop_archive = list()
        self.dyn_pop_size = self.pop_size
        self.k_counter = 0
        self.n_min = int(self.pop_size / 5)

    ### Survivor Selection
    def weighted_lehmer_mean(self, list_objects, list_weights):
        up = sum(list_weights * list_objects ** 2)
        down = sum(list_weights * list_objects)
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
            top = int(self.dyn_pop_size * p)
            x_best = pop_sorted[np.random.randint(0, top)]
            x_r1 = self.pop[np.random.choice(list(set(range(0, self.dyn_pop_size)) - {idx}))]
            new_pop = self.pop + self.dyn_pop_archive
            while True:
                x_r2 = new_pop[np.random.randint(0, len(new_pop))]
                if np.any(x_r2[self.ID_POS] - x_r1[self.ID_POS]) and np.any(x_r2[self.ID_POS] - self.pop[idx][self.ID_POS]):
                    break
            x_new = self.pop[idx][self.ID_POS] + f * (x_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + f * (x_r1[self.ID_POS] - x_r2[self.ID_POS])
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < cr, x_new, self.pop[idx][self.ID_POS])
            j_rand = np.random.randint(0, self.problem.n_dims)
            pos_new[j_rand] = x_new[j_rand]
            pos_new = self.amend_position(pos_new)
            pop.append([pos_new, None])
        pop = self.update_fitness_population(pop)

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
            total_fit = sum(np.abs(list_fit_new - list_fit_old))
            list_weights = 0 if total_fit == 0 else np.abs(list_fit_new - list_fit_old) / total_fit
            self.dyn_miu_cr[self.k_counter] = sum(list_weights * np.array(list_cr))
            self.dyn_miu_f[self.k_counter] = self.weighted_lehmer_mean(np.array(list_f), list_weights)
            self.k_counter += 1
            if self.k_counter >= self.dyn_pop_size:
                self.k_counter = 0

        # Linear Population Size Reduction
        self.dyn_pop_size = round(self.pop_size + epoch * ((self.n_min - self.pop_size) / self.epoch))


class SAP_DE(Optimizer):
    """
    The original version of: Differential Evolution with Self-Adaptive Populations (SAP_DE)

    Links:
        1. https://doi.org/10.1007/s00500-005-0537-1

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + wf (float): [0.6, 0.95], weighting factor, default = 0.8
        + cr (float): [0.6, 0.95], crossover rate, default = 0.9
        + branch (str): ["ABS" or "REL"], gaussian (absolute) or uniform (relative) method

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.DE import SAP_DE
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> miu_f = 0.5
    >>> miu_cr = 0.5
    >>> model = SAP_DE(problem_dict1, epoch, pop_size, miu_f, miu_cr)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Teo, J., 2006. Exploring dynamic self-adaptive populations in differential evolution. Soft Computing, 10(8), pp.673-686.
    """

    ID_CR = 2
    ID_MR = 3
    ID_PS = 4

    def __init__(self, problem, epoch=750, pop_size=100, wf=0.8, cr=0.9, branch="ABS", **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wf (float): weighting factor, default = 0.8
            cr (float): crossover rate, default = 0.9
            branch (str): gaussian (absolute) or uniform (relative) method
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.weighting_factor = wf
        self.crossover_rate = cr
        self.fixed_pop_size = pop_size
        self.branch = branch  # np.absolute (ABS) or relative (REL)

    def create_solution(self):
        """
        To get the position, fitness wrapper, target and obj list
            + A[self.ID_POS]                  --> Return: position
            + A[self.ID_TAR]                  --> Return: [target, [obj1, obj2, ...]]
            + A[self.ID_TAR][self.ID_FIT]     --> Return: target
            + A[self.ID_TAR][self.ID_OBJ]     --> Return: [obj1, obj2, ...]

        Returns:
            list: wrapper of solution with format [position, [target, [obj1, obj2, ...]], crossover_rate, mutation_rate, pop_size]
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        position = self.amend_position(position)
        fitness = self.get_fitness_position(position=position)
        crossover_rate = np.random.uniform(0, 1)
        mutation_rate = np.random.uniform(0, 1)
        if self.branch == "ABS":
            pop_size = int(10 * self.problem.n_dims + np.random.normal(0, 1))
        else:  # elif self.branch == "REL":
            pop_size = int(10 * self.problem.n_dims + np.random.uniform(-0.5, 0.5))
        return [position, fitness, crossover_rate, mutation_rate, pop_size]

    def edit_to_range(self, var=None, lower=0, upper=1, func_value=None):
        while var <= lower or var >= upper:
            if var <= lower:
                var += func_value()
            if var >= upper:
                var -= func_value()
        return var

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = []
        for idx in range(0, self.pop_size):
            # Choose 3 random element and different to idx
            idxs = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
            j = np.random.randint(0, self.pop_size)
            self.F = np.random.uniform(0, 1)

            ## Crossover
            if np.random.uniform(0, 1) < self.pop[idx][self.ID_CR] or idx == j:
                pos_new = self.pop[idxs[0]][self.ID_POS] + self.F * (self.pop[idxs[1]][self.ID_POS] - self.pop[idxs[2]][self.ID_POS])
                cr_new = self.pop[idxs[0]][self.ID_CR] + self.F * (self.pop[idxs[1]][self.ID_CR] - self.pop[idxs[2]][self.ID_CR])
                mr_new = self.pop[idxs[0]][self.ID_MR] + self.F * (self.pop[idxs[1]][self.ID_MR] - self.pop[idxs[2]][self.ID_MR])
                if self.branch == "ABS":
                    ps_new = self.pop[idxs[0]][self.ID_PS] + int(self.F * (self.pop[idxs[1]][self.ID_PS] - self.pop[idxs[2]][self.ID_PS]))
                else:  # elif self.branch == "REL":
                    ps_new = self.pop[idxs[0]][self.ID_PS] + self.F * (self.pop[idxs[1]][self.ID_PS] - self.pop[idxs[2]][self.ID_PS])
                pos_new = self.amend_position(pos_new)
                cr_new = self.edit_to_range(cr_new, 0, 1, np.random.random)
                mr_new = self.edit_to_range(mr_new, 0, 1, np.random.random)
                pop.append([pos_new, None, cr_new, mr_new, ps_new])
            else:
                pop.append(deepcopy(self.pop[idx]))
            ## Mutation
            if np.random.uniform(0, 1) < self.pop[idxs[0]][self.ID_MR]:
                pos_new = self.pop[idx][self.ID_POS] + np.random.normal(0, self.pop[idxs[0]][self.ID_MR])
                cr_new = np.random.normal(0, 1)
                mr_new = np.random.normal(0, 1)
                if self.branch == "ABS":
                    ps_new = self.pop[idx][self.ID_PS] + int(np.random.normal(0.5, 1))
                else:  # elif self.branch == "REL":
                    ps_new = self.pop[idx][self.ID_PS] + np.random.normal(0, self.pop[idxs[0]][self.ID_MR])
                pos_new = self.amend_position(pos_new)
                pop.append([pos_new, None, cr_new, mr_new, ps_new])
        pop = self.update_fitness_population(pop)

        # Calculate new population size
        total = sum([pop[i][self.ID_PS] for i in range(0, self.pop_size)])
        if self.branch == "ABS":
            m_new = int(total / self.pop_size)
        else:  # elif self.branch == "REL":
            m_new = int(self.pop_size + total)
        if m_new <= 4:
            m_new = self.fixed_pop_size + int(np.random.uniform(0, 4))
        elif m_new > 4 * self.fixed_pop_size:
            m_new = self.fixed_pop_size - int(np.random.uniform(0, 4))

        ## Change population by population size
        if m_new <= self.pop_size:
            self.pop = pop[:m_new]
        else:
            pop_sorted = self.get_sorted_strim_population(pop)
            self.pop = pop + pop_sorted[:m_new - self.pop_size]
        self.pop_size = len(self.pop)
