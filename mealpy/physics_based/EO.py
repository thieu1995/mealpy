#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalEO(Optimizer):
    """
    The original version of: Equilibrium Optimizer (EO)

    Links:
        1. https://doi.org/10.1016/j.knosys.2019.105190
        2. https://www.mathworks.com/matlabcentral/fileexchange/73352-equilibrium-optimizer-eo

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.EO import OriginalEO
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
    >>> model = OriginalEO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Faramarzi, A., Heidarinejad, M., Stephens, B. and Mirjalili, S., 2020. Equilibrium optimizer: A novel
    optimization algorithm. Knowledge-Based Systems, 191, p.105190.
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
        self.sort_flag = False
        ## Fixed parameter proposed by authors
        self.V = 1
        self.a1 = 2
        self.a2 = 1
        self.GP = 0.5

    def make_equilibrium_pool__(self, list_equilibrium=None):
        pos_list = [item[self.ID_POS] for item in list_equilibrium]
        pos_mean = np.mean(pos_list, axis=0)
        pos_mean = self.amend_position(pos_mean, self.problem.lb, self.problem.ub)
        target = self.get_target_wrapper(pos_mean)
        list_equilibrium.append([pos_mean, target])
        return list_equilibrium

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # ---------------- Memory saving-------------------  make equilibrium pool
        _, c_eq_list, _ = self.get_special_solutions(self.pop, best=4)
        c_pool = self.make_equilibrium_pool__(c_eq_list)
        # Eq. 9
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            lamda = np.random.uniform(0, 1, self.problem.n_dims)  # lambda in Eq. 11
            r = np.random.uniform(0, 1, self.problem.n_dims)  # r in Eq. 11
            c_eq = c_pool[np.random.randint(0, len(c_pool))][self.ID_POS]  # random selection 1 of candidate from the pool
            f = self.a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1.0)  # Eq. 11
            r1 = np.random.uniform()
            r2 = np.random.uniform()  # r1, r2 in Eq. 15
            gcp = 0.5 * r1 * np.ones(self.problem.n_dims) * (r2 >= self.GP)  # Eq. 15
            g0 = gcp * (c_eq - lamda * self.pop[idx][self.ID_POS])  # Eq. 14
            g = g0 * f  # Eq. 13
            pos_new = c_eq + (self.pop[idx][self.ID_POS] - c_eq) * f + (g * self.V / lamda) * (1.0 - f)  # Eq. 16
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)


class ModifiedEO(OriginalEO):
    """
    The original version of: Modified Equilibrium Optimizer (MEO)

    Links:
        1. https://doi.org/10.1016/j.asoc.2020.106542

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.EO import ModifiedEO
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
    >>> model = ModifiedEO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Gupta, S., Deep, K. and Mirjalili, S., 2020. An efficient equilibrium optimizer with mutation
    strategy for numerical optimization. Applied Soft Computing, 96, p.106542.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.sort_flag = False
        self.pop_len = int(self.pop_size / 3)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # ---------------- Memory saving-------------------  make equilibrium pool
        _, c_eq_list, _ = self.get_special_solutions(self.pop, best=4)
        c_pool = self.make_equilibrium_pool__(c_eq_list)

        # Eq. 9
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)

        pop_new = []
        for idx in range(0, self.pop_size):
            lamda = np.random.uniform(0, 1, self.problem.n_dims)  # lambda in Eq. 11
            r = np.random.uniform(0, 1, self.problem.n_dims)  # r in Eq. 11
            c_eq = c_pool[np.random.randint(0, len(c_pool))][self.ID_POS]  # random selection 1 of candidate from the pool
            f = self.a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1.0)  # Eq. 11
            r1 = np.random.uniform()
            r2 = np.random.uniform()  # r1, r2 in Eq. 15
            gcp = 0.5 * r1 * np.ones(self.problem.n_dims) * (r2 >= self.GP)  # Eq. 15
            g0 = gcp * (c_eq - lamda * self.pop[idx][self.ID_POS])  # Eq. 14
            g = g0 * f  # Eq. 13
            pos_new = c_eq + (self.pop[idx][self.ID_POS] - c_eq) * f + (g * self.V / lamda) * (1.0 - f)  # Eq. 16
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        ## Sort the updated population based on fitness
        _, pop_s1, _ = self.get_special_solutions(self.pop, best=self.pop_len)

        ## Mutation scheme
        pop_s2 = pop_s1.copy()
        pop_s2_new = []
        for i in range(0, self.pop_len):
            pos_new = pop_s2[i][self.ID_POS] * (1 + np.random.normal(0, 1, self.problem.n_dims))  # Eq. 12
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_s2_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                pop_s2[i] = self.get_better_solution([pos_new, target], pop_s2[i])
        if self.mode in self.AVAILABLE_MODES:
            pop_s2_new = self.update_target_wrapper_population(pop_s2_new)
            pop_s2 = self.greedy_selection_population(pop_s2_new, pop_s2)

        ## Search Mechanism
        pos_s1_list = [item[self.ID_POS] for item in pop_s1]
        pos_s1_mean = np.mean(pos_s1_list, axis=0)
        pop_s3 = []
        for i in range(0, self.pop_len):
            pos_new = (c_pool[0][self.ID_POS] - pos_s1_mean) - np.random.random() * \
                      (self.problem.lb + np.random.random() * (self.problem.ub - self.problem.lb))
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_s3.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_s3[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_s3 = self.update_target_wrapper_population(pop_s3)
        ## Construct a new population
        self.pop = pop_s1 + pop_s2 + pop_s3
        n_left = self.pop_size - len(self.pop)
        idx_selected = np.random.choice(range(0, len(c_pool)), n_left, replace=False)
        for i in range(0, n_left):
            self.pop.append(c_pool[idx_selected[i]])


class AdaptiveEO(OriginalEO):
    """
    The original version of: Adaptive Equilibrium Optimization (AEO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2020.103836

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.EO import AdaptiveEO
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
    >>> model = AdaptiveEO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Wunnava, A., Naik, M.K., Panda, R., Jena, B. and Abraham, A., 2020. A novel interdependence based
    multilevel thresholding technique using adaptive equilibrium optimizer. Engineering Applications of
    Artificial Intelligence, 94, p.103836.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.sort_flag = False
        self.pop_len = int(self.pop_size / 3)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # ---------------- Memory saving-------------------  make equilibrium pool
        _, c_eq_list, _ = self.get_special_solutions(self.pop, best=4)
        c_pool = self.make_equilibrium_pool__(c_eq_list)
        # Eq. 9
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)
        ## Memory saving, Eq 20, 21
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            lamda = np.random.uniform(0, 1, self.problem.n_dims)
            r = np.random.uniform(0, 1, self.problem.n_dims)
            c_eq = c_pool[np.random.randint(0, len(c_pool))][self.ID_POS]  # random selection 1 of candidate from the pool
            f = self.a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1.0)  # Eq. 14
            r1 = np.random.uniform()
            r2 = np.random.uniform()
            gcp = 0.5 * r1 * np.ones(self.problem.n_dims) * (r2 >= self.GP)
            g0 = gcp * (c_eq - lamda * self.pop[idx][self.ID_POS])
            g = g0 * f
            fit_average = np.mean([item[self.ID_TAR][self.ID_FIT] for item in self.pop])  # Eq. 19
            pos_new = c_eq + (self.pop[idx][self.ID_POS] - c_eq) * f + (g * self.V / lamda) * (1.0 - f)  # Eq. 9
            if self.pop[idx][self.ID_TAR][self.ID_FIT] >= fit_average:
                pos_new = np.multiply(pos_new, (0.5 + np.random.uniform(0, 1, self.problem.n_dims)))
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
