#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalHGSO(Optimizer):
    """
    The original version of: Henry Gas Solubility Optimization (HGSO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0167739X19306557

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_clusters (int): [2, 10], number of clusters, default = 2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.HGSO import OriginalHGSO
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
    >>> n_clusters = 3
    >>> model = OriginalHGSO(epoch, pop_size, n_clusters)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Hashim, F.A., Houssein, E.H., Mabrouk, M.S., Al-Atabany, W. and Mirjalili, S., 2019. Henry gas solubility
    optimization: A novel physics-based algorithm. Future Generation Computer Systems, 101, pp.646-667.
    """

    def __init__(self, epoch=10000, pop_size=100, n_clusters=2, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_clusters (int): number of clusters, default = 2
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_clusters = self.validator.check_int("n_clusters", n_clusters, [2, int(self.pop_size/5)])
        self.set_parameters(["epoch", "pop_size", "n_clusters"])
        self.n_elements = int(self.pop_size / self.n_clusters)
        self.sort_flag = False
        self.T0 = 298.15
        self.K = 1.0
        self.beta = 1.0
        self.alpha = 1
        self.epxilon = 0.05
        self.l1 = 5E-2
        self.l2 = 100.0
        self.l3 = 1E-2

    def initialize_variables(self):
        self.H_j = self.l1 * np.random.uniform()
        self.P_ij = self.l2 * np.random.uniform()
        self.C_j = self.l3 * np.random.uniform()
        self.pop_group, self.p_best = None, None

    def initialization(self):
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)
        self.pop_group = self.create_pop_group(self.pop, self.n_clusters, self.n_elements)
        self.p_best = self.get_best_solution_in_team__(self.pop_group)  # multiple element

    def flatten_group__(self, group):
        pop = []
        for idx in range(0, self.n_clusters):
            pop += group[idx]
        return pop

    def get_best_solution_in_team__(self, group=None):
        list_best = []
        for i in range(len(group)):
            _, best_agent = self.get_global_best_solution(group[i])
            list_best.append(best_agent)
        return list_best

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Loop based on the number of cluster in swarm (number of gases type)
        for i in range(self.n_clusters):
            ### Loop based on the number of individual in each gases type
            pop_new = []
            for j in range(self.n_elements):
                F = -1.0 if np.random.uniform() < 0.5 else 1.0

                ##### Based on Eq. 8, 9, 10
                self.H_j = self.H_j * np.exp(-self.C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / self.T0))
                S_ij = self.K * self.H_j * self.P_ij
                gama = self.beta * np.exp(- ((self.p_best[i][self.ID_TAR][self.ID_FIT] + self.epxilon) /
                                             (self.pop_group[i][j][self.ID_TAR][self.ID_FIT] + self.epxilon)))
                X_ij = self.pop_group[i][j][self.ID_POS] + F * np.random.uniform() * gama * \
                       (self.p_best[i][self.ID_POS] - self.pop_group[i][j][self.ID_POS]) + \
                       F * np.random.uniform() * self.alpha * (S_ij * self.g_best[self.ID_POS] - self.pop_group[i][j][self.ID_POS])
                pos_new = self.amend_position(X_ij, self.problem.lb, self.problem.ub)
                pop_new.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop_group[i] = pop_new
        self.pop = self.flatten_group__(self.pop_group)

        ## Update Henry's coefficient using Eq.8
        self.H_j = self.H_j * np.exp(-self.C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / self.T0))
        ## Update the solubility of each gas using Eq.9
        S_ij = self.K * self.H_j * self.P_ij
        ## Rank and select the number of worst agents using Eq. 11
        N_w = int(self.pop_size * (np.random.uniform(0, 0.1) + 0.1))
        ## Update the position of the worst agents using Eq. 12
        sorted_id_pos = np.argsort([x[self.ID_TAR][self.ID_FIT] for x in self.pop])

        pop_new = []
        pop_idx = []
        for item in range(N_w):
            id = sorted_id_pos[item]
            X_new = np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.amend_position(X_new, self.problem.lb, self.problem.ub)
            pop_idx.append(id)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)
        for idx, id_selected in enumerate(pop_idx):
            self.pop[id_selected] = deepcopy(pop_new[idx])
        self.pop_group = self.create_pop_group(self.pop, self.n_clusters, self.n_elements)
        self.p_best = self.get_best_solution_in_team__(self.pop_group)
