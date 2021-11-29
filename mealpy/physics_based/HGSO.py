#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 07:03, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseHGSO(Optimizer):
    """
        The original version of: Henry Gas Solubility Optimization (HGSO)
            Henry gas solubility optimization: A novel physics-based algorithm
        Link:
            https://www.sciencedirect.com/science/article/abs/pii/S0167739X19306557
    """

    def __init__(self, problem, epoch=10000, pop_size=100, n_clusters=2, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_clusters (int): number of clusters, default = 2
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.n_clusters = n_clusters
        self.n_elements = int(self.pop_size / self.n_clusters)

        self.T0 = 298.15
        self.K = 1.0
        self.beta = 1.0
        self.alpha = 1
        self.epxilon = 0.05

        self.l1 = 5E-2
        self.l2 = 100.0
        self.l3 = 1E-2
        self.H_j = self.l1 * np.random.uniform()
        self.P_ij = self.l2 * np.random.uniform()
        self.C_j = self.l3 * np.random.uniform()

        self.pop_group, self.p_best = None, None

    def _create_group(self, pop):
        pop_group = []
        for idx in range(0, self.n_clusters):
            pop_group.append(pop[idx*self.n_elements:(idx+1)*self.n_elements])
        return pop_group

    def _flatten_group(self, group):
        pop = []
        for idx in range(0, self.n_clusters):
            pop += group[idx]
        return pop

    def initialization(self):
        self.pop = self.create_population(self.pop_size)
        _, self.g_best = self.get_global_best_solution(self.pop)
        self.pop_group = self._create_group(self.pop)
        self.p_best = self._get_best_solution_in_team(self.pop_group)  # multiple element

    def _get_best_solution_in_team(self, group=None):
        list_best = []
        for i in range(len(group)):
            _, best_agent = self.get_global_best_solution(group[i])
            list_best.append(best_agent)
        return list_best

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        ## Loop based on the number of cluster in swarm (number of gases type)
        for i in range(self.n_clusters):
            ### Loop based on the number of individual in each gases type
            pop_new = []
            for j in range(self.n_elements):
                F = -1.0 if np.random.uniform() < 0.5 else 1.0

                ##### Based on Eq. 8, 9, 10
                self.H_j = self.H_j * np.exp(-self.C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / self.T0))
                S_ij = self.K * self.H_j * self.P_ij
                gama = self.beta * np.exp(- ((self.p_best[i][self.ID_FIT][self.ID_TAR] + self.epxilon) /
                                             (self.pop_group[i][j][self.ID_FIT][self.ID_TAR] + self.epxilon)))
                X_ij = self.pop_group[i][j][self.ID_POS] + F * np.random.uniform() * gama * \
                       (self.p_best[i][self.ID_POS] - self.pop_group[i][j][self.ID_POS]) + \
                       F * np.random.uniform() * self.alpha * (S_ij * self.g_best[self.ID_POS] - self.pop_group[i][j][self.ID_POS])
                pos_new = self.amend_position_faster(X_ij)
                pop_new.append([pos_new, None])
                nfe_epoch += 1
            self.pop_group[i] = self.update_fitness_population(pop_new)
        self.pop = self._flatten_group(self.pop_group)

        ## Update Henry's coefficient using Eq.8
        self.H_j = self.H_j * np.exp(-self.C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / self.T0))
        ## Update the solubility of each gas using Eq.9
        S_ij = self.K * self.H_j * self.P_ij
        ## Rank and select the number of worst agents using Eq. 11
        N_w = int(self.pop_size * (np.random.uniform(0, 0.1) + 0.1))
        ## Update the position of the worst agents using Eq. 12
        sorted_id_pos = np.argsort([x[self.ID_FIT][self.ID_TAR] for x in self.pop])

        pop_new = []
        pop_idx = []
        for item in range(N_w):
            id = sorted_id_pos[item]
            X_new = np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.amend_position_faster(X_new)
            pop_new.append([pos_new, None])
            pop_idx.append(id)
            nfe_epoch += 1
        pop_new = self.update_fitness_population(pop_new)
        for idx, id_selected in enumerate(pop_idx):
            self.pop[id_selected] = deepcopy(pop_new[idx])
        self.pop_group = self._create_group(self.pop)
        self.p_best = self._get_best_solution_in_team(self.pop_group)
        self.nfe_per_epoch = nfe_epoch
