#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 07:03, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
import time
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
        self.nfe_per_epoch = 1.5 * pop_size

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

    def _create_population__(self, n_clusters=2):
        pop = []
        group = []
        for i in range(n_clusters):
            team = []
            for j in range(self.n_elements):
                pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
                fit_new = self.get_fitness_position(pos_new)
                team.append([pos_new.copy(), fit_new, i])
                pop.append([pos_new.copy(), fit_new, i])
            group.append(team)
        return pop, group

    def _get_best_solution_in_team(self, group=None):
        list_best = []
        for i in range(len(group)):
            _, best_agent = self.get_global_best_solution(group[i])
            list_best.append(best_agent)
        return list_best

    def solve(self, mode='sequential'):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        if mode != "sequential":
            print("HGSO is support sequential mode only!")
            exit(0)
        self.termination_start()
        pop, group = self._create_population__(self.n_clusters)
        _, g_best = self.get_global_best_solution(pop)
        self.history.save_initial_best(g_best)
        p_best = self._get_best_solution_in_team(group)  # multiple element

        for epoch in range(0, self.epoch):
            time_epoch = time.time()

            ## Loop based on the number of cluster in swarm (number of gases type)
            for i in range(self.n_clusters):

                ### Loop based on the number of individual in each gases type
                for j in range(self.n_elements):
                    F = -1.0 if np.random.uniform() < 0.5 else 1.0

                    ##### Based on Eq. 8, 9, 10
                    self.H_j = self.H_j * np.exp(-self.C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / self.T0))
                    S_ij = self.K * self.H_j * self.P_ij
                    gama = self.beta * np.exp(- ((p_best[i][self.ID_FIT][self.ID_TAR] + self.epxilon) /
                                                 (group[i][j][self.ID_FIT][self.ID_TAR] + self.epxilon)))

                    X_ij = group[i][j][self.ID_POS] + F * np.random.uniform() * gama * (p_best[i][self.ID_POS] - group[i][j][self.ID_POS]) + \
                           F * np.random.uniform() * self.alpha * (S_ij * g_best[self.ID_POS] - group[i][j][self.ID_POS])
                    pos_new = self.amend_position_faster(X_ij)
                    fit_new = self.get_fitness_position(pos_new)
                    group[i][j] = [pos_new, fit_new, i]
                    pop[i * self.n_elements + j] = [pos_new, fit_new, i]

            ## Update Henry's coefficient using Eq.8
            self.H_j = self.H_j * np.exp(-self.C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / self.T0))
            ## Update the solubility of each gas using Eq.9
            S_ij = self.K * self.H_j * self.P_ij
            ## Rank and select the number of worst agents using Eq. 11
            N_w = int(self.pop_size * (np.random.uniform(0, 0.1) + 0.1))
            ## Update the position of the worst agents using Eq. 12
            sorted_id_pos = np.argsort([x[self.ID_FIT][self.ID_TAR] for x in pop])

            for item in range(N_w):
                id = sorted_id_pos[item]
                j = id % self.n_elements
                i = int((id - j) / self.n_elements)
                X_new = np.random.uniform(self.problem.lb, self.problem.ub)
                pos_new = self.amend_position_faster(X_new)
                fit_new = self.get_fitness_position(pos_new)
                pop[id] = [pos_new, fit_new, i]
                group[i][j] = [pos_new, fit_new, i]

            p_best = self._get_best_solution_in_team(group)
            # update global best position
            _, g_best = self.update_global_best_solution(pop)  # We sort the population

            ## Additional information for the framework
            time_epoch = time.time() - time_epoch
            self.history.list_epoch_time.append(time_epoch)
            self.history.list_population.append(pop.copy())
            self.print_epoch(epoch + 1, time_epoch)
            if self.termination_flag:
                if self.termination.mode == 'TB':
                    if time.time() - self.count_terminate >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                elif self.termination.mode == 'FE':
                    self.count_terminate += self.nfe_per_epoch
                    if self.count_terminate >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                elif self.termination.mode == 'MG':
                    if epoch >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                else:  # Early Stopping
                    temp = self.count_terminate + self.history.get_global_repeated_times(self.ID_FIT, self.ID_TAR, self.EPSILON)
                    if temp >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break

        ## Additional information for the framework
        self.save_optimization_process()
        return self.solution[self.ID_POS], self.solution[self.ID_FIT][self.ID_TAR]
