#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 07:03, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import argsort, exp
from numpy.random import uniform
from copy import deepcopy
from mealpy.root import Root


class BaseHGSO(Root):
    """
        The original version of: Henry Gas Solubility Optimization (HGSO)
            Henry gas solubility optimization: A novel physics-based algorithm
        Link:
            https://www.sciencedirect.com/science/article/abs/pii/S0167739X19306557
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, n_clusters=2, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
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
        self.H_j = self.l1 * uniform()
        self.P_ij = self.l2 * uniform()
        self.C_j = self.l3 * uniform()

    def _create_population__(self, minmax=0, n_clusters=0):
        pop = []
        group = []
        for i in range(n_clusters):
            team = []
            for j in range(self.n_elements):
                solution = uniform(self.lb, self.ub)
                fitness = self.get_fitness_position(position=solution, minmax=minmax)
                team.append([solution, fitness, i])
                pop.append([solution, fitness, i])
            group.append(team)
        return pop, group

    def _get_best_solution_in_team(self, group=None):
        list_best = []
        for i in range(len(group)):
            sorted_team = sorted(group[i], key=lambda temp: temp[self.ID_FIT])
            list_best.append(deepcopy(sorted_team[self.ID_MIN_PROB]))
        return list_best

    def train(self):
        pop, group = self._create_population__(self.ID_MIN_PROB, self.n_clusters)
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)      # single element
        p_best = self._get_best_solution_in_team(group)                                 # multiple element

        # Loop iterations
        for epoch in range(self.epoch):

            ## Loop based on the number of cluster in swarm (number of gases type)
            for i in range(self.n_clusters):

                ### Loop based on the number of individual in each gases type
                for j in range( self.n_elements):

                    F = -1.0 if uniform() < 0.5 else 1.0

                    ##### Based on Eq. 8, 9, 10
                    self.H_j = self.H_j * exp(-self.C_j * ( 1.0/exp(-epoch/self.epoch) - 1.0/self.T0 ))
                    S_ij = self.K * self.H_j * self.P_ij
                    gama = self.beta * exp(- ((p_best[i][self.ID_FIT] + self.epxilon) / (group[i][j][self.ID_FIT] + self.epxilon)))

                    X_ij = group[i][j][self.ID_POS] + F * uniform() * gama * (p_best[i][self.ID_POS] - group[i][j][self.ID_POS]) + \
                        F * uniform() * self.alpha * (S_ij * g_best[self.ID_POS] - group[i][j][self.ID_POS])

                    fit = self.get_fitness_position(X_ij, self.ID_MIN_PROB)
                    group[i][j] = [X_ij, fit, i]
                    pop[i*self.n_elements + j] = [X_ij, fit, i]

            ## Update Henry's coefficient using Eq.8
            self.H_j = self.H_j * exp(-self.C_j * (1.0 / exp(-epoch / self.epoch) - 1.0 / self.T0))
            ## Update the solubility of each gas using Eq.9
            S_ij = self.K * self.H_j * self.P_ij
            ## Rank and select the number of worst agents using Eq. 11
            N_w = int(self.pop_size * (uniform(0, 0.1) + 0.1))
            ## Update the position of the worst agents using Eq. 12
            sorted_id_pos = argsort([ x[self.ID_FIT] for x in pop ])

            for item in range(N_w):
                id = sorted_id_pos[item]
                j = id % self.n_elements
                i = int((id-j) / self.n_elements)
                X_new = uniform(self.lb, self.ub)
                fit = self.get_fitness_position(X_new, self.ID_MIN_PROB)
                pop[id] = [X_new, fit, i]
                group[i][j] = [X_new, fit, i]

            p_best = self._get_best_solution_in_team(group)
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OppoHGSO(BaseHGSO):

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, n_clusters=2, **kwargs):
        BaseHGSO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, n_clusters, kwargs = kwargs)

    def train(self):
        pop, group = self._create_population__(self.ID_MIN_PROB, self.n_clusters)
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)      # single element
        p_best = self._get_best_solution_in_team(group)                                 # multiple element

        # Loop iterations
        for epoch in range(self.epoch):

            ## Loop based on the number of cluster in swarm (number of gases type)
            for i in range(self.n_clusters):

                ### Loop based on the number of individual in each gases type
                for j in range( self.n_elements):

                    ##### Based on Eq. 8, 9, 10
                    self.H_j = self.H_j * exp(-self.C_j * (1.0 / exp(-epoch / self.epoch) - 1.0 / self.T0))
                    self.S_ij = self.K * self.H_j * self.P_ij
                    F = -1.0 if uniform() < 0.5 else 1.0
                    gama = self.beta * exp(- ((p_best[i][self.ID_FIT] + self.epxilon) / (group[i][j][self.ID_FIT] + self.epxilon)))

                    X_ij = group[i][j][self.ID_POS] + F * uniform() * gama * (p_best[i][self.ID_POS] - group[i][j][self.ID_POS]) + \
                        F * uniform() * self.alpha * (self.S_ij * g_best[self.ID_POS] - group[i][j][self.ID_POS])
                    X_ij = self.amend_position_faster(X_ij)

                    fit = self.get_fitness_position(X_ij, self.ID_MIN_PROB)
                    group[i][j] = [X_ij, fit, i]
                    pop[i*self.n_elements + j] = [X_ij, fit, i]

            ## Rank and select the number of worst agents using Eq. 11
            N_w = int(self.pop_size * (uniform(0, 0.1) + 0.1))
            ## Update the position of the worst agents using Eq. 12
            sorted_id_pos = argsort([ x[self.ID_FIT] for x in pop ])

            for item in range(N_w):
                id = sorted_id_pos[item]
                j = id % self.n_elements
                i = int((id-j) / self.n_elements)
                X_new = uniform(self.lb, self.ub)
                fit = self.get_fitness_position(X_new, self.ID_MIN_PROB)
                if fit < pop[id][self.ID_FIT]:
                    pop[id] = [X_new, fit, i]
                    group[i][j] = [X_new, fit, i]
                else:
                    C_op = self.create_opposition_position(pop[i][self.ID_POS], g_best[self.ID_POS])
                    C_op = self.amend_position_faster(C_op)
                    fit_op = self.get_fitness_position(C_op, self.ID_MIN_PROB)
                    if fit_op < pop[id][self.ID_FIT]:
                        pop[id] = [X_new, fit, i]
                        group[i][j] = [X_new, fit, i]

            p_best = self._get_best_solution_in_team(group)
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevyHGSO(BaseHGSO):

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, n_clusters=2, **kwargs):
        BaseHGSO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, n_clusters, kwargs=kwargs)

    def train(self):
        pop, group = self._create_population__(self.ID_MIN_PROB, self.n_clusters)
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)        # single element
        p_best = self._get_best_solution_in_team(group)                             # multiple element

        # Loop iterations
        for epoch in range(self.epoch):

            ## Loop based on the number of cluster in swarm (number of gases type)
            for i in range(self.n_clusters):

                ### Loop based on the number of individual in each gases type
                for j in range( self.n_elements):

                    ##### Based on Levy
                    if uniform() < 0.5:
                        X_ij = self.levy_flight(epoch, group[i][j][self.ID_POS], g_best[self.ID_POS], step=0.001, case=1)
                    else:   ##### Based on Eq. 8, 9, 10
                        self.H_j = self.H_j * exp(-self.C_j * (1.0 / exp(-epoch / self.epoch) - 1.0 / self.T0))
                        self.S_ij = self.K * self.H_j * self.P_ij
                        F = -1.0 if uniform() < 0.5 else 1.0
                        gama = self.beta * exp(- ((p_best[i][self.ID_FIT] + self.epxilon) / (group[i][j][self.ID_FIT] + self.epxilon)))

                        X_ij = group[i][j][self.ID_POS] + F * uniform() * gama * (p_best[i][self.ID_POS] - group[i][j][self.ID_POS]) + \
                            F * uniform() * self.alpha * (self.S_ij * g_best[self.ID_POS] - group[i][j][self.ID_POS])

                    X_ij = self.amend_position_faster(X_ij)
                    fit = self.get_fitness_position(X_ij, self.ID_MIN_PROB)
                    group[i][j] = [X_ij, fit, i]
                    pop[i*self.n_elements + j] = [X_ij, fit, i]

            ## Rank and select the number of worst agents using Eq. 11
            N_w = int(self.pop_size * (uniform(0, 0.1) + 0.1))
            ## Update the position of the worst agents using Eq. 12
            sorted_id_pos = argsort([item[self.ID_FIT] for item in pop])

            for item in range(N_w):
                id = sorted_id_pos[item]
                j = id % self.n_elements
                i = int((id-j) / self.n_elements)
                X_new = uniform(self.lb, self.ub)
                fit = self.get_fitness_position(X_new, self.ID_MIN_PROB)
                if fit < pop[id][self.ID_FIT]:
                    pop[id] = [X_new, fit, i]
                    group[i][j] = [X_new, fit, i]

            p_best = self._get_best_solution_in_team(group)
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
