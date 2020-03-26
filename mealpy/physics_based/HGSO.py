#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 07:03, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import argsort, exp, sin, pi, abs, sqrt, sign, power, ones
from numpy.random import uniform, normal
from copy import deepcopy
from math import gamma
from mealpy.root import Root


class BaseHGSO(Root):
    """
    Henry gas solubility optimization: A novel physics-based algorithm
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, n_clusters=2):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
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
                solution = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                fitness = self._fitness_model__(solution=solution, minmax=minmax)
                team.append([solution, fitness, i])
                pop.append([solution, fitness, i])
            group.append(team)
        return pop, group

    def _get_best_solution_in_team(self, group=None):
        list_best = []
        for i in range(len(group)):
            sorted_team = sorted(group[i], key=lambda temp: temp[self.ID_FIT])
            list_best.append( deepcopy(sorted_team[self.ID_MIN_PROB]) )
        return list_best

    def _train__(self):
        pop, group = self._create_population__(self.ID_MIN_PROB, self.n_clusters)
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)    # single element
        p_best = self._get_best_solution_in_team(group)                         # multiple element

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

                    fit = self._fitness_model__(X_ij, self.ID_MIN_PROB)
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
                X_new = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                fit = self._fitness_model__(solution=X_new, minmax=self.ID_MIN_PROB)
                pop[id] = [X_new, fit, i]
                group[i][j] = [X_new, fit, i]

            p_best = self._get_best_solution_in_team(group)
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OppoHGSO(BaseHGSO):
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, n_clusters=2):
        BaseHGSO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size, n_clusters)

    def _train__(self):
        pop, group = self._create_population__(self.ID_MIN_PROB, self.n_clusters)
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)    # single element
        p_best = self._get_best_solution_in_team(group)                         # multiple element

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
                    X_ij = self._amend_solution_faster__(X_ij)

                    fit = self._fitness_model__(X_ij, self.ID_MIN_PROB)
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
                X_new = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                fit = self._fitness_model__(solution=X_new, minmax=self.ID_MIN_PROB)
                if fit < pop[id][self.ID_FIT]:
                    pop[id] = [X_new, fit, i]
                    group[i][j] = [X_new, fit, i]
                else:
                    t1 = self.domain_range[1] * ones(self.problem_size) + self.domain_range[0] * ones(self.problem_size)
                    t2 = -1 * g_best[self.ID_POS] + uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                    C_op = t1 + t2
                    C_op = self._amend_solution_faster__(C_op)
                    fit_op = self._fitness_model__(C_op, self.ID_MIN_PROB)
                    if fit_op < pop[id][self.ID_FIT]:
                        pop[id] = [X_new, fit, i]
                        group[i][j] = [X_new, fit, i]

            p_best = self._get_best_solution_in_team(group)
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevyHGSO(BaseHGSO):
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, n_clusters=2):
        BaseHGSO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size, n_clusters)

    def _levy_flight__(self, epoch, solution, prey):
        beta = 1
        # muy and v are two random variables which follow normal distribution
        # sigma_muy : standard deviation of muy
        sigma_muy = power(gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = normal(0, sigma_muy)
        v = normal(0, sigma_v)
        s = muy / power(abs(v), 1 / beta)
        # D is a random solution
        D = self._create_solution__(minmax=self.ID_MAX_PROB)
        LB = 0.001 * s * (solution[self.ID_POS] - prey[self.ID_POS])

        levy = D[self.ID_POS] * LB
        #return levy

        x_new = solution[0] + 1.0/sqrt(epoch+1) * sign(uniform() - 0.5) * levy
        return x_new

    def _levy_flight_2__(self, solution=None, g_best=None):
        alpha = 0.01
        xichma_v = 1
        xichma_u = ((gamma(1 + 1.5) * sin(pi * 1.5 / 2)) / (gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))) ** (1.0 / 1.5)
        levy_b = (normal(0, xichma_u ** 2)) / (sqrt(normal(0, xichma_v ** 2)) ** (1.0 / 1.5))
        return solution[self.ID_POS] + alpha * levy_b * (solution[self.ID_POS] - g_best[self.ID_POS])

    def _train__(self):
        pop, group = self._create_population__(self.ID_MIN_PROB, self.n_clusters)
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)        # single element
        p_best = self._get_best_solution_in_team(group)                             # multiple element

        # Loop iterations
        for epoch in range(self.epoch):

            ## Loop based on the number of cluster in swarm (number of gases type)
            for i in range(self.n_clusters):

                ### Loop based on the number of individual in each gases type
                for j in range( self.n_elements):

                    ##### Based on Levy
                    if uniform() < 0.5:
                        X_ij = self._levy_flight__(epoch+1, group[i][j], g_best)
                        #X_ij = self._levy_flight_2__(group[i][j], g_best)
                    else:   ##### Based on Eq. 8, 9, 10
                        self.H_j = self.H_j * exp(-self.C_j * (1.0 / exp(-epoch / self.epoch) - 1.0 / self.T0))
                        self.S_ij = self.K * self.H_j * self.P_ij
                        F = -1.0 if uniform() < 0.5 else 1.0
                        gama = self.beta * exp(- ((p_best[i][self.ID_FIT] + self.epxilon) / (group[i][j][self.ID_FIT] + self.epxilon)))

                        X_ij = group[i][j][self.ID_POS] + F * uniform() * gama * (p_best[i][self.ID_POS] - group[i][j][self.ID_POS]) + \
                            F * uniform() * self.alpha * (self.S_ij * g_best[self.ID_POS] - group[i][j][self.ID_POS])

                    X_ij = self._amend_solution_faster__(X_ij)
                    fit = self._fitness_model__(X_ij, self.ID_MIN_PROB)
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
                X_new = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                fit = self._fitness_model__(solution=X_new, minmax=self.ID_MIN_PROB)
                if fit < pop[id][self.ID_FIT]:
                    pop[id] = [X_new, fit, i]
                    group[i][j] = [X_new, fit, i]

            p_best = self._get_best_solution_in_team(group)
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
