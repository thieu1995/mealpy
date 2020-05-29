#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:49, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, normal, randint
from numpy import pi, sin, cos, zeros, minimum, maximum, abs, where, sign
from copy import deepcopy
from mealpy.root import Root


class BasePSO(Root):
    """
    Particle Swarm Optimization
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, c1=1.2, c2=1.2, w_min=0.4, w_max=0.9):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c1 = c1            # [0-2]  -> [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]  Local and global coefficient
        self.c2 = c2
        self.w_min = w_min      # [0-1] -> [0.4-0.9]      Weight of bird
        self.w_max = w_max

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        v_max = 0.5 * (self.domain_range[1] - self.domain_range[0])
        v_list = uniform(0, v_max, (self.pop_size, self.problem_size))
        pop_local = deepcopy(pop)
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # Update weight after each move count  (weight down)
            w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
            for i in range(self.pop_size):
                v_new = w * v_list[i, :] + self.c1 * uniform() * (pop_local[i][self.ID_POS] - pop[i][self.ID_POS]) +\
                            self.c2 * uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                x_new = pop[i][self.ID_POS] + v_new             # Xi(new) = Xi(old) + Vi(new) * deltaT (deltaT = 1)
                x_new = self._amend_solution_random_faster__(x_new)
                fit_new = self._fitness_model__(x_new)
                pop[i] = [x_new, fit_new]

                # Update current position, current velocity and compare with past position, past fitness (local best)
                if fit_new < pop_local[i][self.ID_FIT]:
                    pop_local[i] = [x_new, fit_new]

            g_best = self._update_global_best__(pop_local, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class PPSO(Root):
    """
        A variant version of PSO: Phasor particle swarm optimization: a simple and efficient variant of PSO
        Matlab code sent by one of the author: Ebrahim Akbari
        I convert matlab to python code
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        v_max = 0.5 * (self.domain_range[1] - self.domain_range[0])
        v_list = zeros((self.pop_size, self.problem_size))
        delta_list = uniform(0, 2*pi, self.pop_size)
        pop_local = deepcopy(pop)
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            for i in range(0, self.pop_size):
                aa = 2 * (sin(delta_list[i]))
                bb = 2 * (cos(delta_list[i]))
                ee = abs(cos(delta_list[i])) ** aa
                tt = abs(sin(delta_list[i])) ** bb

                v_list[i, :] = ee * (pop_local[i][self.ID_POS] - pop[i][self.ID_POS]) + tt * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                v_list[i, :] = minimum(maximum(v_list[i], -v_max), v_max)

                x_temp = pop[i][self.ID_POS] + v_list[i, :]
                x_temp = minimum(maximum(x_temp, self.domain_range[0]), self.domain_range[1])
                fit = self._fitness_model__(x_temp)
                pop[i] = [x_temp, fit]

                delta_list[i] += abs(aa + bb) * (2 * pi)
                v_max = (abs(cos(delta_list[i])) ** 2) * (self.domain_range[1] - self.domain_range[0])

                if fit < pop_local[i][self.ID_FIT]:
                    pop_local[i] = [x_temp, fit]
                    if pop_local[i][self.ID_FIT] < g_best[self.ID_FIT]:
                        g_best = deepcopy(pop_local[i])

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class PSO_W(Root):
    """
        A variant version of PSO: Phasor particle swarm optimization: a simple and efficient variant of PSO
        Matlab code sent by one of the author: Ebrahim Akbari
        I convert matlab to python code
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        v_max = 0.5 * (self.domain_range[1] - self.domain_range[0])
        v_list = zeros((self.pop_size, self.problem_size))
        delta_list = uniform(0, 2 * pi, self.pop_size)
        pop_local = deepcopy(pop)
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            for i in range(0, self.pop_size):
                aa = 2 * (sin(delta_list[i]))
                bb = 2 * (cos(delta_list[i]))
                ee = abs(cos(delta_list[i])) ** aa
                tt = abs(sin(delta_list[i])) ** bb

                v_temp = ee * (pop_local[i][self.ID_POS] - pop[i][self.ID_POS]) + tt * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                v_list[i, :] = (ee/(i+1)) * v_list[i, :] + v_temp
                v_list[i, :] = minimum(maximum(v_list[i], -v_max), v_max)

                x_temp = pop[i][self.ID_POS] + v_list[i, :]
                x_temp = minimum(maximum(x_temp, self.domain_range[0]), self.domain_range[1])
                fit = self._fitness_model__(x_temp)
                pop[i] = [x_temp, fit]

                delta_list[i] += abs(aa + bb) * (2 * pi)
                v_max = (abs(cos(delta_list[i])) ** 2) * (self.domain_range[1] - self.domain_range[0])

                if fit < pop_local[i][self.ID_FIT]:
                    pop_local[i] = [x_temp, fit]
                    if pop_local[i][self.ID_FIT] < g_best[self.ID_FIT]:
                        g_best = deepcopy(pop_local[i])

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class HPSO_TVA(Root):
    """
        A variant version of PSO: New self-organising  hierarchical PSO with jumping time-varying acceleration coefficients
        Matlab code sent by one of the author: Ebrahim Akbari
        I convert matlab to python code
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, ci=0.5, cf=0.0):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.ci = ci
        self.cf = cf

    def _train__(self):
        # Initialization
        v_max = 0.5 * (self.domain_range[1] - self.domain_range[0])
        v_list = zeros((self.pop_size, self.problem_size))

        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop_local = deepcopy(pop)
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            c_it = ((self.cf - self.ci) * ((epoch+1) / self.epoch)) + self.ci

            for i in range(0, self.pop_size):
                idx_k = randint(0, self.pop_size)
                w = normal()
                while(abs(w - 1.0) < 0.01):
                    w = normal()
                c1_it = abs(w) ** (c_it * w)
                c2_it = abs(1 - w) ** (c_it / (1 - w))

                #################### HPSO
                v_list[i] = c1_it * uniform(0, 1, self.problem_size) * (pop_local[i][self.ID_POS] - pop[i][self.ID_POS]) + \
                    c2_it * uniform(0, 1, self.problem_size) * (g_best[self.ID_POS] + pop_local[idx_k][self.ID_POS] - 2*pop[i][self.ID_POS])

                where(v_list[i] == 0, sign(0.5-uniform()) * uniform() * v_max, v_list[i])
                v_list[i] = sign(v_list[i]) * minimum(abs(v_list[i]), v_max)
                #########################

                v_list[i] = minimum(maximum(v_list[i], -v_max), v_max)
                x_temp = pop[i][self.ID_POS] + v_list[i]
                x_temp = minimum(maximum(x_temp, self.domain_range[0]), self.domain_range[1])
                fit = self._fitness_model__(x_temp)
                pop[i] = [x_temp, fit]

                if fit < pop_local[i][self.ID_FIT]:
                    pop_local[i] = [x_temp, fit]
                    if pop_local[i][self.ID_FIT] < g_best[self.ID_FIT]:
                        g_best = deepcopy(pop_local[i])

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

