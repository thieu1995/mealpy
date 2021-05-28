#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:49, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, normal, randint, rand
from numpy import pi, sin, cos, zeros, minimum, maximum, abs, where, sign, mean, stack
from numpy import min as np_min
from numpy import max as np_max
from copy import deepcopy
from mealpy.root import Root


class BasePSO(Root):
    """
        The original version of: Particle Swarm Optimization (PSO)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 c1=1.2, c2=1.2, w_min=0.4, w_max=0.9, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c1 = c1            # [0-2]  -> [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]  Local and global coefficient
        self.c2 = c2
        self.w_min = w_min      # [0-1] -> [0.4-0.9]      Weight of bird
        self.w_max = w_max

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        v_max = 0.5 * (self.ub - self.lb)
        v_min = zeros(self.problem_size)
        v_list = uniform(v_min, v_max, (self.pop_size, self.problem_size))
        pop_local = deepcopy(pop)
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # Update weight after each move count  (weight down)
            w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
            for i in range(self.pop_size):
                v_new = w * v_list[i] + self.c1 * uniform() * (pop_local[i][self.ID_POS] - pop[i][self.ID_POS]) +\
                            self.c2 * uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                x_new = pop[i][self.ID_POS] + v_new             # Xi(new) = Xi(old) + Vi(new) * deltaT (deltaT = 1)
                x_new = self.amend_position_random_faster(x_new)
                fit_new = self.get_fitness_position(x_new)
                pop[i] = [x_new, fit_new]

                # Update current position, current velocity and compare with past position, past fitness (local best)
                if fit_new < pop_local[i][self.ID_FIT]:
                    pop_local[i] = [x_new, fit_new]

                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop_local, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size:
                        g_best = self.update_global_best_solution(pop_local, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class PPSO(Root):
    """
        A variant version of PSO: Phasor particle swarm optimization: a simple and efficient variant of PSO
        Matlab code sent by one of the author: Ebrahim Akbari
        I convert matlab to python code
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        v_max = 0.5 * (self.ub - self.lb)
        v_list = zeros((self.pop_size, self.problem_size))
        delta_list = uniform(0, 2*pi, self.pop_size)
        pop_local = deepcopy(pop)
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            for i in range(0, self.pop_size):
                aa = 2 * (sin(delta_list[i]))
                bb = 2 * (cos(delta_list[i]))
                ee = abs(cos(delta_list[i])) ** aa
                tt = abs(sin(delta_list[i])) ** bb

                v_list[i, :] = ee * (pop_local[i][self.ID_POS] - pop[i][self.ID_POS]) + tt * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                v_list[i, :] = minimum(maximum(v_list[i], -v_max), v_max)

                x_temp = pop[i][self.ID_POS] + v_list[i, :]
                x_temp = minimum(maximum(x_temp, self.lb), self.ub)
                fit = self.get_fitness_position(x_temp)
                pop[i] = [x_temp, fit]

                delta_list[i] += abs(aa + bb) * (2 * pi)
                v_max = (abs(cos(delta_list[i])) ** 2) * (self.ub - self.lb)

                if fit < pop_local[i][self.ID_FIT]:
                    pop_local[i] = [x_temp, fit]

                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class PSO_W(Root):
    """
        A variant version of PSO: Phasor particle swarm optimization: a simple and efficient variant of PSO
        Matlab code sent by one of the author: Ebrahim Akbari
        I convert matlab to python code
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        v_max = 0.5 * (self.ub - self.lb)
        v_list = zeros((self.pop_size, self.problem_size))
        delta_list = uniform(0, 2 * pi, self.pop_size)
        pop_local = deepcopy(pop)
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

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
                x_temp = minimum(maximum(x_temp, self.lb), self.ub)
                fit = self.get_fitness_position(x_temp)
                pop[i] = [x_temp, fit]

                delta_list[i] += abs(aa + bb) * (2 * pi)
                v_max = (abs(cos(delta_list[i])) ** 2) * (self.ub - self.lb)

                if fit < pop_local[i][self.ID_FIT]:
                    pop_local[i] = [x_temp, fit]

                ## batch size
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop_local, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size:
                        g_best = self.update_global_best_solution(pop_local, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class HPSO_TVA(Root):
    """
        A variant version of PSO: New self-organising hierarchical PSO with jumping time-varying acceleration coefficients
        Matlab code sent by one of the author: Ebrahim Akbari
        I convert matlab to python code
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, ci=0.5, cf=0.0, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.ci = ci
        self.cf = cf

    def train(self):
        # Initialization
        v_max = 0.5 * (self.ub - self.lb)
        v_list = zeros((self.pop_size, self.problem_size))

        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop_local = deepcopy(pop)
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

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
                x_temp = minimum(maximum(x_temp, self.lb), self.ub)
                fit = self.get_fitness_position(x_temp)
                pop[i] = [x_temp, fit]

                if fit < pop_local[i][self.ID_FIT]:
                    pop_local[i] = [x_temp, fit]

                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop_local, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size:
                        g_best = self.update_global_best_solution(pop_local, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class CPSO(Root):
    """
            Chaos Particle Swarm Optimization
        Paper: Improved particle swarm optimization combined with chaos
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 c1=1.2, c2=1.2, w_min=0.2, w_max=1.2, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c1 = c1  # [0-2]  -> [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]  Local and global coefficient
        self.c2 = c2
        self.w_min = w_min  # [0-1] -> [0.4-0.9]      Weight of bird
        self.w_max = w_max

    def __get_weights__(self, fit, fit_avg, fit_min):
        if fit <= fit_avg:
            return self.w_min + (self.w_max-self.w_min)*(fit - fit_min) / (fit_avg - fit_min)
        else:
            return self.w_max

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        v_max = 0.5 * (self.ub - self.lb)
        v_min = zeros(self.problem_size)
        v_list = uniform(v_min, v_max, (self.pop_size, self.problem_size))
        pop_local = deepcopy(pop)
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        N_CLS = int(self.pop_size / 5)      # Number of chaotic local searches
        for epoch in range(self.epoch):
            r = rand()

            list_fits = [item[self.ID_FIT] for item in pop]
            fit_avg = mean(list_fits)
            fit_min = np_min(list_fits)
            for i in range(self.pop_size):
                w = self.__get_weights__(pop[i][self.ID_FIT], fit_avg, fit_min)
                v_new = w * v_list[i] + self.c1 * rand() * (pop_local[i][self.ID_POS] - pop[i][self.ID_POS]) + \
                        self.c2 * rand() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                x_new = pop[i][self.ID_POS] + v_new
                x_new = self.amend_position_random_faster(x_new)
                fit_new = self.get_fitness_position(x_new)
                pop[i] = [x_new, fit_new]
                # Update current position, current velocity and compare with past position, past fitness (local best)
                if fit_new < pop_local[i][self.ID_FIT]:
                    pop_local[i] = [x_new, fit_new]

            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            ## Implement chaostic local search for the best solution
            cx_best_0 = (g_best[self.ID_POS] - self.lb) / (self.ub - self.lb)       # Eq. 7
            cx_best_1 = 4 * cx_best_0 * (1 - cx_best_0)                             # Eq. 6
            x_best = self.lb + cx_best_1 * (self.ub - self.lb)                      # Eq. 8
            fit_best = self.get_fitness_position(x_best)
            if fit_best < g_best[self.ID_FIT]:
                g_best = [x_best, fit_best]

            bound_min = stack([self.lb, g_best[self.ID_POS] - r * (self.ub - self.lb) ])
            self.lb = np_max(bound_min, axis=0)
            bound_max = stack([self.ub, g_best[self.ID_POS] + r * (self.ub - self.lb) ])
            self.ub = np_min(bound_max, axis=0)

            pop_new_child = [self.create_solution() for _ in range(self.pop_size-N_CLS)]
            pop_new = sorted(pop, key=lambda item: item[self.ID_FIT])
            pop = pop_new[:N_CLS] + pop_new_child

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

