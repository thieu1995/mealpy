#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:53, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform
from numpy.linalg import norm
from numpy import exp, power, pi, zeros, array, mean, ones, dot
from math import gamma
from copy import deepcopy
from mealpy.root import Root


class BaseRH(Root):
    """
    The original version of: Rhino Herd Optimization (RH)
        (A Novel Metaheuristic Algorithm inspired by Rhino Herd Behavior)
    Link:
        https://doi.org/10.3384/ecp171421026
    """

    def __init__(self, root_paras=None, epoch=750, pop_size=100, c=0.53, a=2831, r=0.04, A=1):
        Root.__init__(self, root_paras)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c = c              # shape parameter - default = 0.53 > 0
        self.a = a              # scale parameter - default = 2831 > 0
        self.r = r              # default = 0.04
        self.A = A              # the area of each grid cell - default = 1


    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)

        # Epoch loop
        for epoch in range(self.epoch):

            pos_list = array([item[self.ID_POS] for item in pop])
            fit_list = array([item[self.ID_FIT] for item in pop])
            fx_list = deepcopy(fit_list)
            pos_center = mean(pos_list, axis=0)

            ## Each individual loop
            for i in range(0, self.pop_size):
                # Eq. 1
                exp_component = -1 * power(norm(pop[i][self.ID_POS] - pos_center) / self.a, 2.0 / self.c)
                fx = 2 * exp(exp_component) / (self.c ** 2 * pi * self.a ** 2 * gamma(self.c))
                fx_list[i] = fx

            # Eq. 7
            s_component = ones(self.problem_size)
            for j in range(0, self.problem_size):
                sum_temp = 0
                for i in range(0, self.pop_size):
                    sum_temp += fx_list[i] * (1 + pop[i][self.ID_POS][j] / (self.EPSILON + pop[i][self.ID_FIT]))
                s_component[j] = self.A * sum_temp

            for i in range(0, self.pop_size):
                x_new = pop[i][self.ID_POS]
                for j in range(0, self.problem_size):
                    # Eq. 7
                    s_x = fx_list[i] * (1 + pop[i][self.ID_FIT] * pop[i][self.ID_POS][j]) / s_component[j]

                    # Eq. 9
                    if uniform() <= 0.5:
                        x_new[j] = pop[i][self.ID_POS][j] - uniform() * s_x * pop[i][self.ID_POS][j]
                    else:
                        x_new[j] = pop[i][self.ID_POS][j] + uniform() * s_x * pop[i][self.ID_POS][j]
                x_new = self._amend_solution_faster__(x_new)
                fit = self._fitness_model__(x_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [x_new, fit]

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class MyRH(BaseRH):
    """
    My version of: Rhino Herd Optimization (RH)
        (A Novel Metaheuristic Algorithm inspired by Rhino Herd Behavior)
    """

    def __init__(self, root_paras=None, epoch=750, pop_size=100, c=0.53, a=2831, r=0.04, A=1):
        BaseRH.__init__(self, root_paras, epoch, pop_size, c, a, r, A)

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)
        pop_size = self.pop_size

        # Epoch loop
        for epoch in range(self.epoch):
            pop_new = deepcopy(pop)

            pos_list = array([item[self.ID_POS] for item in pop])
            fit_list = array([item[self.ID_FIT] for item in pop])
            fx_list = deepcopy(fit_list)
            pos_center = mean(pos_list, axis=0)

            ## Calculate the fx for each individual
            for i in range(0, pop_size):
                # Eq. 1
                exp_component = -1 * power( norm(pop[i][self.ID_POS] - pos_center) / self.a , 2.0/self.c )
                fx = 2 * exp(exp_component) / (self.c ** 2 * pi * self.a ** 2 * gamma(self.c))
                fx_list[i] = fx

            #print(fx_list)

            # Eq. 7
            sum_temp = zeros(self.problem_size)
            for i in range(0, pop_size):
                sum_temp += fx_list[i] * (1 + pop[i][self.ID_POS] * pop[i][self.ID_FIT])
            sum_temp = self.A * sum_temp

            for i in range(0, pop_size):
                s_x = fx_list[i] * (1 + pop[i][self.ID_POS]/pop[i][self.ID_FIT]) / sum_temp
                if uniform() <= 0.5:
                    x_new = pop[i][self.ID_POS] - uniform() * dot(s_x, pop[i][self.ID_POS])
                else:
                    x_new = pop[i][self.ID_POS] + uniform() * dot(s_x, pop[i][self.ID_POS])
                x_new = self._amend_solution_faster__(x_new)
                fit = self._fitness_model__(x_new)
                if fit < pop[i][self.ID_FIT]:
                    pop_new[i] = [x_new, fit]

            if epoch % 100 == 0:
                pop_size = self.pop_size
                pop_new = sorted(pop_new, key=lambda item: item[self.ID_FIT])
                pop = deepcopy(pop_new[:pop_size])
            else:
                pop_size = pop_size + int(self.r * pop_size)
                n_new = pop_size - len(pop)
                for i in range(0, n_new):
                    pop_new.extend([self._create_solution__()])
                pop = deepcopy(pop_new)

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class Version3RH(BaseRH):
    """
    My version of: Rhino Herd Optimization (RH)
        (A Novel Metaheuristic Algorithm inspired by Rhino Herd Behavior)
    """

    def __init__(self, root_paras=None, epoch=750, pop_size=100, c=0.53, a=2831, r=0.04, A=1):
        BaseRH.__init__(self, root_paras, epoch, pop_size, c, a, r, A)

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)
        pop_size = self.pop_size

        # Epoch loop
        for epoch in range(self.epoch):
            pop_new = deepcopy(pop)

            pos_list = array([item[self.ID_POS] for item in pop])
            pos_center = mean(pos_list, axis=0)
            fx_list = zeros(pop_size)

            ## Calculate the fx for each individual
            for i in range(0, pop_size):
                # Eq. 1
                exp_component = -1 * power( norm(pop[i][self.ID_POS] - pos_center) / self.a , 2.0/self.c )
                fx = 2 * exp(exp_component) / (self.c ** 2 * pi * self.a ** 2 * gamma(self.c))
                fx_list[i] = fx
            #print(fx_list)
            # Eq. 7
            sum_temp = zeros(self.problem_size)
            for i in range(0, self.pop_size):
                sum_temp += fx_list[i] * (1 + pop[i][self.ID_POS] / pop[i][self.ID_FIT])
            sum_temp = self.A * sum_temp

            for i in range(0, pop_size):
                s_x = fx_list[i] * (1 + pop[i][self.ID_FIT] * pop[i][self.ID_POS]) / sum_temp
                if uniform() <= 0.5:
                    x_new = pop[i][self.ID_POS] - uniform() * dot(s_x, pop[i][self.ID_POS])
                else:
                    x_new = pop[i][self.ID_POS] + uniform() * dot(s_x, pop[i][self.ID_POS])
                x_new = self._amend_solution_faster__(x_new)
                fit = self._fitness_model__(x_new)
                if fit < pop[i][self.ID_FIT]:
                    pop_new[i] = [x_new, fit]

            if epoch % 100 == 0:
                pop_size = self.pop_size
                pop_new = sorted(pop_new, key=lambda item: item[self.ID_FIT])
                pop = deepcopy(pop_new[:pop_size])
            else:
                pop_size = pop_size + int(self.r * pop_size)
                n_new = pop_size - len(pop)
                for i in range(0, n_new):
                    pop_new.extend([self._create_solution__()])
                pop = deepcopy(pop_new)

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Pop Size: {}, Best Fit: {}".format(epoch+1, pop_size, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

