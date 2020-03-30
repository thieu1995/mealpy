#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:51, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

"""
After spending several day to customize and optimize, I finally implement successfully the orignal verion and
    three variant version of PathFinder Algorithm.

class: BasePFA is the final version of original version of PFA
class: OPFA is an enhanced version of PFA based on Opposition-based Learning
class: LPFA is an enhanced version of PFA based on Levy-flight trajectory
class: IPFA is an improved version of PFA based on both Opposition-based Learning and Levy-flight

Simple test with CEC14:
Lets try C1 objective function

BasePFA: after 12 loop, it reaches value 100.0
OPFA: after 10 loop, it reaches value 100.0     (a little improvement)
LPFA: after 4 loop, it reaches value 100.0      (a huge improvement)
IPFA: after 2 loop, it reaches value 100.0      (best improvement)
"""

from numpy import exp, sqrt, setxor1d, array, pi, power, abs, sin, sign
from numpy.random import uniform, choice, normal, random
from numpy.linalg import norm
from copy import deepcopy
from math import gamma, pi
from mealpy.root import Root


class BasePFA(Root):
    """
    A new meta-heuristic optimizer: Pathfinder algorithm
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size

    def _train__(self):
        # Init pop and calculate fitness
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        gbest_present = deepcopy(g_best)

        for i in range(self.epoch):
            alpha, beta = uniform(1, 2, 2)
            A = uniform(self.domain_range[0], self.domain_range[1]) * exp(-2 * (i + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present[self.ID_POS] + 2 * uniform() * (gbest_present[self.ID_POS] - g_best[self.ID_POS]) + A
            temp = self._amend_solution_faster__(temp)
            fit = self._fitness_model__(temp)
            g_best = deepcopy(gbest_present)
            if fit < gbest_present[self.ID_FIT]:
                gbest_present = [temp, fit]
            pop[0] = deepcopy(gbest_present)

            ## Update positions of members, check the bound and calculate new fitness
            for j in range(1, self.pop_size):
                temp1 = deepcopy(pop[j][self.ID_POS])
                temp2 = deepcopy(pop[j][self.ID_POS])

                t1 = beta * uniform() * (gbest_present[self.ID_POS] - temp1)
                for k in range(1, self.pop_size):
                    dist = norm(pop[k][self.ID_POS] - temp1)
                    t2 = alpha * uniform() * (pop[k][self.ID_POS] - temp1)
                    t3 = uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * (1 - (i + 1) * 1.0 / self.epoch) * dist
                    temp2 += t2 + t3
                temp2 += t1

                ## Update members
                temp2 = self._amend_solution_faster__(temp2)
                fit = self._fitness_model__(temp2)
                if fit < pop[j][self.ID_FIT]:
                    pop[j] = [temp2, fit]

            ## Update the best solution found so far (current pathfinder)
            pop, gbest_present = self._update_global_best__(pop, self.ID_MIN_PROB, gbest_present)
            self.loss_train.append(gbest_present[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(i + 1, gbest_present[self.ID_FIT]))

        return gbest_present[self.ID_POS], gbest_present[self.ID_FIT], self.loss_train


class OPFA(BasePFA):

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        BasePFA.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)

    def _train__(self):
        # Init pop and calculate fitness
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        gbest_present = deepcopy(g_best)

        for i in range(self.epoch):
            alpha, beta = uniform(1, 2, 2)
            A = uniform(self.domain_range[0], self.domain_range[1]) * exp(-2 * (i + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present[self.ID_POS] + 2 * uniform() * (gbest_present[self.ID_POS] - g_best[self.ID_POS]) + A
            temp = self._amend_solution_faster__(temp)
            fit = self._fitness_model__(temp)
            g_best = deepcopy(gbest_present)
            if fit < gbest_present[self.ID_FIT]:
                gbest_present = [temp, fit]
            pop[0] = deepcopy(gbest_present)

            ## Update positions of members, check the bound and calculate new fitness
            for j in range(1, self.pop_size):
                temp1 = deepcopy(pop[j][self.ID_POS])
                temp2 = deepcopy(temp1)

                t1 = beta * uniform() * (gbest_present[self.ID_POS] - temp1)
                for k in range(1, self.pop_size):
                    dist = norm(pop[k][self.ID_POS] - temp1)
                    t2 = alpha * uniform() * (pop[k][self.ID_POS] - temp1)
                    t3 = uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * (1 - (i + 1) * 1.0 / self.epoch) * dist
                    temp2 += t2 + t3
                temp2 += t1

                ## Update members based on Opposition-based learning
                temp2 = self._amend_solution_faster__(temp2)
                fit = self._fitness_model__(temp2)
                if fit < pop[j][self.ID_FIT]:
                    pop[j] = [temp2, fit]
                else:
                    C_op = self._create_opposition_solution__(temp2, gbest_present[self.ID_POS])
                    fit_op = self._fitness_model__(C_op)
                    if fit_op < pop[j][self.ID_FIT]:
                        pop[j] = [C_op, fit_op]

            ## Update the best solution found so far (current pathfinder)
            gbest_present = self._update_global_best__(pop, self.ID_MIN_PROB, gbest_present)
            self.loss_train.append(gbest_present[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(i + 1, gbest_present[self.ID_FIT]))

        return gbest_present[self.ID_POS], gbest_present[self.ID_FIT], self.loss_train


class LPFA(BasePFA):
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        BasePFA.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)

    def _train__(self):
        # Init pop and calculate fitness
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        gbest_present = deepcopy(g_best)

        for epoch in range(self.epoch):
            alpha, beta = uniform(1, 2, 2)
            A = uniform(self.domain_range[0], self.domain_range[1]) * exp(-2 * (epoch + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present[self.ID_POS] + 2 * uniform() * (gbest_present[self.ID_POS] - g_best[self.ID_POS]) + A
            temp = self._amend_solution_faster__(temp)
            fit = self._fitness_model__(temp)
            g_best = deepcopy(gbest_present)
            if fit < gbest_present[self.ID_FIT]:
                gbest_present = [temp, fit]
            pop[0] = deepcopy(gbest_present)

            ## Update positions of members, check the bound and calculate new fitness
            for i in range(1, self.pop_size):
                temp1 = deepcopy(pop[i][self.ID_POS])
                temp2 = deepcopy(temp1)
                if uniform() < 0.5:
                    t1 = beta * uniform() * (gbest_present[self.ID_POS] - temp1)
                    for k in range(1, self.pop_size):
                        dist = norm(pop[k][self.ID_POS] - temp1)
                        t2 = alpha * uniform() * (pop[k][self.ID_POS] - temp1)
                        t3 = uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * (1 - (epoch + 1) * 1.0 / self.epoch) * dist
                        temp2 += t2 + t3
                    temp2 += t1
                else:
                    ## Using Levy-flight to boost algorithm's convergence speed
                    temp2 = self._levy_flight__(epoch, pop[i][self.ID_POS], gbest_present[self.ID_POS])

                ## Update members
                temp2 = self._amend_solution_faster__(temp2)
                fit = self._fitness_model__(temp2)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp2, fit]

            ## Update the best solution found so far (current pathfinder)
            gbest_present = self._update_global_best__(pop, self.ID_MIN_PROB, gbest_present)
            self.loss_train.append(gbest_present[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, gbest_present[self.ID_FIT]))

        return gbest_present[self.ID_POS], gbest_present[self.ID_FIT], self.loss_train


class IPFA(LPFA):
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        LPFA.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)

    def _train__(self):
        # Init pop and calculate fitness
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        gbest_present = deepcopy(g_best)

        for epoch in range(0, self.epoch):
            alpha, beta = uniform(1, 2, 2)
            A = uniform(self.domain_range[0], self.domain_range[1]) * exp(-2 * (epoch + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present[self.ID_POS] + 2*uniform()*( gbest_present[self.ID_POS] - g_best[self.ID_POS]) + A
            temp = self._amend_solution_faster__(temp)
            fit = self._fitness_model__(temp)
            g_best = deepcopy(gbest_present)
            if fit < gbest_present[self.ID_FIT]:
                gbest_present = [temp, fit]
            pop[0] = deepcopy(gbest_present)

            ## Update positions of members, check the bound and calculate new fitness
            for i in range(1, self.pop_size):
                temp1 = deepcopy(pop[i][self.ID_POS])
                temp2 = deepcopy(temp1)
                if uniform() < 0.5:
                    t1 = beta * uniform() * (gbest_present[self.ID_POS] - temp1)
                    for k in range(1, self.pop_size):
                        dist = norm(pop[k][self.ID_POS] - temp1)
                        t2 = alpha * uniform() * (pop[k][self.ID_POS] - temp1)
                        t3 = uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * (1 - (epoch + 1) * 1.0 / self.epoch) * dist
                        temp2 += t2 + t3
                    temp2 += t1
                else:
                    ## Using Levy-flight to boost algorithm's convergence speed
                    temp2 = self._levy_flight__(epoch, pop[i][self.ID_POS], gbest_present[self.ID_POS])

                ## Update members based on Opposition-based learning
                temp2 = self._amend_solution_faster__(temp2)
                fit = self._fitness_model__(temp2)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp2, fit]
                else:
                    C_op = self._create_opposition_solution__(temp, gbest_present[self.ID_POS])
                    fit_op = self._fitness_model__(C_op)
                    if fit_op < pop[i][self.ID_FIT]:
                        pop[i] = [C_op, fit_op]

            ## Update the best solution found so far (current pathfinder)
            gbest_present = self._update_global_best__(pop, self.ID_MIN_PROB, gbest_present)
            self.loss_train.append(gbest_present[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, gbest_present[self.ID_FIT]))

        return gbest_present[self.ID_POS], gbest_present[self.ID_FIT], self.loss_train



class DePFA(BasePFA):
    """
    A new meta-heuristic optimizer: Pathfinder algorithm
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        BasePFA.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)

    def _train__(self):
        # Init pop and calculate fitness
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        gbest_present = deepcopy(g_best)

        for i in range(self.epoch):
            alpha, beta = uniform(1, 2, 2)
            A = uniform(self.domain_range[0], self.domain_range[1]) * exp(-2 * (i + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present[self.ID_POS] + 2 * uniform() * (gbest_present[self.ID_POS] - g_best[self.ID_POS]) + A
            temp = self._amend_solution_faster__(temp)
            fit = self._fitness_model__(temp)
            g_best = deepcopy(gbest_present)
            if fit < gbest_present[self.ID_FIT]:
                gbest_present = [temp, fit]
            pop[0] = deepcopy(gbest_present)

            ## Update positions of members, check the bound and calculate new fitness
            for j in range(1, self.pop_size):
                temp1 = deepcopy(pop[j][self.ID_POS])

                t1 = beta * uniform() * (gbest_present[self.ID_POS] - temp1)
                my_list_idx = setxor1d( array(range(1, self.pop_size)) , array([j]) )
                idx = choice(my_list_idx)
                dist = norm(pop[idx][self.ID_POS] - temp1)
                t2 = alpha * uniform() * (pop[idx][self.ID_POS] - temp1)
                t3 = uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * (1 - (i + 1) * 1.0 / self.epoch) * dist
                temp1 += t1 + t2 + t3

                ## Update members
                temp1 = self._amend_solution_faster__(temp1)
                fit = self._fitness_model__(temp1)
                if fit < pop[j][self.ID_FIT]:
                    pop[j] = [temp1, fit]

            ## Update the best solution found so far (current pathfinder)
            pop, gbest_present = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, gbest_present)
            self.loss_train.append(gbest_present[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(i + 1, gbest_present[self.ID_FIT]))

        return gbest_present[self.ID_POS], gbest_present[self.ID_FIT], self.loss_train


class LevyDePFA(DePFA):
    """
    A new meta-heuristic optimizer: Pathfinder algorithm
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        DePFA.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)

    def _caculate_xichma__(self, beta):
        up = gamma(1 + beta) * sin(pi * beta / 2)
        down = (gamma((1 + beta) / 2) * beta * power(2, (beta - 1) / 2))
        xich_ma_1 = power(up / down, 1 / beta)
        xich_ma_2 = 1
        return xich_ma_1, xich_ma_2

    def _shrink_encircling_Levy__(self, current_sea_lion, epoch_i, dist, c, beta=1):
        xich_ma_1, xich_ma_2 = self._caculate_xichma__(beta)
        a = normal(0, xich_ma_1, 1)
        b = normal(0, xich_ma_2, 1)
        LB = 0.01 * a / (power(abs(b), 1 / beta)) * dist * c
        D = uniform(self.domain_range[0], self.domain_range[1], 1)
        levy = LB * D
        return (current_sea_lion - sqrt(epoch_i + 1) * sign(random(1) - 0.5)) * levy

    def _train__(self):
        # Init pop and calculate fitness
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        current_best = deepcopy(g_best)

        for epoch in range(self.epoch):
            alpha, beta = uniform(1, 2, 2)
            A = uniform(self.domain_range[0], self.domain_range[1]) * exp(-2 * (epoch + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = g_best[self.ID_POS] + 2 * uniform() * (g_best[self.ID_POS] - current_best[self.ID_POS]) + A
            temp = self._amend_solution_faster__(temp)
            fit = self._fitness_model__(temp)
            current_best = deepcopy(g_best)
            if fit < g_best[self.ID_FIT]:
                g_best = [temp, fit]
            pop[0] = deepcopy([temp, fit])

            ## Update positions of members, check the bound and calculate new fitness
            for i in range(1, self.pop_size):
                t1 = beta * uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                idx = choice( setxor1d(array(range(1, self.pop_size)), array([i])) )
                dist = norm(pop[idx][self.ID_POS] - pop[i][self.ID_POS])
                t2 = alpha * uniform() * (pop[idx][self.ID_POS] - pop[i][self.ID_POS])
                t3 = uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * (1 - (epoch + 1) * 1.0 / self.epoch) * dist
                temp = pop[i][self.ID_POS] + t1 + t2 + t3

                ## Update members
                temp = self._amend_solution_faster__(temp)
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

            ## Update the best solution found so far (current pathfinder)
            pop, g_best = self._sort_pop_and_update_global_best__(pop,self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

