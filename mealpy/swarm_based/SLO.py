#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:05, 03/06/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint, normal, random, choice, rand
from numpy import abs, sign, cos, pi, sin, sqrt, power
from math import gamma
from copy import deepcopy
from mealpy.root import Root


class BaseSLO(Root):
    """
        The original version of: Sea Lion Optimization Algorithm (SLO)
            (Sea Lion Optimization Algorithm)
        Link:
            https://www.researchgate.net/publication/333516932_Sea_Lion_Optimization_Algorithm
            DOI: 10.14569/IJACSA.2019.0100548
        Notes:
            + The original paper is unclear in some equations and parameters
            + This version is based on my expertise
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            c = 2 - 2 * epoch / self.epoch
            t0 = rand()
            v1 = sin(2 * pi * t0)
            v2 = sin(2 * pi * (1 - t0))
            SP_leader = abs(v1 * (1 + v2) / v2)     # In the paper this is not clear how to calculate

            for i in range(self.pop_size):
                if SP_leader < 0.25:
                    if c < 1:
                        pos_new = g_best[self.ID_POS] - c * abs(2 * rand() * g_best[self.ID_POS] - pop[i][self.ID_POS])
                    else:
                        ri = choice(list(set(range(0, self.pop_size)) - {i}))  # random index
                        pos_new = pop[ri][self.ID_POS] - c * abs(2 * rand() * pop[ri][self.ID_POS] - pop[i][self.ID_POS])
                else:
                    pos_new = abs(g_best[self.ID_POS] - pop[i][self.ID_POS]) * cos(2 * pi * uniform(-1, 1)) + g_best[self.ID_POS]

                pos_new = self.amend_position_random(pos_new)    # In the paper doesn't check also doesn't update old solution at this point
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class ISLO(Root):
    """
        My improved version of: Improved Sea Lion Optimization Algorithm (ISLO)
            (Sea Lion Optimization Algorithm)
        Link:
            https://www.researchgate.net/publication/333516932_Sea_Lion_Optimization_Algorithm
            DOI: 10.14569/IJACSA.2019.0100548
    """
    ID_POS_LOC = 2
    ID_POS_FIT = 3

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, c1=1.2, c2=1.2, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2

    def create_solution(self, minmax=0):
        ## Increase exploration at the first initial population using opposition-based learning.
        position = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position, minmax=minmax)
        local_pos = self.lb + self.ub - position
        local_fit = self.get_fitness_position(position=local_pos, minmax=minmax)
        if fitness < local_fit:
            return [local_pos, local_fit, position, fitness]
        else:
            return [position, fitness, local_pos, local_fit]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            c = 2 - 2 * epoch / self.epoch
            t0 = rand()
            v1 = sin(2 * pi * t0)
            v2 = sin(2 * pi * (1 - t0))
            SP_leader = abs(v1 * (1 + v2) / v2)

            for i in range(self.pop_size):
                if SP_leader < 0.5:
                    if c < 1:  # Exploitation improved by historical movement + global best affect
                        # pos_new = g_best[self.ID_POS] - c * abs(2 * rand() * g_best[self.ID_POS] - pop[i][self.ID_POS])
                        dif1 = abs(2 * rand() * g_best[self.ID_POS] - pop[i][self.ID_POS])
                        dif2 = abs(2 * rand() * pop[i][self.ID_POS_LOC] - pop[i][self.ID_POS])
                        pos_new = self.c1 * rand() * (pop[i][self.ID_POS] - c * dif1) + self.c2 * rand() * (pop[i][self.ID_POS] - c * dif2)
                    else:  # Exploration improved by opposition-based learning
                        # Create a new solution by equation below
                        # Then create an opposition solution of above solution
                        # Compare both of them and keep the good one (Searching at both direction)
                        pos_new = g_best[self.ID_POS] + c * normal(0, 1, self.problem_size) * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                        fit_new = self.get_fitness_position(pos_new)
                        pos_new_oppo = self.lb + self.ub - g_best[self.ID_POS] + rand() * (g_best[self.ID_POS] - pos_new)
                        fit_new_oppo = self.get_fitness_position(pos_new_oppo)
                        if fit_new_oppo < fit_new:
                            pos_new = pos_new_oppo
                else:   # Exploitation
                    pos_new = g_best[self.ID_POS] + cos(2 * pi * uniform(-1, 1)) * abs(g_best[self.ID_POS] - pop[i][self.ID_POS])
                pos_new = self.amend_position_random(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_POS_FIT]:
                    pop[i] = [pos_new, fit, deepcopy(pos_new), deepcopy(fit)]
                else:
                    pop[i][self.ID_POS] = pos_new
                    pop[i][self.ID_FIT] = fit
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class ModifiedSLO(BaseSLO):
    """
        My modified version of: Sea Lion Optimization (ISLO)
            (Sea Lion Optimization Algorithm)
        Noted:
            + Using the idea of shrink encircling combine with levy flight techniques
            + Also using the idea of local best in PSO
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseSLO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def _shrink_encircling_levy__(self, current_pos, epoch, dist, c, beta=1):
        up = gamma(1 + beta) * sin(pi * beta / 2)
        down = (gamma((1 + beta) / 2) * beta * power(2, (beta - 1) / 2))
        xich_ma_1 = power(up / down, 1 / beta)
        xich_ma_2 = 1
        a = normal(0, xich_ma_1, 1)
        b = normal(0, xich_ma_2, 1)
        LB = 0.01 * a / (power(abs(b), 1 / beta)) * dist * c
        D = uniform(self.lb, self.ub)
        levy = LB * D
        return (current_pos - sqrt(epoch + 1) * sign(random(1) - 0.5)) * levy

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop_local = deepcopy(pop)
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            c = 2 - 2 * epoch / self.epoch
            if c > 1:
                pa = 0.3  # At the beginning of the process, the probability for shrinking encircling is small
            else:
                pa = 0.7  # But at the end of the process, it become larger. Because sea lion are shrinking encircling prey
            SP_leader = uniform(0, 1)
            for i in range(self.pop_size):
                if SP_leader >= 0.6:
                    new_pos = cos(2 * pi * normal(0, 1)) * abs(g_best[self.ID_POS] - pop[i][self.ID_POS]) + g_best[self.ID_POS]
                else:
                    if uniform() < pa:
                        dist1 = uniform() * abs(2 * g_best[self.ID_POS] - pop[i][self.ID_POS])
                        new_pos = self._shrink_encircling_levy__(pop[i][self.ID_POS], epoch, dist1, c)
                    else:
                        rand_index = randint(0, self.pop_size)
                        rand_SL = pop_local[rand_index][self.ID_POS]
                        rand_SL = 2 * g_best[self.ID_POS] - rand_SL
                        new_pos = rand_SL - c * abs(uniform() * rand_SL - pop[i][self.ID_POS])

                new_pos = self.amend_position_random(new_pos)
                new_fit = self.get_fitness_position(new_pos)
                pop[i] = [new_pos, new_fit]  # Move to new position anyway
                # Update current_pos, current_fit, compare and update past_pos and past_fit
                if new_fit < pop_local[i][self.ID_FIT]:
                    pop_local[i] = deepcopy([new_pos, new_fit])

                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
