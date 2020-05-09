#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:08, 19/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import ceil, sqrt, abs, array, ones, mean, repeat, sin, cos
from numpy.random import uniform, normal, random, choice
from copy import deepcopy
from mealpy.root import Root
from mealpy.human_based.LCBO import BaseLCBO
from mealpy.human_based.SSDO import LevySSDO


class BaseCEM(Root):
    """
        The original version of: Cross-Entropy Method (CEM)
            http://www.cleveralgorithms.com/nature-inspired/probabilistic/cross_entropy.html
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, n_best=30, alpha=0.7):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha
        self.n_best = n_best
        self.means, self.stdevs = None, None

    def _create_solution_ce__(self, minmax=0):
        pos = normal(self.means, self.stdevs, self.problem_size)
        pos = self._amend_solution_random_faster__(pos)
        fit = self._fitness_model__(pos, minmax=minmax)
        return [pos, fit]

    def _train__(self):
        self.means = random(self.problem_size) * (self.domain_range[1] - self.domain_range[0]) + self.domain_range[0]
        self.stdevs = abs((self.domain_range[1] - self.domain_range[0]) * ones(self.problem_size))
        pop = [self._create_solution_ce__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            ## Selected the best samples and update means and stdevs
            pop_best = pop[:self.n_best]
            pos_list = array([item[self.ID_POS] for item in pop_best])

            means_new = mean(pos_list, axis=0)
            means_new_repeat = repeat(means_new.reshape((1, -1)), self.n_best, axis=0)
            stdevs_new = mean((pos_list - means_new_repeat) ** 2, axis=0)
            self.means = self.alpha * self.means + (1.0 - self.alpha) * means_new
            self.stdevs = abs(self.alpha * self.means + (1.0 - self.alpha) * stdevs_new)

            ## Update elite if a bower becomes fitter than the elite
            g_best = self._update_global_best__(pop_best, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

            ## Create new population for next generation
            pop = [self._create_solution_ce__() for _ in range(self.pop_size)]
            pop = sorted(pop, key=lambda item: item[self.ID_FIT])

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class CEBaseLCBO(BaseLCBO):
    """
        The hybrid version of: Cross-Entropy Method (CEM) and Life Choice-Based Optimization
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, alpha=0.7, r1=2.35):
        BaseLCBO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size, r1)
        self.n1 = int(ceil(sqrt(self.pop_size)))                    # n best solution in LCBO
        self.n2 = self.n1 + int((self.pop_size - self.n1) / 2)      # 50% for both 2 group left
        self.n_best = int(ceil(sqrt(self.pop_size)))                # n nest solution in CE
        self.alpha = alpha                                          # alpha in CE
        self.epoch_ce = int(ceil(epoch/5))                          # Epoch in CE
        self.means, self.stdevs = None, None

    def _create_solution_ce__(self, minmax=0):
        pos = normal(self.means, self.stdevs, self.problem_size)
        pos = self._amend_solution_random_faster__(pos)
        fit = self._fitness_model__(pos, minmax=minmax)
        return [pos, fit]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        # epoch: current chance, self.epoch: number of chances
        for epoch in range(self.epoch):
            ## Here for LCBO algorithm (Exploration)
            for i in range(0, self.pop_size):
                ## Since we already sorted population, we know which ones are 1st group
                if i < self.n1:
                    temp = array([uniform() * pop[j][self.ID_POS] for j in range(0, self.n1)])
                    temp = mean(temp, axis=0)
                elif i < self.n2:  # People in group 2 learning from the best person in the history, because they want to be better than the
                    # current best person
                    temp = self._levy_flight__(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])
                else:  # People in group 2 learning from the current best person and the person slightly better than them, because they don't have vision
                    f1 = 1 - (epoch + 1) / self.epoch
                    f2 = 1 - f1
                    better_diff = f2 * self.r1 * (pop[i - 1][self.ID_POS] - pop[i][self.ID_POS])
                    best_diff = f1 * self.r1 * (pop[0][self.ID_POS] - pop[i][self.ID_POS])
                    temp = pop[i][self.ID_POS] + uniform() * better_diff + uniform() * best_diff
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]
            # Update the global best
            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)

            # Initialization process of CE
            pos_list = array([item[self.ID_POS] for item in pop])
            self.means = mean(pos_list, axis=0)
            means_new_repeat = repeat(self.means.reshape((1, -1)), self.pop_size, axis=0)
            self.stdevs = mean(((pos_list - means_new_repeat) ** 2), axis=0)

            ## Here for CE algorithm (Exploitation)
            for epoch_ce in range(self.epoch_ce):
                ## Selected the best samples and update means and stdevs
                pop_best = pop[:self.n_best]
                pos_list = array([item[self.ID_POS] for item in pop_best])

                means_new = mean(pos_list, axis=0)
                means_new_repeat = repeat(means_new.reshape((1, -1)), self.n_best, axis=0)
                stdevs_new = mean(((pos_list - means_new_repeat) ** 2), axis=0)

                self.means = self.alpha * self.means + (1.0 - self.alpha) * means_new
                self.stdevs = abs(self.alpha * self.stdevs + (1.0 - self.alpha) * stdevs_new)

                ## Update elite if a bower becomes fitter than the elite
                g_best = self._update_global_best__(pop_best, self.ID_MIN_PROB, g_best)

                ## Create new population for next generation
                pop = [self._create_solution_ce__() for _ in range(self.pop_size)]
                pop = sorted(pop, key=lambda item: item[self.ID_FIT])

            # Update the final global best
            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class CEBaseLCBONew(BaseLCBO):
    """
        The hybrid version of: Cross-Entropy Method (CEM) and Life Choice-Based Optimization
        Version 2: Instead replace the old population, now it will replace only few worst individuals
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, alpha=0.7, r1=2.35):
        BaseLCBO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size, r1)
        self.n1 = int(ceil(sqrt(self.pop_size)))                # n best solution in LCBO
        self.n2 = self.n1 + int((self.pop_size - self.n1) / 2)  # 50% for both 2 group left
        self.n_best = int(ceil(sqrt(self.pop_size)))            # n best solution in CE
        self.pop_size_ce = int(pop_size/2)                      # population size of CE algorithm
        self.alpha = alpha                                      # alpha in CE
        self.epoch_ce = int(ceil(epoch / 5))                    # Epoch in CE
        self.means, self.stdevs = None, None

    def _create_solution_ce__(self, minmax=0):
        pos = normal(self.means, self.stdevs, self.problem_size)
        pos = self._amend_solution_random_faster__(pos)
        fit = self._fitness_model__(pos, minmax=minmax)
        return [pos, fit]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        # epoch: current chance, self.epoch: number of chances
        for epoch in range(self.epoch):
            ## Here for LCBO algorithm (Exploration)
            for i in range(0, self.pop_size):
                ## Since we already sorted population, we know which ones are 1st group
                if i < self.n1:
                    temp = array([uniform() * pop[j][self.ID_POS] for j in range(0, self.n1)])
                    temp = mean(temp, axis=0)
                elif self.n1 <= i < self.n2:  # People in group 2 learning from the best person in the history, because they want to be better than the
                    # current best person
                    f = (epoch + 1) / self.epoch
                    better_diff = f * self.r1 * (pop[i - 1][self.ID_POS] - pop[i][self.ID_POS])
                    best_diff = (1 - f) * self.r1 * (pop[0][self.ID_POS] - pop[i][self.ID_POS])
                    temp = pop[i][self.ID_POS] + uniform() * better_diff + uniform() * best_diff
                else:  # People in group 2 learning from the current best person and the person slightly better than them, because they don't have vision
                    #temp = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                    x_min = self.domain_range[0] * ones(self.problem_size)
                    x_max = self.domain_range[1] * ones(self.problem_size)
                    temp = x_max - (pop[i][self.ID_POS] - x_min) * uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]
            # Update the global best
            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)

            pop_ce = deepcopy(pop[:self.pop_size_ce])
            # Initialization process of CE
            pos_list = array([item[self.ID_POS].copy() for item in pop_ce])
            self.means = mean(pos_list, axis=0)
            means_new_repeat = repeat(self.means.reshape((1, -1)), self.pop_size_ce, axis=0)
            self.stdevs = mean(((pos_list - means_new_repeat) ** 2), axis=0)

            ## Here for CE algorithm (Exploitation)
            for epoch_ce in range(self.epoch_ce):
                ## Selected the best samples and update means and stdevs
                pop_best = pop_ce[:self.n_best]
                pos_list = array([item[self.ID_POS] for item in pop_best])

                means_new = mean(pos_list, axis=0)
                means_new_repeat = repeat(means_new.reshape((1, -1)), self.n_best, axis=0)
                stdevs_new = mean(((pos_list - means_new_repeat) ** 2), axis=0)

                self.means = self.alpha * self.means + (1.0 - self.alpha) * means_new
                self.stdevs = abs(self.alpha * self.stdevs + (1.0 - self.alpha) * stdevs_new)

                ## Update elite if a bower becomes fitter than the elite
                g_best = self._update_global_best__(pop_best, self.ID_MIN_PROB, g_best)

                ## Create new population for next generation
                pop_ce = [self._create_solution_ce__() for _ in range(self.pop_size_ce)]
                pop_ce = sorted(pop_ce, key=lambda item: item[self.ID_FIT])

            ## Replace the worst in pop by pop_ce
            idx_list = choice(range(0, self.pop_size), self.pop_size_ce, replace=False)
            for idx, idx_item in enumerate(idx_list):
                pop[idx_item] = deepcopy(pop_ce[idx])
            #pop = pop[(self.pop_size - self.pop_size_ce):] + pop_ce

            # Update the final global best
            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class CEBaseSSDO(LevySSDO):
    ID_POS = 0
    ID_FIT = 1
    ID_VEL = 2  # velocity
    ID_LBS = 3  # local best solution
    """
        The hybrid version of: Cross-Entropy Method (CEM) and Social Sky-Driving Optimization
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, alpha=0.7):
        LevySSDO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)
        self.n_best = int(ceil(sqrt(self.pop_size)))        # n nest solution in CE
        self.alpha = alpha                                  # alpha in CE
        self.epoch_ce = int(ceil(epoch / 5))                # Epoch in CE
        self.means, self.stdevs = None, None

    def _create_solution_ce_(self, pop, idx):
        pos = normal(self.means, self.stdevs, self.problem_size)
        pos = self._amend_solution_random_faster__(pos)
        fit = self._fitness_model__(pos)
        velocity = pop[idx][self.ID_VEL]
        local_best_solution = pop[idx][self.ID_LBS]
        return [pos, fit, velocity, local_best_solution]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            c = 2 - epoch * (2.0 / self.epoch)  # a decreases linearly from 2 to 0

            ## Calculate the mean of the best three solutions in each dimension. Eq 9
            pos_list_3 = array([indi[self.ID_POS] for indi in pop[:3]])
            pos_mean = mean(pos_list_3, axis=0)

            # Updating velocity vectors
            for i in range(0, self.pop_size):
                r1 = uniform()
                if uniform() < 0.5:  ## Use Sine function to move
                    vel_new = c * sin(r1) * (pop[i][self.ID_LBS] - pop[i][self.ID_POS]) + (2 - c) * sin(r1) * (pos_mean - pop[i][self.ID_POS])
                else:  ## Use Cosine function to move
                    vel_new = c * cos(r1) * (pop[i][self.ID_LBS] - pop[i][self.ID_POS]) + (2 - c) * cos(r1) * (pos_mean - pop[i][self.ID_POS])
                pop[i][self.ID_VEL] = vel_new

            # Update Position based on velocity
            for i in range(0, self.pop_size):
                ## In real life, there are lots of cases skydrive person death because they can't follow the instructor, or something went wrong with their
                # parasol. Inspired by that I added levy-flight is the a bold moves
                if uniform() < 0.5:
                    temp = pop[i][self.ID_POS] + pop[i][self.ID_VEL]
                else:
                    temp = self._levy_flight__(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])
                temp = self._amend_solution_faster__(temp)
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit, pop[i][self.ID_VEL], pop[i][self.ID_POS]]

            # Update the global best
            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)

            # Initialization process of CE
            pos_list = array([item[self.ID_POS] for item in pop])
            self.means = mean(pos_list, axis=0)
            means_new_repeat = repeat(self.means.reshape((1, -1)), self.pop_size, axis=0)
            self.stdevs = mean(((pos_list - means_new_repeat) ** 2), axis=0)

            ## Here for CE algorithm (Exploitation)
            for epoch_ce in range(self.epoch_ce):
                ## Selected the best samples and update means and stdevs
                pop_best = pop[:self.n_best]
                pos_list = array([item[self.ID_POS] for item in pop_best])

                means_new = mean(pos_list, axis=0)
                means_new_repeat = repeat(means_new.reshape((1, -1)), self.n_best, axis=0)
                stdevs_new = mean(((pos_list - means_new_repeat) ** 2), axis=0)

                self.means = self.alpha * self.means + (1.0 - self.alpha) * means_new
                self.stdevs = abs(self.alpha * self.stdevs + (1.0 - self.alpha) * stdevs_new)

                ## Update elite if a bower becomes fitter than the elite
                g_best = self._update_global_best__(pop_best, self.ID_MIN_PROB, g_best)

                ## Create new population for next generation
                pop = [self._create_solution_ce_(pop, idx) for idx in range(self.pop_size)]
                pop = sorted(pop, key=lambda item: item[self.ID_FIT])

            # Update the final global best
            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
