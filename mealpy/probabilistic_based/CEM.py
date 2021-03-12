#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:08, 19/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import ceil, sqrt, abs, array, mean, repeat, sin, cos, clip, where
from numpy.random import uniform, normal, choice, randint
from copy import deepcopy
from mealpy.root import Root
from mealpy.human_based.LCBO import BaseLCBO
from mealpy.bio_based.SBO import BaseSBO


class BaseCEM(Root):
    """
        The original version of: Cross-Entropy Method (CEM)
            https://github.com/clever-algorithms/CleverAlgorithms
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, n_best=30, alpha=0.7, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha
        self.n_best = n_best
        self.means, self.stdevs = None, None

    def _create_solution_ce__(self, minmax=0):
        pos = normal(self.means, self.stdevs, self.problem_size)
        pos = self.amend_position_random_faster(pos)
        fit = self.get_fitness_position(pos, minmax=minmax)
        return [pos, fit]

    def train(self):
        self.means = uniform(self.lb, self.ub)
        self.stdevs = abs(self.ub - self.lb)
        pop = [self._create_solution_ce__() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

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
            g_best = self.update_global_best_solution(pop_best, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

            ## Create new population for next generation
            pop = [self._create_solution_ce__() for _ in range(self.pop_size)]
            pop = sorted(pop, key=lambda item: item[self.ID_FIT])

        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class CEBaseLCBO(BaseLCBO):
    """
        The hybrid version of: Cross-Entropy Method (CEM) and Life Choice-Based Optimization
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, alpha=0.7, r1=2.35, **kwargs):
        BaseLCBO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, r1, kwargs = kwargs)
        self.n1 = int(ceil(sqrt(self.pop_size)))                    # n best position in LCBO
        self.n2 = self.n1 + int((self.pop_size - self.n1) / 2)      # 50% for both 2 group left
        self.n_best = int(sqrt(self.pop_size))                      # n nest position in CE
        self.alpha = alpha                                          # alpha in CE
        self.epoch_ce = int(sqrt(epoch))                            # Epoch in CE
        self.means, self.stdevs = None, None

    def _create_solution_ce__(self, minmax=0):
        pos = normal(self.means, self.stdevs, self.problem_size)
        pos = self.amend_position_random_faster(pos)
        fit = self.get_fitness_position(pos, minmax=minmax)
        return [pos, fit]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

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
                    temp = self.levy_flight(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])
                else:  # People in group 2 learning from the current best person and the person slightly better than them, because they don't have vision
                    f1 = 1 - (epoch + 1) / self.epoch
                    f2 = 1 - f1
                    better_diff = f2 * self.r1 * (pop[i - 1][self.ID_POS] - pop[i][self.ID_POS])
                    best_diff = f1 * self.r1 * (pop[0][self.ID_POS] - pop[i][self.ID_POS])
                    temp = pop[i][self.ID_POS] + uniform() * better_diff + uniform() * best_diff
                fit = self.get_fitness_position(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]
            # Update the global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            # Initialization process of CE
            pop_ce = deepcopy(pop)
            pos_list = array([item[self.ID_POS] for item in pop])
            self.means = mean(pos_list, axis=0)
            means_new_repeat = repeat(self.means.reshape((1, -1)), self.pop_size, axis=0)
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
                g_best = self.update_global_best_solution(pop_best, self.ID_MIN_PROB, g_best)

                ## Create new population for next generation
                pop_ce = [self._create_solution_ce__() for _ in range(self.n_best)]
                pop_ce = sorted(pop_ce, key=lambda item: item[self.ID_FIT])

            ## Replace the worst in pop by pop_ce
            pop = pop[:(self.pop_size - self.n_best)] + pop_ce

            # Update the final global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class CEBaseLCBONew(BaseLCBO):
    """
        The hybrid version of: Cross-Entropy Method (CEM) and Life Choice-Based Optimization
        Version 2: Instead replace the old population, now it will replace only few worst individuals
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, alpha=0.7, r1=2.35, **kwargs):
        BaseLCBO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, r1, kwargs=kwargs)
        self.n1 = int(ceil(sqrt(self.pop_size)))                # n best position in LCBO
        self.n2 = self.n1 + int((self.pop_size - self.n1) / 2)  # 50% for both 2 group left
        self.n_best = int(sqrt(pop_size))                       # n best position in CE
        self.alpha = alpha                                      # alpha in CE
        self.epoch_ce = int(sqrt(epoch))                        # Epoch in CE
        self.means, self.stdevs = None, None

    def _create_solution_ce__(self, minmax=0):
        pos = normal(self.means, self.stdevs, self.problem_size)
        pos = self.amend_position_random_faster(pos)
        fit = self.get_fitness_position(pos, minmax=minmax)
        return [pos, fit]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

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
                    temp = self.ub - (pop[i][self.ID_POS] - self.lb) * uniform(self.lb, self.ub)
                fit = self.get_fitness_position(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]
            # Update the global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            pop_ce = deepcopy(pop)
            # Initialization process of CE
            pos_list = array([item[self.ID_POS].copy() for item in pop_ce])
            self.means = mean(pos_list, axis=0)
            means_new_repeat = repeat(self.means.reshape((1, -1)), self.pop_size, axis=0)
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
                g_best = self.update_global_best_solution(pop_best, self.ID_MIN_PROB, g_best)

                ## Create new population for next generation
                pop_ce = [self._create_solution_ce__() for _ in range(self.n_best)]
                pop_ce = sorted(pop_ce, key=lambda item: item[self.ID_FIT])

            ## Replace the worst in pop by pop_ce
            pop = pop[:(self.pop_size - self.n_best)] + pop_ce

            # Update the final global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class CEBaseSSDO(Root):
    """
        The hybrid version of: Cross-Entropy Method (CEM) and Social Sky-Driving Optimization
    """
    ID_POS = 0
    ID_FIT = 1
    ID_VEL = 2  # velocity
    ID_LBS = 3  # local best position

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, alpha=0.7, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.n_best = int(sqrt(self.pop_size))              # n nest position in CE
        self.alpha = alpha                                  # alpha in CE
        self.epoch_ce = int(sqrt(epoch))                    # Epoch in CE
        self.means, self.stdevs = None, None

    def create_solution(self, minmax=0):
        position = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position, minmax=minmax)
        velocity = uniform(self.lb, self.ub)
        local_best_solution = deepcopy(position)
        return [position, fitness, velocity, local_best_solution]

    def _create_solution_ce_(self, pop, idx):
        pos = normal(self.means, self.stdevs, self.problem_size)
        pos = self.amend_position_random_faster(pos)
        fit = self.get_fitness_position(pos)
        velocity = pop[idx][self.ID_VEL]
        local_best_solution = pop[idx][self.ID_LBS]
        return [pos, fit, velocity, local_best_solution]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            c = 2 - epoch * (2.0 / self.epoch)  # a decreases linearly from 2 to 0

            ## Calculate the mean of the best three solutions in each dimension. Eq 9
            pos_mean = mean([indi[self.ID_POS] for indi in pop[:3]], axis=0)

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
                    temp = uniform() * pop[i][self.ID_POS] + pop[i][self.ID_VEL]
                else:
                    temp = self.levy_flight(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])
                temp = self.amend_position_faster(temp)
                fit = self.get_fitness_position(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i][self.ID_LBS] = deepcopy(temp)
                pop[i][self.ID_POS] = deepcopy(temp)
                pop[i][self.ID_FIT] = fit

            # Update the global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            pop_ce = deepcopy(pop)
            # Initialization process of CE
            pos_list = array([item[self.ID_POS] for item in pop_ce])
            self.means = mean(pos_list, axis=0)
            means_new_repeat = repeat(self.means.reshape((1, -1)), self.pop_size, axis=0)
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
                g_best = self.update_global_best_solution(pop_best, self.ID_MIN_PROB, g_best)

                ## Create new population for next generation
                pop_ce = [self._create_solution_ce_(pop_ce, idx) for idx in range(self.n_best)]
                pop_ce = sorted(pop_ce, key=lambda item: item[self.ID_FIT])

            ## Replace the worst in pop by pop_ce
            pop = pop[:(self.pop_size - self.n_best)] + pop_ce
            # Update the final global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class CEBaseSBO(BaseSBO):
    """
        The hybrid version of: Cross-Entropy Method (CEM) and Satin Bowerbird Optimizer (SBO)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, alpha=0.94, pm=0.05, z=0.02, **kwargs):
        BaseSBO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, alpha, pm, z, kwargs=kwargs)
        self.n_best = int(sqrt(self.pop_size))          # n nest position in CE
        self.alpha = alpha                              # alpha in CE
        self.epoch_ce = int(sqrt(epoch))                # Epoch in CE
        self.means, self.stdevs = None, None

    def _create_solution_ce_(self):
        pos = normal(self.means, self.stdevs, self.problem_size)
        pos = self.amend_position_random_faster(pos)
        fit = self.get_fitness_position(pos)
        return [pos, fit]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):

            ## Calculate the probability of bowers using my equation
            fit_list = array([item[self.ID_FIT] for item in pop])

            for i in range(0, self.pop_size):
                ### Select a bower using roulette wheel
                idx = self.get_index_roulette_wheel_selection(fit_list)
                ### Calculating Step Size
                lamda = self.alpha * uniform()
                pos_new = pop[i][self.ID_POS] + lamda * ((pop[idx][self.ID_POS] + g_best[self.ID_POS]) / 2 - pop[i][self.ID_POS])
                ### Mutation
                temp = pop[i][self.ID_POS] + normal(0, 1, self.problem_size) * self.sigma
                pos_new = where(uniform(0, 1, self.problem_size) < self.p_m, temp, pos_new)
                ### In-bound position
                pos_new = clip(pos_new, self.lb, self.ub)
                fit = self.get_fitness_position(pos_new)
                pop[i] = [pos_new, fit]

            ## Update elite if a bower becomes fitter than the elite
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            pop_ce = deepcopy(pop)
            # Initialization process of CE
            pos_list = array([item[self.ID_POS] for item in pop_ce])
            self.means = mean(pos_list, axis=0)
            means_new_repeat = repeat(self.means.reshape((1, -1)), self.pop_size, axis=0)
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
                g_best = self.update_global_best_solution(pop_best, self.ID_MIN_PROB, g_best)

                ## Create new population for next generation
                pop_ce = [self._create_solution_ce_() for idx in range(self.n_best)]
                pop_ce = sorted(pop_ce, key=lambda item: item[self.ID_FIT])

            ## Replace the worst in pop by pop_ce
            pop = pop[:(self.pop_size - self.n_best)] + pop_ce
            # Update the final global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class CEBaseFBIO(Root):
    """
    My hybrid version of: Cross-Entropy and Forensic-Based Investigation Optimization (CE-FBIO)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.n_best = int(sqrt(self.pop_size))  # n nest position in CE
        self.alpha = 0.94                       # alpha in CE
        self.epoch_ce = int(sqrt(epoch))        # Epoch in CE
        self.means, self.stdevs = None, None

    def _create_solution_ce_(self):
        pos = normal(self.means, self.stdevs, self.problem_size)
        pos = self.amend_position_random_faster(pos)
        fit = self.get_fitness_position(pos)
        return [pos, fit]

    def probability(self, list_fitness=None):  # Eq.(3) in FBI Inspired Meta-Optimization
        max1 = max(list_fitness)
        min1 = min(list_fitness)
        prob = (max1 - list_fitness) / (max1 - min1 + self.EPSILON)
        return prob

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Optimization Cycle
        for epoch in range(self.epoch):
            # Investigation team - team A
            # Step A1
            for i in range(0, self.pop_size):
                n_change = randint(0, self.problem_size)
                nb1, nb2 = choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                # Eq.(2) in FBI Inspired Meta - Optimization
                pos_a = deepcopy(pop[i][self.ID_POS])
                pos_a[n_change] = pop[i][self.ID_POS][n_change] + normal() * (pop[i][self.ID_POS][n_change] -
                                                                              (pop[nb1][self.ID_POS][n_change] + pop[nb2][self.ID_POS][n_change]) / 2)
                pos_a = self.amend_position_random_faster(pos_a)
                fit_a = self.get_fitness_position(pos_a)
                if fit_a < pop[i][self.ID_FIT]:
                    pop[i] = [pos_a, fit_a]
                    if fit_a < g_best[self.ID_FIT]:
                        g_best = [pos_a, fit_a]
            # Step A2
            list_fitness = array([item[self.ID_FIT] for item in pop])
            prob = self.probability(list_fitness)
            for i in range(0, self.pop_size):
                if uniform() > prob[i]:
                    r1, r2, r3 = choice(list(set(range(0, self.pop_size)) - {i}), 3, replace=False)
                    ## Remove third loop here, the condition also not good, need to remove also. No need Rnd variable
                    pos_a = deepcopy(pop[i][self.ID_POS])
                    temp = g_best[self.ID_POS] + pop[r1][self.ID_POS] + uniform() * (pop[r2][self.ID_POS] - pop[r3][self.ID_POS])
                    pos_a = where(uniform(0, 1, self.problem_size) < 0.5, temp, pos_a)
                    pos_a = self.amend_position_random_faster(pos_a)
                    fit_a = self.get_fitness_position(pos_a)
                    if fit_a < pop[i][self.ID_FIT]:
                        pop[i] = [pos_a, fit_a]
                        if fit_a < g_best[self.ID_FIT]:
                            g_best = [pos_a, fit_a]
            ## Persuing team - team B
            ## Step B1
            for i in range(0, self.pop_size):
                ### Remove third loop here also
                ### Eq.(6) in FBI Inspired Meta-Optimization
                pos_b = uniform(0, 1, self.problem_size) * pop[i][self.ID_POS] + uniform(0, 1, self.problem_size) * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                pos_b = self.amend_position_random_faster(pos_b)
                fit_b = self.get_fitness_position(pos_b)
                if fit_b < pop[i][self.ID_FIT]:
                    pop[i] = [pos_b, fit_b]
                    if fit_b < g_best[self.ID_FIT]:
                        g_best = [pos_b, fit_b]

            ## Step B2
            for i in range(0, self.pop_size):
                rr = choice(list(set(range(0, self.pop_size)) - {i}))
                ## Eq.(7) + Eq. (8) in FBI Inspired Meta-Optimization
                pos_b = pop[i][self.ID_POS] + normal(0, 1, self.problem_size) * (pop[rr][self.ID_POS] - pop[i][self.ID_POS]) + \
                        uniform() * (g_best[self.ID_POS] - pop[rr][self.ID_POS])
                pos_b = self.amend_position_random_faster(pos_b)
                fit_b = self.get_fitness_position(pos_b)
                if fit_b < pop[i][self.ID_FIT]:
                    pop[i] = [pos_b, fit_b]
                    if fit_b < g_best[self.ID_FIT]:
                        g_best = [pos_b, fit_b]

            ## Update elite if a bower becomes fitter than the elite
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            pop_ce = deepcopy(pop)
            # Initialization process of CE
            pos_list = array([item[self.ID_POS] for item in pop_ce])
            self.means = mean(pos_list, axis=0)
            means_new_repeat = repeat(self.means.reshape((1, -1)), self.pop_size, axis=0)
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
                g_best = self.update_global_best_solution(pop_best, self.ID_MIN_PROB, g_best)

                ## Create new population for next generation
                pop_ce = [self._create_solution_ce_() for _ in range(self.n_best)]
                pop_ce = sorted(pop_ce, key=lambda item: item[self.ID_FIT])

            ## Replace the worst in pop by pop_ce
            pop = pop[:(self.pop_size - self.n_best)] + pop_ce
            # Update the final global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class CEBaseFBIONew(Root):
    """
    My hybrid version of: Cross-Entropy and Forensic-Based Investigation Optimization (CE-FBIO)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.n_best = int(sqrt(self.pop_size))      # n nest position in CE
        self.alpha = 0.94                           # alpha in CE
        self.epoch_ce = int(sqrt(epoch))            # Epoch in CE
        self.means, self.stdevs = None, None

    def _create_solution_ce_(self):
        pos = normal(self.means, self.stdevs, self.problem_size)
        pos = self.amend_position_random_faster(pos)
        fit = self.get_fitness_position(pos)
        return [pos, fit]

    def probability(self, list_fitness=None):  # Eq.(3) in FBI Inspired Meta-Optimization
        max1 = max(list_fitness)
        min1 = min(list_fitness)
        prob = (max1 - list_fitness) / (max1 - min1 + self.EPSILON)
        return prob

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Optimization Cycle
        for epoch in range(self.epoch):
            # Investigation team - team A
            # Step A1
            for i in range(0, self.pop_size):
                n_change = randint(0, self.problem_size)
                nb1, nb2 = choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                # Eq.(2) in FBI Inspired Meta - Optimization
                pos_a = deepcopy(pop[i][self.ID_POS])
                pos_a[n_change] = pop[i][self.ID_POS][n_change] + normal() * (pop[i][self.ID_POS][n_change] -
                                                                              (pop[nb1][self.ID_POS][n_change] + pop[nb2][self.ID_POS][n_change]) / 2)
                pos_a = self.amend_position_random_faster(pos_a)
                fit_a = self.get_fitness_position(pos_a)
                if fit_a < pop[i][self.ID_FIT]:
                    pop[i] = [pos_a, fit_a]
                    if fit_a < g_best[self.ID_FIT]:
                        g_best = [pos_a, fit_a]
            # Step A2
            list_fitness = array([item[self.ID_FIT] for item in pop])
            prob = self.probability(list_fitness)
            for i in range(0, self.pop_size):
                if uniform() > prob[i]:
                    r1, r2, r3 = choice(list(set(range(0, self.pop_size)) - {i}), 3, replace=False)
                    ## Remove third loop here, the condition also not good, need to remove also. No need Rnd variable
                    pos_a = deepcopy(pop[i][self.ID_POS])
                    temp = g_best[self.ID_POS] + pop[r1][self.ID_POS] + uniform() * (pop[r2][self.ID_POS] - pop[r3][self.ID_POS])
                    pos_a = where(uniform(0, 1, self.problem_size) < 0.5, temp, pos_a)
                    pos_a = self.amend_position_random_faster(pos_a)
                    fit_a = self.get_fitness_position(pos_a)
                    if fit_a < pop[i][self.ID_FIT]:
                        pop[i] = [pos_a, fit_a]
                        if fit_a < g_best[self.ID_FIT]:
                            g_best = [pos_a, fit_a]
            ## Persuing team - team B
            ## Step B1
            for i in range(0, self.pop_size):
                ### Remove third loop here also
                ### Eq.(6) in FBI Inspired Meta-Optimization
                pos_b = uniform(0, 1, self.problem_size) * pop[i][self.ID_POS] + uniform(0, 1, self.problem_size) * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                pos_b = self.amend_position_random_faster(pos_b)
                fit_b = self.get_fitness_position(pos_b)
                if fit_b < pop[i][self.ID_FIT]:
                    pop[i] = [pos_b, fit_b]
                    if fit_b < g_best[self.ID_FIT]:
                        g_best = [pos_b, fit_b]

            ## Step B2
            for i in range(0, self.pop_size):
                rr = choice(list(set(range(0, self.pop_size)) - {i}))
                ## Eq.(7) + Eq. (8) in FBI Inspired Meta-Optimization
                pos_b = pop[i][self.ID_POS] + normal(0, 1, self.problem_size) * (pop[rr][self.ID_POS] - pop[i][self.ID_POS]) + \
                        uniform() * (g_best[self.ID_POS] - pop[rr][self.ID_POS])
                pos_b = self.amend_position_random_faster(pos_b)
                fit_b = self.get_fitness_position(pos_b)
                if fit_b < pop[i][self.ID_FIT]:
                    pop[i] = [pos_b, fit_b]
                    if fit_b < g_best[self.ID_FIT]:
                        g_best = [pos_b, fit_b]

            ##  Initialization process of CE
            pos_list = array([item[self.ID_POS] for item in pop])
            self.means = mean(pos_list, axis=0)
            means_new_repeat = repeat(self.means.reshape((1, -1)), self.pop_size, axis=0)
            self.stdevs = mean(((pos_list - means_new_repeat) ** 2), axis=0)
            ## Selected the best samples for CEM
            pop_ce = deepcopy(pop[:self.n_best])

            ## Here for CE algorithm (Exploitation)
            for epoch_ce in range(self.epoch_ce):
                ## Update means and stdevs
                pos_list = array([item[self.ID_POS] for item in pop_ce])
                means_new = mean(pos_list, axis=0)
                means_new_repeat = repeat(means_new.reshape((1, -1)), self.n_best, axis=0)
                stdevs_new = mean(((pos_list - means_new_repeat) ** 2), axis=0)

                self.means = self.alpha * self.means + (1.0 - self.alpha) * means_new
                self.stdevs = abs(self.alpha * self.stdevs + (1.0 - self.alpha) * stdevs_new)

                ## Update elite if a bower becomes fitter than the elite
                g_best = self.update_global_best_solution(pop_ce, self.ID_MIN_PROB, g_best)

                ## Create new population for next generation
                pop_ce = [self._create_solution_ce_() for _ in range(self.n_best)]
                pop_ce = sorted(pop_ce, key=lambda item: item[self.ID_FIT])

            ## Replace the worst in pop by pop_ce
            pop = pop[:(self.pop_size - self.n_best)] + pop_ce
            # Update the final global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

