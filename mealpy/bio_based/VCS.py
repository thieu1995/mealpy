#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 22:07, 11/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import sum, log1p, array, mean, prod, abs, where
from numpy.random import uniform, normal, choice
from copy import deepcopy
from mealpy.root import Root


class BaseVCS(Root):
    """
        My version of: Virus Colony Search (VCS)
            A Novel Nature-inspired Algorithm For Optimization: Virus Colony Search
        Notes:
            + Remove all third loop, make algrithm 10 times faster than original
            + In Immune response process, updating whole position instead of updating each variable in position
            + Drop batch-size idea to 3 main process of this algorithm, make it more robust
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, lamda=0.5, xichma=0.3, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.xichma = xichma        # Weight factor
        self.lamda = lamda          # Number of the best will keep
        if lamda < 1:
            self.n_best = int(lamda * self.pop_size)
        else:
            self.n_best = int(lamda)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        pos_list = [item[self.ID_POS] for item in pop]
        x_mean = mean(pos_list, axis=0)

        for epoch in range(self.epoch):
            ## Viruses diffusion
            for i in range(0, self.pop_size):
                xichma = (log1p(epoch + 1) / self.epoch) * (pop[i][self.ID_POS] - g_best[self.ID_POS])
                gauss = normal(normal(g_best[self.ID_POS], abs(xichma)))
                pos_new = gauss + uniform() * g_best[self.ID_POS] - uniform() * pop[i][self.ID_POS]
                pos_new = self.amend_position_random_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]

                # Batch-size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            ## Host cells infection
            xichma = self.xichma * (1 - (epoch + 1) / self.epoch)
            for i in range(0, self.pop_size):
                pos_new = x_mean + xichma * normal(0, 1, self.problem_size)  ## Basic / simple version, not the original version in the paper
                pos_new = self.amend_position_random_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]

                # Batch-size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            ## Calculate the weighted mean of the λ best individuals by
            pop = sorted(pop, key=lambda item: item[self.ID_FIT])
            pos_list = [item[self.ID_POS] for item in pop[:self.n_best]]

            factor_down = self.n_best * log1p(self.n_best + 1) - log1p(prod(range(1, self.n_best + 1)))
            weight = log1p(self.n_best + 1) / factor_down
            weight = weight / self.n_best
            x_mean = weight * sum(pos_list, axis=0)

            ## Immune response
            for i in range(0, self.pop_size):
                pr = (self.problem_size - i + 1) / self.problem_size

                id1, id2 = choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                temp = pop[id1][self.ID_POS] - (pop[id2][self.ID_POS] - pop[i][self.ID_POS]) * uniform()
                pos_new = deepcopy(pop[i][self.ID_POS])
                pos_new = where(uniform(0, 1, self.problem_size) < pr, pos_new, temp)

                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]

                # Batch-size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            ## Update elite if a bower becomes fitter than the elite
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalVCS(Root):
    """
        The original version of: Virus Colony Search (VCS)
            A Novel Nature-inspired Algorithm For Optimization: Virus Colony Search
            - This is basic version, not the full version of the paper
        Link:
            https://doi.org/10.1016/j.advengsoft.2015.11.004
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, lamda=0.5, xichma=0.3, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.xichma = xichma                # Weight factor
        self.lamda = lamda                  # Number of the best will keep
        if lamda < 1:
            self.n_best = int(lamda * self.pop_size)
        else:
            self.n_best = int(lamda)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        pos_list = [item[self.ID_POS] for item in pop]
        x_mean = mean(pos_list, axis=0)

        for epoch in range(self.epoch):
            ## Viruses diffusion
            for i in range(0, self.pop_size):
                xichma = (log1p(epoch + 1) / self.epoch) * (pop[i][self.ID_POS] - g_best[self.ID_POS])
                gauss = array([normal(g_best[self.ID_POS][idx], abs(xichma[idx])) for idx in range(0, self.problem_size)])
                pos_new = gauss + uniform() * g_best[self.ID_POS] - uniform() * pop[i][self.ID_POS]
                pos_new = self.amend_position_random_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]

            ## Host cells infection
            xichma = self.xichma * (1 - (epoch+1)/self.epoch)
            for i in range(0, self.pop_size):
                pos_new = x_mean + xichma * normal(0, 1, self.problem_size)         ## Basic / simple version, not the original version in the paper
                pos_new = self.amend_position_random_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]

            ## Calculate the weighted mean of the λ best individuals by
            pop = sorted(pop, key=lambda item: item[self.ID_FIT])
            pos_list = [item[self.ID_POS] for item in pop[:self.n_best]]

            factor_down = self.n_best * log1p(self.n_best + 1) - log1p(prod(range(1, self.n_best + 1)))
            weight = log1p(self.n_best + 1) / factor_down
            weight = weight / self.n_best
            x_mean = weight * sum(pos_list, axis=0)

            ## Immune response
            for i in range(0, self.pop_size):
                pr = (self.problem_size - i + 1) / self.problem_size
                pos_new = pop[i][self.ID_POS]
                for j in range(0, self.problem_size):
                    if uniform() > pr:
                        id1, id2 = choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                        pos_new[j] = pop[id1][self.ID_POS][j] - (pop[id2][self.ID_POS][j] - pop[i][self.ID_POS][j]) * uniform()
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]

            ## Update elite if a bower becomes fitter than the elite
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
