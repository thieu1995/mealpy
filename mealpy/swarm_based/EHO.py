#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:41, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform
from numpy import array, mean
from copy import deepcopy
from mealpy.root import Root


class BaseEHO(Root):
    """
    The original version of: Elephant Herding Optimization (EHO)
        (Elephant Herding Optimization )
    Link:
        https://doi.org/10.1109/ISCBI.2015.8
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 alpha=0.5, beta=0.5, n_clans=5, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha              # a factor that determines the influence of the best in each clan
        self.beta = beta                # a factor that determines the influence of the x_center
        self.n_clans = n_clans
        self.n_individuals = int(self.pop_size / self.n_clans)

    def _creat_population__(self):
        pop = []
        for i in range(0, self.n_clans):
            group = [self.create_solution() for _ in range(0, self.n_individuals)]
            pop.append(group)
        return pop

    def _sort_clan_and_find_center__(self, pop=None):
        centers = []
        for i in range(0, self.n_clans):
            pop[i] = sorted(pop[i], key=lambda item: item[self.ID_FIT])
            center = mean(array([item[self.ID_POS] for item in pop[i]]), axis=0)
            centers.append(deepcopy(center))
        return pop, centers

    def train(self):
        pop = self._creat_population__()
        pop, centers = self._sort_clan_and_find_center__(pop)
        pop_best = [item[0] for item in pop]
        g_best = self.get_global_best_solution(pop_best, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):

            # Clan updating operator
            for i in range(0, self.pop_size):
                clan_idx = int(i / self.n_individuals)
                pos_clan_idx = int(i % self.n_individuals)

                if pos_clan_idx == 0:       # The best in clan, because all clans are sorted based on fitness
                    pos_new = self.beta * centers[clan_idx]
                else:
                    pos_new = pop[clan_idx][pos_clan_idx][self.ID_POS] + self.alpha * uniform() * \
                                    (pop[clan_idx][0][self.ID_POS] - pop[clan_idx][pos_clan_idx][ self.ID_POS])
                pos_new = self.amend_position_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[clan_idx][pos_clan_idx][self.ID_FIT]:
                    pop[clan_idx][pos_clan_idx] = [pos_new, fit]

            # Separating operator
            for i in range(0, self.n_clans):
                pop[i] = sorted(pop[i], key=lambda item: item[self.ID_FIT])
                sol_new = self.create_solution()
                pop[i][-1] = sol_new

            ## Update the global best
            pop, centers = self._sort_clan_and_find_center__(pop)
            pop_best = [item[0] for item in pop]
            g_best = self.update_global_best_solution(pop_best, self.ID_MIN_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevyEHO(BaseEHO):
    """
    The levy version of: Elephant Herding Optimization (EHO)
        (Elephant Herding Optimization )
    Link:
        + Applied levy-flight
        + Using global best solution 117789
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 alpha=0.5, beta=0.5, n_clans=5, **kwargs):
        BaseEHO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, alpha, beta, n_clans, kwargs=kwargs)

    def train(self):
        pop = self._creat_population__()
        pop, centers = self._sort_clan_and_find_center__(pop)
        pop_best = [item[0] for item in pop]
        g_best = self.get_global_best_solution(pop_best, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):

            # Clan updating operator
            for i in range(0, self.pop_size):
                clan_idx = int(i / self.n_individuals)
                pos_clan_idx = int(i % self.n_individuals)

                if pos_clan_idx == 0:  # The best in clan, because all clans are sorted based on fitness
                    pos_new = self.beta * centers[clan_idx]
                else:
                    pos_new = pop[clan_idx][pos_clan_idx][self.ID_POS] + self.alpha * uniform() * \
                              (pop[clan_idx][0][self.ID_POS] - pop[clan_idx][pos_clan_idx][self.ID_POS])
                pos_new = self.amend_position_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[clan_idx][pos_clan_idx][self.ID_FIT]:
                    pop[clan_idx][pos_clan_idx] = [pos_new, fit]

            # Separating operator
            for i in range(0, self.n_clans):
                pop[i] = sorted(pop[i], key=lambda item: item[self.ID_FIT])
                if uniform() < 0.5:
                    pos_new = uniform(self.lb, self.ub)
                else:
                    pos_new = self.levy_flight(epoch, pop[i][-1][self.ID_POS], g_best[self.ID_POS])
                pos_new = self.amend_position_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                pop[i][-1] = [pos_new, fit]

            ## Update the global best
            pop, centers = self._sort_clan_and_find_center__(pop)
            pop_best = [item[0] for item in pop]
            g_best = self.update_global_best_solution(pop_best, self.ID_MIN_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
