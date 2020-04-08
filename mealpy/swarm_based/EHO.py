#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:41, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, normal
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
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=700, pop_size=50, alpha=0.5, beta=0.1, n_clans=5):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha      # a factor that determines the influence of the best in each clan
        self.beta = beta        # a factor that determines the influence of the x_center
        self.n_clans = n_clans
        self.n_individuals = int(self.pop_size / self.n_clans)

    def _creat_population__(self):
        pop = []
        for i in range(0, self.n_clans):
            group = [self._create_solution__() for _ in range(0, self.n_individuals)]
            pop.append(group)
        return pop

    def _sort_clan_and_find_center__(self, pop):
        centers = []
        for i in range(0, self.n_clans):
            pop[i] = sorted(pop[i], key=lambda item: item[self.ID_FIT])
            center = mean(array([item[self.ID_POS] for item in pop[i]]), axis=0)
            centers.append(deepcopy(center))
        return pop, centers

    def _train__(self):
        pop = self._creat_population__()
        pop, centers = self._sort_clan_and_find_center__(pop)
        pop_best = [item[0] for item in pop]
        g_best = self._get_global_best__(pop_best, self.ID_FIT, self.ID_MIN_PROB)

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
                pos_new = self._amend_solution_faster__(pos_new)
                fit = self._fitness_model__(pos_new)
                if fit < pop[clan_idx][pos_clan_idx][self.ID_FIT]:
                    pop[clan_idx][pos_clan_idx] = [pos_new, fit]

            # Separating operator
            for i in range(0, self.n_clans):
                pop[i] = sorted(pop[i], key=lambda item: item[self.ID_FIT])
                sol_new = self._create_solution__()
                pop[i][-1] = sol_new

            ## Update the global best
            pop, centers = self._sort_clan_and_find_center__(pop)
            pop_best = [item[0] for item in pop]
            g_best = self._update_global_best__(pop_best, self.ID_MIN_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevyEHO(BaseEHO):
    """
    The levy version of: Elephant Herding Optimization (EHO)
        (Elephant Herding Optimization )
    Link:
        + Applied levy-flight
        + Replace Uniform distribution by Gaussian distribution
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=700, pop_size=50, alpha=0.5, beta=0.1, n_clans=5):
        BaseEHO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size, alpha, beta, n_clans)

    def _train__(self):
        pop = self._creat_population__()
        pop, centers = self._sort_clan_and_find_center__(pop)
        pop_best = [item[0] for item in pop]
        g_best = self._get_global_best__(pop_best, self.ID_FIT, self.ID_MIN_PROB)

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
                pos_new = self._amend_solution_faster__(pos_new)
                fit = self._fitness_model__(pos_new)
                if fit < pop[clan_idx][pos_clan_idx][self.ID_FIT]:
                    pop[clan_idx][pos_clan_idx] = [pos_new, fit]

            # Separating operator
            for i in range(0, self.n_clans):
                pop[i] = sorted(pop[i], key=lambda item: item[self.ID_FIT])
                if uniform() < 0.5:
                    pos_new = normal(0, 1, self.problem_size)
                else:
                    pos_new = self._levy_flight__(epoch, pop[i][-1][self.ID_POS], g_best[self.ID_POS])
                pos_new = self._amend_solution_faster__(pos_new)
                fit = self._fitness_model__(pos_new)
                pop[i][-1] = [pos_new, fit]

            ## Update the global best
            pop, centers = self._sort_clan_and_find_center__(pop)
            pop_best = [item[0] for item in pop]
            g_best = self._update_global_best__(pop_best, self.ID_MIN_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
