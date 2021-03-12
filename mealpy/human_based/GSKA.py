#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:58, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice, normal
from numpy import ceil
from copy import deepcopy
from mealpy.root import Root


class BaseGSKA(Root):
    """
    My version of: Gaining Sharing Knowledge-based Algorithm (GSKA)
        (Gaining‑sharing Knowledge-Based Algorithm For Solving Optimization Problems: A Novel Nature‑inspired Algorithm)
    Notes:
        + Remove all third loop
        + Solution represent junior or senior instead of dimension of solution
        + Remove 2 parameters
        + Change some equations for large-scale optimization
        + Apply the ideas of levy-flight and global best
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, p=0.1, kr=0.7, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size    # n: pop_size, m: clusters
        self.p = p                  # percent of the best   0.1%, 0.8%, 0.1%
        self.kr = kr                # knowledge ratio

    def train(self):
        pop = [self.create_solution() for _ in range(0, self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            D = int(ceil(self.pop_size * (1 - (epoch + 1) / self.epoch)))
            for i in range(1, self.pop_size - 1):
                if i < D:  # senior gaining and sharing
                    if uniform() <= self.kr:
                        rand_idx = choice(list(set(range(0, self.pop_size)) - {i - 1, i, i + 1}))
                        if pop[i][self.ID_FIT] > pop[rand_idx][self.ID_FIT]:
                            pos_new = pop[i][self.ID_POS] + uniform(0, 1, self.problem_size) * \
                                      (pop[i - 1][self.ID_POS] - pop[i + 1][self.ID_POS] + pop[rand_idx][self.ID_POS] - pop[i][self.ID_POS])
                        else:
                            pos_new = g_best[self.ID_POS] + uniform(0, 1, self.problem_size) * (pop[rand_idx][self.ID_POS] - pop[i][self.ID_POS])
                    else:
                        pos_new = uniform(self.lb, self.ub)
                else:  # junior gaining and sharing
                    if uniform() <= self.kr:
                        id1 = int(self.p * self.pop_size)
                        id2 = id1 + int(self.pop_size - 2 * 100 * self.p)
                        rand_best = choice(list(set(range(0, id1)) - {i}))
                        rand_worst = choice(list(set(range(id2, self.pop_size)) - {i}))
                        rand_mid = choice(list(set(range(id1, id2)) - {i}))
                        if pop[i][self.ID_FIT] > pop[rand_mid][self.ID_FIT]:
                            pos_new = pop[i][self.ID_POS] + uniform(0, 1, self.problem_size) * \
                                      (pop[rand_best][self.ID_POS] - pop[rand_worst][self.ID_POS] + pop[rand_mid][self.ID_POS] - pop[i][self.ID_POS])
                        else:
                            pos_new = g_best[self.ID_POS] + uniform(0, 1, self.problem_size) * (pop[rand_mid][self.ID_FIT] - pop[i][self.ID_POS])
                    else:
                        pos_new = self.levy_flight(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])
                pos_new = self.amend_position_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]

            ## Sort the population and update the global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalGSKA(Root):
    """
    The original version of: Gaining Sharing Knowledge-based Algorithm (GSKA)
        (Gaining‑sharing Knowledge-Based Algorithm For Solving Optimization Problems: A Novel Nature‑inspired Algorithm)
    Link:
        DOI: https://doi.org/10.1007/s13042-019-01053-x
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 p=0.1, kf=0.5, kr=0.9, k=10, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size    # n: pop_size, m: clusters
        self.p = p                  # percent of the best   0.1%, 0.8%, 0.1%
        self.kf = kf                # knowledge factor that controls the total amount of gained and shared knowledge added from others to the current
                                    # individuals during generations
        self.kr = kr                # knowledge ratio
        self.k = k                  # KNOWLEDGE rate

    def train(self):
        pop = [self.create_solution() for _ in range(0, self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            D = int(self.problem_size * (1 - (epoch+1)/self.epoch) ** self.k)
            for i in range(self.pop_size):
                # If it is the best it chooses best+2, best+1
                if i == 0:
                    previ = i+2
                    nexti = i+1
                # If it is the worse it chooses worst-2, worst-1
                elif i == self.pop_size-1:
                    previ = i-2
                    nexti = i-1
                # Other case it chooses i-1, i+1
                else:
                    previ = i-1
                    nexti = i+1

                # The random individual is for all dimension values
                rand_idx = choice(list(set(range(0, self.pop_size)) - {previ, i, nexti}))

                pos_new = deepcopy(pop[i][self.ID_POS])
                for j in range(0, self.problem_size):
                    if j < D:                       # junior gaining and sharing
                        if uniform() <= self.kr:
                            if pop[i][self.ID_FIT] > pop[rand_idx][self.ID_FIT]:
                                pos_new[j] = pop[i][self.ID_POS][j] + self.kf * \
                                          (pop[previ][self.ID_POS][j] - pop[nexti][self.ID_POS][j] + pop[rand_idx][self.ID_POS][j] - pop[i][self.ID_POS][j])
                            else:
                                pos_new[j] = pop[i][self.ID_POS][j] + self.kf * \
                                          (pop[previ][self.ID_POS][j] - pop[nexti][self.ID_POS][j] + pop[i][self.ID_POS][j] - pop[rand_idx][self.ID_POS][j])
                    else:                           # senior gaining and sharing
                        if uniform() <= self.kr:
                            id1 = int(self.p * self.pop_size)
                            id2 = id1 + int(self.pop_size - 2 * 100 * self.p)
                            rand_best = choice(list(set(range(0, id1)) - {i}))
                            rand_worst = choice(list(set(range(id2, self.pop_size)) - {i}))
                            rand_mid = choice(list(set(range(id1, id2)) - {i}))
                            if pop[i][self.ID_FIT] > pop[rand_mid][self.ID_FIT]:
                                pos_new[j] = pop[i][self.ID_POS][j] + self.kf * \
                                    (pop[rand_best][self.ID_POS][j] - pop[rand_worst][self.ID_POS][j] + pop[rand_mid][self.ID_POS][j] - pop[i][self.ID_POS][j])
                            else:
                                pos_new[j] = pop[i][self.ID_POS][j] + self.kf * \
                                    (pop[rand_best][self.ID_POS][j] - pop[rand_worst][self.ID_POS][j] + pop[i][self.ID_POS][j] - pop[rand_mid][self.ID_POS][j])
                pos_new = self.amend_position_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]

            ## Sort the population and update the global best
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

