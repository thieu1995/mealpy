#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:17, 09/11/2020                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import uniform
from numpy import ceil, log10, round, reshape, array, argmin, nonzero, std, clip, maximum, minimum, mean
from scipy.spatial.distance import cdist
from copy import deepcopy
from mealpy.root import Root


class BaseBRO(Root):
    """
        My best version of: Battle Royale Optimization (BRO)
            (Battle royale optimization algorithm)
        Link:
            https://doi.org/10.1007/s00521-020-05004-4
    """
    ID_DAM = 2

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, threshold=3, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.threshold = threshold

    def create_solution(self, minmax=0):
        position = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position, minmax=minmax)
        damage = 0
        return [position, fitness, damage]

    def find_argmin_distance(self, target_pos=None, pop=None):
        list_pos = array([pop[idx][self.ID_POS] for idx in range(0, self.pop_size)])
        target_pos = reshape(target_pos, (1, -1))
        dist_list = cdist(list_pos, target_pos, 'euclidean')
        dist_list = reshape(dist_list, (-1))
        idx = argmin(dist_list[nonzero(dist_list)])
        return idx

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)
        shrink = ceil(log10(self.epoch))
        delta = round(self.epoch / shrink)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                # Compare ith soldier with nearest one (jth)
                j = self.find_argmin_distance(pop[i][self.ID_POS], pop)
                if pop[i][self.ID_FIT] < pop[j][self.ID_FIT]:
                    ## Update Winner based on global best solution
                    pos_new = pop[i][self.ID_POS] + uniform() * mean(array([pop[i][self.ID_POS], g_best[self.ID_POS]]), axis=0)
                    fit_new = self.get_fitness_position(pos_new)
                    dam_new = pop[i][self.ID_DAM] - 1  ## Substract damaged hurt -1 to go next battle
                    pop[i] = [pos_new, fit_new, dam_new]

                    ## Update Loser
                    if pop[j][self.ID_DAM] < self.threshold:  ## If loser not dead yet, move it based on general
                        pop[j][self.ID_POS] = uniform() * (maximum(pop[j][self.ID_POS], g_best[self.ID_POS]) -
                                                           minimum(pop[j][self.ID_POS], g_best[self.ID_POS])) + \
                                              maximum(pop[j][self.ID_POS], g_best[self.ID_POS])
                        pop[j][self.ID_DAM] += 1
                        pop[j][self.ID_FIT] = self.get_fitness_position(pop[j][self.ID_POS])
                    else:  ## Loser dead and respawn again
                        pop[j] = self.create_solution()
                else:
                    ## Update Loser by following position of Winner
                    pop[i] = deepcopy(pop[j])

                    ## Update Winner by following position of General to protect the King and General
                    pos_new = pop[j][self.ID_POS] + uniform() * (g_best[self.ID_POS] - pop[j][self.ID_POS])
                    fit_new = self.get_fitness_position(pos_new)
                    dam_new = 0
                    pop[j] = [pos_new, fit_new, dam_new]

            if epoch >= delta:  # max_epoch = 1000 -> delta = 300, 450, >500,....
                pos_list = array([pop[idx][self.ID_POS] for idx in range(0, self.pop_size)])
                pos_std = std(pos_list, axis=0)
                lb = g_best[self.ID_POS] - pos_std
                ub = g_best[self.ID_POS] + pos_std
                self.lb = clip(lb, self.lb, self.ub)
                self.ub = clip(ub, self.lb, self.ub)
                delta += round(delta / 2)

            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalBRO(BaseBRO):
    """
        The original version of: Battle royale optimization (BRO)
            (Battle royale optimization algorithm)
        Link:
            https://doi.org/10.1007/s00521-020-05004-4
        - Original category: Human-based
    """
    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, threshold=3, **kwargs):
        BaseBRO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, threshold, kwargs=kwargs)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)
        shrink = ceil(log10(self.epoch))
        delta = round(self.epoch / shrink)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                # Compare ith soldier with nearest one (jth)
                j = self.find_argmin_distance(pop[i][self.ID_POS], pop)
                dam, vic = i, j             ## This error in the algorithm's flow in the paper, But in the matlab code, he changed.
                if pop[i][self.ID_FIT] < pop[j][self.ID_FIT]:
                    dam, vic = j, i         ## The mistake also here in the paper.
                if pop[dam][self.ID_DAM] < self.threshold:
                    for d in range(0, self.problem_size):
                        pop[dam][self.ID_POS][d] = uniform()*(max(pop[dam][self.ID_POS][d], g_best[self.ID_POS][d]) -
                                                 min(pop[dam][self.ID_POS][d], g_best[self.ID_POS][d])) + \
                                      max(pop[dam][self.ID_POS][d], g_best[self.ID_POS][d])
                    pop[dam][self.ID_DAM] += 1
                    pop[vic][self.ID_DAM] = 0
                else:
                    for d in range(0, self.problem_size):
                        pop[dam][self.ID_POS][d] = uniform() * (self.ub[d] - self.lb[d]) + self.lb[d]
                    fit_dam = self.get_fitness_position(pop[dam][self.ID_POS])
                    pop[dam][self.ID_FIT] = fit_dam
                    pop[dam][self.ID_DAM] = 0
            if epoch >= delta:
                pos_list = array([pop[idx][self.ID_POS] for idx in range(0, self.pop_size)])
                pos_std = std(pos_list, axis=0)
                lb = g_best[self.ID_POS] - pos_std
                ub = g_best[self.ID_POS] + pos_std
                self.lb = clip(lb, self.lb, self.ub)
                self.ub = clip(ub, self.lb, self.ub)
                delta += round(delta/2)

            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

