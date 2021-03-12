#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import abs, pi, ceil, sqrt, sin, clip
from numpy.random import uniform
from math import gamma
from copy import deepcopy
from mealpy.root import Root


class BaseMSA(Root):
    """
    My modified version of: Moth Search Algorithm (MSA)
        (Moth search algorithm: a bio-inspired metaheuristic algorithm for global optimization problems.)
    Link:
        https://www.mathworks.com/matlabcentral/fileexchange/59010-moth-search-ms-algorithm
        http://doi.org/10.1007/s12293-016-0212-3
    Notes:
        + Simply the matlab version above is not working (or bad at convergence characteristics).
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 n_best=5, partition=0.5, max_step_size=1.0, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.n_best = n_best            # how many of the best moths to keep from one generation to the next
        self.partition = partition      # The proportional of first partition
        self.max_step_size = max_step_size
        self.n_moth1 = int(ceil(self.partition * self.pop_size))     # np1 in paper
        self.n_moth2 = self.pop_size - self.n_moth1                  # np2 in paper
        self.golden_ratio = (sqrt(5) - 1) / 2.0     # you can change this ratio so as to get much better performance

    def _levy_walk__(self, iteration):
        beta = 1.5      # Eq. 2.23
        sigma = (gamma(1+beta) * sin(pi*(beta-1)/2) / (gamma(beta/2) * (beta-1) * 2 ** ((beta-2) / 2))) ** (1/(beta-1))
        u = uniform(self.lb, self.ub) * sigma
        v = uniform(self.lb, self.ub)
        step = u / abs(v) ** (1.0 / (beta - 1))     # Eq. 2.21
        scale = self.max_step_size / (iteration+1)
        delta_x = scale * step
        return delta_x

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        pop_best = deepcopy(pop[:self.n_best])

        for epoch in range(self.epoch):
            # Migration operator
            for i in range(0, self.n_moth1):
                #scale = self.max_step_size / (epoch+1)       # Smaller step for local walk
                temp = pop[i][self.ID_POS] + self._levy_walk__(epoch)
                temp = clip(temp, self.lb, self.ub)
                fit = self.get_fitness_position(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

            # Flying in a straight line
            for i in range(self.n_moth1, self.n_moth2):
                temp = pop[i][self.ID_POS]
                for j in range(0, self.problem_size):
                    if uniform() >= 0.5:
                        temp[j] = pop[i][self.ID_POS][j] + self.golden_ratio * (pop_best[0][self.ID_POS][j] - pop[i][self.ID_POS][j])
                    else:
                        temp[j] = pop[i][self.ID_POS][j] + (1.0/self.golden_ratio) * (pop_best[0][self.ID_POS][j] - pop[i][self.ID_POS][j])

                temp = uniform() * temp
                temp = clip(temp, self.lb, self.ub)
                fit = self.get_fitness_position(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i][self.ID_POS] = [temp, fit]

            # Replace the worst with the previous generation's elites.
            for i in range(0, self.n_best):
                pop[-1-i] = deepcopy(pop_best[i])

            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            pop_current_best = deepcopy(pop[:self.n_best])

            # Update the global best population
            for i in range(0, self.n_best):
                if pop_best[i][self.ID_FIT] > pop_current_best[i][self.ID_FIT]:
                    pop_best[i] = deepcopy(pop_current_best[i])

            self.loss_train.append(pop_best[0][self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, pop_best[0][self.ID_FIT]))
        self.solution = pop_best[0]
        return pop_best[0][self.ID_POS], pop_best[0][self.ID_FIT], self.loss_train
