#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import abs, pi, ceil, sqrt, sin, clip
from numpy.random import uniform
from math import gamma
from copy import deepcopy
from mealpy.root import Root


class BaseMSA(Root):
    """
    Standard version of: Moth Search Algorithm (MSA)
        (Moth search algorithm: a bio-inspired metaheuristic algorithm for global optimization problems.)
        https://www.mathworks.com/matlabcentral/fileexchange/59010-moth-search-ms-algorithm
        http://doi.org/10.1007/s12293-016-0212-3

    It will look so difference in comparison with the mathlab version above. Simply the matlab version above is not working
        (or bad at convergence characteristics). I changed a little bit and it worked now.!!!)
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True,
                 epoch=750, pop_size=100, n_best=5,partition=0.5, max_step_size=1.0):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
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
        u = uniform(self.domain_range[0], self.domain_range[1], self.problem_size) * sigma
        v = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        step = u / abs(v) ** (1.0 / (beta - 1))     # Eq. 2.21
        scale = self.max_step_size / (iteration+1)
        delta_x = scale * step
        return delta_x

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        pop_best = deepcopy(pop[:self.n_best])

        for epoch in range(self.epoch):
            # Migration operator
            for i in range(0, self.n_moth1):
                #scale = self.max_step_size / (epoch+1)       # Smaller step for local walk
                temp = pop[i][self.ID_POS] + self._levy_walk__(epoch)
                temp = clip(temp, self.domain_range[0], self.domain_range[1])
                fit = self._fitness_model__(temp)
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
                temp = clip(temp, self.domain_range[0], self.domain_range[1])
                fit = self._fitness_model__(temp)
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
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, pop_best[0][self.ID_FIT]))

        return pop_best[0][self.ID_POS], pop_best[0][self.ID_FIT], self.loss_train
