#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:51, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice
from numpy import abs
from copy import deepcopy
from mealpy.root import Root


class BaseSFO(Root):
    """
    The original version of: SailFish Optimizer (SFO)
    Link:
        https://doi.org/10.1016/j.engappai.2019.01.001
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 pp=0.1, A=4, epxilon=0.0001, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size        # SailFish pop size
        self.pp = pp                    # the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
        self.A = A                      # A = 4, 6,... (coefficient for decreasing the value of Power Attack linearly from A to 0)
        self.epxilon = epxilon          # = 0.0001, 0.001

    def train(self):
        s_size = int(self.pop_size / self.pp)
        sf_pop = [self.create_solution() for _ in range(0, self.pop_size)]
        s_pop = [self.create_solution() for _ in range(0, s_size)]
        sf_gbest = self.get_global_best_solution(sf_pop, self.ID_FIT, self.ID_MIN_PROB)
        s_gbest = self.get_global_best_solution(s_pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):

            ## Calculate lamda_i using Eq.(7)
            ## Update the position of sailfish using Eq.(6)
            for i in range(0, self.pop_size):
                PD = 1 - len(sf_pop) / ( len(sf_pop) + len(s_pop) )
                lamda_i = 2 * uniform() * PD - PD
                sf_pop[i][self.ID_POS] = s_gbest[self.ID_POS] - lamda_i * ( uniform() *
                                        ( sf_gbest[self.ID_POS] + s_gbest[self.ID_POS] ) / 2 - sf_pop[i][self.ID_POS] )

            ## Calculate AttackPower using Eq.(10)
            AP = self.A * ( 1 - 2 * (epoch + 1) * self.epxilon )
            if AP < 0.5:
                alpha = int(len(s_pop) * abs(AP) )
                beta = int(self.problem_size * abs(AP))
                ### Random choice number of sardines which will be updated their position
                list1 = choice(range(0, len(s_pop)), alpha)
                for i in range(0, len(s_pop)):
                    if i in list1:
                        #### Random choice number of dimensions in sardines updated, remove third loop by numpy vector computation
                        list2 = choice(range(0, self.problem_size), beta, replace=False)
                        s_pop[i][self.ID_POS][list2] = (uniform(0, 1, self.problem_size) * (sf_gbest[self.ID_POS] - s_pop[i][self.ID_POS] + AP))[list2]
            else:
                ### Update the position of all sardine using Eq.(9)
                for i in range(0, len(s_pop)):
                    s_pop[i][self.ID_POS] = uniform()*( sf_gbest[self.ID_POS] - s_pop[i][self.ID_POS] + AP )

            ## Recalculate the fitness of all sardine
            for i in range(0, len(s_pop)):
                s_pop[i][self.ID_FIT] = self.get_fitness_position(s_pop[i][self.ID_POS], self.ID_MIN_PROB)

            ## Sort the population of sailfish and sardine (for reducing computational cost)
            sf_pop = sorted(sf_pop, key=lambda temp: temp[self.ID_FIT])
            s_pop = sorted(s_pop, key=lambda temp: temp[self.ID_FIT])
            for i in range(0, self.pop_size):
                for j in range(0, len(s_pop)):
                    ### If there is a better position in sardine population.
                    if sf_pop[i][self.ID_FIT] > s_pop[j][self.ID_FIT]:
                        sf_pop[i] = deepcopy(s_pop[j])
                        del s_pop[j]
                    break   #### This simple keyword helped reducing ton of comparing operation.
                            #### Especially when sardine pop size >> sailfish pop size

            s_pop = s_pop + [self.create_solution() for _ in range(0, s_size - len(s_pop))]

            sf_gbest = self.update_global_best_solution(sf_pop, self.ID_MIN_PROB, sf_gbest)
            s_gbest = self.update_global_best_solution(s_pop, self.ID_MIN_PROB, s_gbest)

            self.loss_train.append(sf_gbest[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, sf_gbest[self.ID_FIT]))
        self.solution = sf_gbest
        return sf_gbest[self.ID_POS], sf_gbest[self.ID_FIT], self.loss_train


class ImprovedSFO(Root):
    """
    My improved version of: Sailfish Optimizer (SFO)
    Notes:
        + Reform Energy equation,
        + No need parameter A and epxilon
        + Based on idea of Opposition-based Learning
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, pp=0.1, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size       # SailFish pop size
        self.pp = pp                   # the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1

    def train(self):
        s_size = int(self.pop_size / self.pp)
        sf_pop = [self.create_solution() for _ in range(0, self.pop_size)]
        s_pop = [self.create_solution() for _ in range(0, s_size)]
        sf_gbest = self.get_global_best_solution(sf_pop, self.ID_FIT, self.ID_MIN_PROB)
        s_gbest = self.get_global_best_solution(s_pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            ## Calculate lamda_i using Eq.(7)
            ## Update the position of sailfish using Eq.(6)
            for i in range(0, self.pop_size):
                PD = 1 - len(sf_pop) / ( len(sf_pop) + len(s_pop) )
                lamda_i = 2 * uniform() * PD - PD
                sf_pop[i][self.ID_POS] = s_gbest[self.ID_POS] - lamda_i * ( uniform() *
                                        ( sf_gbest[self.ID_POS] + s_gbest[self.ID_POS] ) / 2 - sf_pop[i][self.ID_POS] )

            ## ## Calculate AttackPower using my Eq.thieu
            #### This is our proposed, simple but effective, no need A and epxilon parameters
            AP = 1 - epoch * 1.0 / self.epoch
            if AP < 0.5:
                for i in range(0, len(s_pop)):
                    temp = (sf_gbest[self.ID_POS] + AP) / 2
                    s_pop[i][self.ID_POS] = self.lb + self.ub - temp + uniform() * (temp - s_pop[i][self.ID_POS])
            else:
                ### Update the position of all sardine using Eq.(9)
                for i in range(0, len(s_pop)):
                    s_pop[i][self.ID_POS] = uniform() * (sf_gbest[self.ID_POS] - s_pop[i][self.ID_POS] + AP)

            ## Recalculate the fitness of all sardine
            for i in range(0, len(s_pop)):
                s_pop[i][self.ID_FIT] = self.get_fitness_position(s_pop[i][self.ID_POS], self.ID_MIN_PROB)

            ## Sort the population of sailfish and sardine (for reducing computational cost)
            sf_pop = sorted(sf_pop, key=lambda temp: temp[self.ID_FIT])
            s_pop = sorted(s_pop, key=lambda temp: temp[self.ID_FIT])
            for i in range(0, self.pop_size):
                for j in range(0, len(s_pop)):
                    ### If there is a better position in sardine population.
                    if sf_pop[i][self.ID_FIT] > s_pop[j][self.ID_FIT]:
                        sf_pop[i] = deepcopy(s_pop[j])
                        del s_pop[j]
                    break   #### This simple keyword helped reducing ton of comparing operation.
                            #### Especially when sardine pop size >> sailfish pop size
            s_pop = s_pop + [self.create_solution() for _ in range(0, s_size - len(s_pop))]

            sf_gbest = self.update_global_best_solution(sf_pop, self.ID_MIN_PROB, sf_gbest)
            s_gbest = self.update_global_best_solution(s_pop, self.ID_MIN_PROB, s_gbest)

            self.loss_train.append(sf_gbest[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, sf_gbest[self.ID_FIT]))
        self.solution = sf_gbest
        return sf_gbest[self.ID_POS], sf_gbest[self.ID_FIT], self.loss_train
