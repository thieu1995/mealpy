#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:24, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform
from numpy import array, where
from copy import deepcopy
from mealpy.root import Root


class BaseBBO(Root):
    """
    My version of: Biogeography-based optimization (BBO)
        Biogeography-Based Optimization
    Link:
        https://ieeexplore.ieee.org/abstract/document/4475427
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, p_m=0.01, elites=2, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.p_m = p_m              # mutation probability
        self.elites = elites        # Number of elites will be keep for next generation

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop_sorted = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        # Save the best solutions and costs in the elite arrays
        pop_elites = deepcopy(pop_sorted[:self.elites])
        # Compute migration rates, assuming the population is sorted from most train to least train
        mu = (self.pop_size + 1 - array(range(1, self.pop_size + 1))) / (self.pop_size + 1)
        mr = 1 - mu
        for epoch in range(self.epoch):

            # Use migration rates to decide how much information to share between solutions
            pop_new = deepcopy(pop)
            for i in range(self.pop_size):

                # Probabilistic migration to the i-th position
                list_fitness = [item[self.ID_FIT] for item in pop]
                pos_old = pop_new[i][self.ID_POS]

                # Pick a position from which to emigrate (roulette wheel selection)
                idx_selected = self.get_index_roulette_wheel_selection(list_fitness)

                # this is the migration step
                pos_new = where(uniform(0, 1, self.problem_size) < mr[i], pop_new[idx_selected][self.ID_POS], pos_old)

                # Mutation
                temp = uniform(self.lb, self.ub)
                pos_new = where(uniform(0, 1, self.problem_size) < self.p_m, temp, pos_new)

                # Re-calculated fitness
                pop_new[i] = [pos_new, self.get_fitness_position(pos_new)]

            # replace the solutions with their new migrated and mutated versions then Merge Populations
            pop = sorted(deepcopy(pop_new + pop_elites), key=lambda temp: temp[self.ID_FIT])
            pop = pop[:self.pop_size]

            # Update all elite solutions
            for i in range(self.elites):
                if pop_elites[i][self.ID_FIT] > pop[i][self.ID_FIT]:
                    pop_elites[i] = deepcopy(pop[i])
            self.loss_train.append(pop_elites[self.ID_MIN_PROB][self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, pop_elites[self.ID_MIN_PROB][self.ID_FIT]))
        self.solution = pop_elites[self.ID_MIN_PROB]
        return pop_elites[self.ID_MIN_PROB][self.ID_POS], pop_elites[self.ID_MIN_PROB][self.ID_FIT], self.loss_train


class OriginalBBO(Root):
    """
    The original version of: Biogeography-based optimization (BBO)
        Biogeography-Based Optimization
    Link:
        https://ieeexplore.ieee.org/abstract/document/4475427
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, p_m=0.01, elites=2, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.p_m = p_m                  # mutation probability
        self.elites = elites            # Number of elites will be keep for next generation

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop_sorted = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        # Save the best solutions and costs in the elite arrays
        pop_elites = deepcopy(pop_sorted[:self.elites])
        # Compute migration rates, assuming the population is sorted from most train to least train
        mu = (self.pop_size + 1 - array(range(1, self.pop_size+1))) / (self.pop_size + 1)
        mr = 1 - mu
        for epoch in range(self.epoch):

            # Use migration rates to decide how much information to share between solutions
            pop_new = deepcopy(pop)
            for i in range(self.pop_size):

                # Probabilistic migration to the i-th position
                for j in range(self.problem_size):

                    if uniform() < mr[i]:     # Should we immigrate?
                        # Pick a position from which to emigrate (roulette wheel selection)
                        random_number = uniform() * sum(mu)
                        select = mu[0]
                        select_index = 0
                        while (random_number > select) and (select_index < self.pop_size - 1):
                            select_index += 1
                            select += mu[select_index]
                        # this is the migration step
                        pop_new[i][self.ID_POS][j] = pop[select_index][self.ID_POS][j]

            # Mutation
            for i in range(self.pop_size):
                temp = uniform(self.lb, self.ub)
                pos_new = where(uniform(0, 1, self.problem_size) < self.p_m, temp, pop_new[i][self.ID_POS])
                # Re-calculated fitness
                pop_new[i][self.ID_FIT] = self.get_fitness_position(pos_new)

            # replace the solutions with their new migrated and mutated versions then Merge Populations
            pop = deepcopy(pop_new)
            pop = pop + pop_elites
            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            pop = pop[:self.pop_size]

            # Update all elite solutions
            for i in range(self.elites):
                if pop_elites[i][self.ID_FIT] > pop[i][self.ID_FIT]:
                    pop_elites[i] = deepcopy(pop[i])
            self.loss_train.append(pop_elites[self.ID_MIN_PROB][self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, pop_elites[self.ID_MIN_PROB][self.ID_FIT]))
        self.solution = pop_elites[self.ID_MIN_PROB]
        return pop_elites[self.ID_MIN_PROB][self.ID_POS], pop_elites[self.ID_MIN_PROB][self.ID_FIT], self.loss_train
