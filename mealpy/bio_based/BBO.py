#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:24, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform
from numpy import array
from copy import deepcopy
from mealpy.root import Root


class BaseBBO(Root):
    """
    Biogeography-based optimization (BBO)
    Link:
        https://ieeexplore.ieee.org/abstract/document/4475427
    """

    def __init__(self, root_paras=None, epoch=750, pop_size=100, p_m=0.01, elites=2):
        Root.__init__(self, root_paras)
        self.epoch = epoch
        self.pop_size = pop_size
        self.p_m = p_m                  # mutation probability
        self.elites = elites            # Number of elites will be keep for next generation

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop_sorted = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        # Save the best solutions and costs in the elite arrays
        pop_elites = deepcopy(pop_sorted[:self.elites])
        # Compute migration rates, assuming the population is sorted from most fit to least fit
        mu = (self.pop_size + 1 - array(range(1, self.pop_size+1))) / (self.pop_size + 1)
        mr = 1 - mu
        for epoch in range(self.epoch):

            # Use migration rates to decide how much information to share between solutions
            pop_new = deepcopy(pop)
            for i in range(self.pop_size):

                # Probabilistic migration to the k-th solution
                for j in range(self.problem_size):

                    if uniform() < mr[i]:     # Should we immigrate?
                        # Pick a solution from which to emigrate (roulette wheel selection)
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
                for j in range(self.problem_size):
                    if uniform() < self.p_m:
                        pop_new[i][self.ID_POS][j] = uniform(self.domain_range[0], self.domain_range[1])

                # Re-calculated fitness
                pop_new[i][self.ID_FIT] = self._fitness_model__(pop_new[i][self.ID_POS])

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
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, pop_elites[self.ID_MIN_PROB][self.ID_FIT]))

        return pop_elites[self.ID_MIN_PROB][self.ID_POS], pop_elites[self.ID_MIN_PROB][self.ID_FIT], self.loss_train
