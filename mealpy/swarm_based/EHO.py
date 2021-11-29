#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:41, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseEHO(Optimizer):
    """
    The original version of: Elephant Herding Optimization (EHO)
        (Elephant Herding Optimization )
    Link:
        https://doi.org/10.1109/ISCBI.2015.8
    """

    def __init__(self, problem, epoch=10000, pop_size=100, alpha=0.5, beta=0.5, n_clans=5, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): a factor that determines the influence of the best in each clan
            beta (float): a factor that determines the influence of the x_center
            n_clans (int): the number of clans
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size + n_clans
        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha
        self.beta = beta
        self.n_clans = n_clans
        self.n_individuals = int(self.pop_size / self.n_clans)

        self.nfe_per_epoch = pop_size + self.n_clans
        self.sort_flag = False

    def _create_pop_group(self, pop):
        pop_group = []
        for i in range(0, self.n_clans):
            group = pop[i*self.n_individuals: (i+1)*self.n_individuals]
            pop_group.append(deepcopy(group))
        return pop_group

    def initialization(self):
        self.pop = self.create_population(self.pop_size)
        self.pop_group = self._create_pop_group(self.pop)
        _, self.g_best = self.get_global_best_solution(self.pop)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        # Clan updating operator
        pop_new = []
        for i in range(0, self.pop_size):
            clan_idx = int(i / self.n_individuals)
            pos_clan_idx = int(i % self.n_individuals)

            if pos_clan_idx == 0:  # The best in clan, because all clans are sorted based on fitness
                center = np.mean(np.array([item[self.ID_POS] for item in self.pop_group[clan_idx]]), axis=0)
                pos_new = self.beta * center
            else:
                pos_new = self.pop_group[clan_idx][pos_clan_idx][self.ID_POS] + self.alpha * np.random.uniform() * \
                          (self.pop_group[clan_idx][0][self.ID_POS] - self.pop_group[clan_idx][pos_clan_idx][self.ID_POS])
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        # Update fitness value
        self.pop = self.update_fitness_population(pop_new)
        self.pop_group = self._create_pop_group(self.pop)

        # Separating operator
        for i in range(0, self.n_clans):
            self.pop_group[i], _ = self.get_global_best_solution(self.pop_group[i])
            self.pop_group[i][-1] = self.create_solution()
        self.pop = [agent for pack in self.pop_group for agent in pack]
