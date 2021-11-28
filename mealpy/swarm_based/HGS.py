#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:37, 19/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalHGS(Optimizer):
    """
        The original version of: Hunger Games Search (HGS)
        Link:
            https://aliasgharheidari.com/HGS.html
            Hunger Games Search (HGS): Visions, Conception, Implementation, Deep Analysis, Perspectives, and Towards Performance Shifts
    """

    ID_HUN = 2      # ID for Hunger value

    def __init__(self, problem, epoch=10000, pop_size=100, PUP=0.08, LH=10000, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            PUP (float): The probability of updating position (L in the paper), default = 0.08
            LH (float): Largest hunger / threshold, default = 10000
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.PUP = PUP
        self.LH = LH

    def create_solution(self):
        """
        Returns:
            The position position with 2 element: index of position/location and index of fitness wrapper
            The general format: [position, [target, [obj1, obj2, ...]], hunger]

        ## To get the position, fitness wrapper, target and obj list
        ##      A[self.ID_POS]                  --> Return: position
        ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
        ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
        ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position=position)
        hunger = 1.0
        return [position, fitness, hunger]

    def sech(self, x):
        if np.abs(x) > 50:
            return 0.5
        return 2 / (np.exp(x) + np.exp(-x))

    def update_hunger_value(self, pop=None, g_best=None, g_worst=None):
        # min_index = pop.index(min(pop, key=lambda x: x[self.ID_FIT][self.ID_TAR]))
        # Eq (2.8) and (2.9)
        for i in range(0, self.pop_size):
            r = np.random.rand()
            # space: since we pass lower bound and upper bound as list. Better take the np.mean of them.
            space = np.mean(self.problem.ub - self.problem.lb)
            H = (pop[i][self.ID_FIT][self.ID_TAR] - g_best[self.ID_FIT][self.ID_TAR]) / \
                (g_worst[self.ID_FIT][self.ID_TAR] - g_best[self.ID_FIT][self.ID_TAR] + self.EPSILON) * r * 2 * space
            if H < self.LH:
                H = self.LH * (1 + r)
            pop[i][self.ID_HUN] += H

            if g_best[self.ID_FIT][self.ID_TAR] == pop[i][self.ID_FIT][self.ID_TAR]:
                pop[i][self.ID_HUN] = 0
        return pop

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        ## Eq. (2.2)
        ### Find the current best and current worst
        g_best, g_worst = self.get_global_best_global_worst_solution(self.pop)
        pop = self.update_hunger_value(self.pop, g_best, g_worst)

        ## Eq. (2.4)
        shrink = 2 * (1 - (epoch + 1) / self.epoch)
        total_hunger = np.sum([pop[idx][self.ID_HUN] for idx in range(0, self.pop_size)])

        pop_new = []
        for idx in range(0, self.pop_size):
            current_agent = deepcopy(self.pop[idx])
            #### Variation control
            E = self.sech(current_agent[self.ID_FIT][self.ID_TAR] - g_best[self.ID_FIT][self.ID_TAR])

            # R is a ranging controller added to limit the range of activity, in which the range of R is gradually reduced to 0
            R = 2 * shrink * np.random.rand() - shrink  # Eq. (2.3)

            ## Calculate the hungry weight of each position
            if np.random.rand() < self.PUP:
                W1 = current_agent[self.ID_HUN] * self.pop_size / (total_hunger + self.EPSILON) * np.random.rand()
            else:
                W1 = 1
            W2 = (1 - np.exp(-abs(current_agent[self.ID_HUN] - total_hunger))) * np.random.rand() * 2

            ### Udpate position of individual Eq. (2.1)
            r1 = np.random.rand()
            r2 = np.random.rand()
            if r1 < self.PUP:
                pos_new = current_agent[self.ID_POS] * (1 + np.random.normal(0, 1))
            else:
                if r2 > E:
                    pos_new = W1 * g_best[self.ID_POS] + R * W2 * abs(g_best[self.ID_POS] - current_agent[self.ID_POS])
                else:
                    pos_new = W1 * g_best[self.ID_POS] - R * W2 * abs(g_best[self.ID_POS] - current_agent[self.ID_POS])
            current_agent[self.ID_POS] = self.amend_position_faster(pos_new)
            pop_new.append(current_agent)
        self.pop = self.update_fitness_population(pop_new)
