#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 17:48, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint, normal
from mealpy.root import Root


class BaseHS(Root):
    """
    My version of: Harmony Search (HS)
        - Using global best in the harmony memories
    """
    def __init__(self, root_paras=None, epoch=750, pop_size=100, n_new=50, c_r=0.95, pa_r=0.05):
        Root.__init__(self, root_paras)
        self.epoch = epoch              # Maximum Number of Iterations
        self.pop_size = pop_size        # Harmony Memory Size
        self.n_new = n_new              # Number of New Harmonies
        self.c_r = c_r                  # Harmony Memory Consideration Rate
        self.pa_r = pa_r                # Pitch Adjustment Rate
        self.fw = 0.0001 * (self.domain_range[1] - self.domain_range[0])        # Fret Width (Bandwidth)
        self.fw_damp = 0.9995                                                   # Fret Width Damp Ratio

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        fw = self.fw

        for epoch in range(self.epoch):
            pop_new = []
            for i in range(self.n_new):
                # Create New Harmony Position
                temp = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                for j in range(self.problem_size):
                    # Use Harmony Memory
                    if uniform() <= self.c_r:
                        temp[j] = g_best[self.ID_POS][j]
                    # Pitch Adjustment
                    if uniform() <= self.pa_r:
                        delta = fw * normal(self.domain_range[0], self.domain_range[1])    # Gaussian(Normal)
                        temp[j] = temp[j] + delta
                temp = self._amend_solution_faster__(temp)           # Check the bound
                fit = self._fitness_model__(temp)                               # Evaluation
                pop_new.append([temp, fit])

            # Merge Harmony Memory and New Harmonies, Then sort them, Then truncate extra harmonies
            pop = pop + pop_new
            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            pop = pop[:self.pop_size]

            # Update Damp Fret Width
            fw = fw * self.fw_damp

            # Update the best solution found so far
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalHS(BaseHS):
    """
    Original version of: Harmony Search (HS)
        - A New Heuristic Optimization Algorithm: Harmony Search
    """
    ID_WEIGHT = 2

    def __init__(self, root_paras=None, epoch=750, pop_size=100, n_new=50, c_r=0.95, pa_r=0.05):
        BaseHS.__init__(self, root_paras, epoch, pop_size, n_new, c_r, pa_r)
        self.fw = 0.02 * (self.domain_range[1] - self.domain_range[0])          # Fret Width (Bandwidth)
        self.fw_damp = 0.995                                                    # Fret Width Damp Ratio

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        fw = self.fw

        for epoch in range(self.epoch):
            pop_new = []
            for i in range(self.n_new):
                # Create New Harmony Position
                temp = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
                for j in range(self.problem_size):
                    # Use Harmony Memory
                    if uniform() <= self.c_r:
                        random_index = randint(0, self.pop_size)
                        temp[j] = pop[random_index][self.ID_POS][j]
                    # Pitch Adjustment
                    if uniform() <= self.pa_r:
                        delta = fw * normal(self.domain_range[0], self.domain_range[1])    # Gaussian(Normal)
                        temp[j] = temp[j] + delta
                temp = self._amend_solution_faster__(temp)           # Check the bound
                fit = self._fitness_model__(temp)                               # Evaluation
                pop_new.append([temp, fit])

            # Merge Harmony Memory and New Harmonies, Then sort them, Then truncate extra harmonies
            pop = pop + pop_new
            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            pop = pop[:self.pop_size]

            # Update Damp Fret Width
            fw = fw * self.fw_damp

            # Update the best solution found so far
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

