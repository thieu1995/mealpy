#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 17:48, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import where
from numpy.random import uniform, randint, normal
from mealpy.root import Root


class BaseHS(Root):
    """
    My version of: Harmony Search (HS)
    Noted:
        - Using global best in the harmony memories
        - Using batch-size idea
        - Remove third for loop
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 n_new=50, c_r=0.95, pa_r=0.05, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch              # Maximum Number of Iterations
        self.pop_size = pop_size        # Harmony Memory Size
        self.n_new = n_new              # Number of New Harmonies
        self.c_r = c_r                  # Harmony Memory Consideration Rate
        self.pa_r = pa_r                # Pitch Adjustment Rate
        self.fw = 0.0001 * (self.ub - self.lb)        # Fret Width (Bandwidth)
        self.fw_damp = 0.9995                         # Fret Width Damp Ratio

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        fw = self.fw

        for epoch in range(self.epoch):
            pop_new = []
            for i in range(self.n_new):
                # Create New Harmony Position
                pos_new = uniform(self.lb, self.ub)
                delta = fw * normal(self.lb, self.ub)

                # Use Harmony Memory
                pos_new = where(uniform(0, 1, self.problem_size) < self.c_r, g_best[self.ID_POS], pos_new)
                # Pitch Adjustment
                x_new = pos_new + delta
                pos_new = where(uniform(0, 1, self.problem_size) < self.pa_r, x_new, pos_new)

                pos_new = self.amend_position_faster(pos_new)           # Check the bound
                fit = self.get_fitness_position(pos_new)                # Evaluation
                pop_new.append([pos_new, fit])

                # Batch-size idea
                if self.batch_idea:
                    if (i+1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            # Merge Harmony Memory and New Harmonies, Then sort them, Then truncate extra harmonies
            pop = pop + pop_new
            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            pop = pop[:self.pop_size]

            # Update Damp Fret Width
            fw = fw * self.fw_damp

            # Update the best position found so far
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best train: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalHS(BaseHS):
    """
    Original version of: Harmony Search (HS)
        A New Heuristic Optimization Algorithm: Harmony Search
    Link:

    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 n_new=50, c_r=0.95, pa_r=0.05, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch              # Maximum Number of Iterations
        self.pop_size = pop_size        # Harmony Memory Size
        self.n_new = n_new              # Number of New Harmonies
        self.c_r = c_r                  # Harmony Memory Consideration Rate
        self.pa_r = pa_r                # Pitch Adjustment Rate
        self.fw = 0.0001 * (self.ub - self.lb)      # Fret Width (Bandwidth)
        self.fw_damp = 0.9995                       # Fret Width Damp Ratio

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        fw = self.fw

        for epoch in range(self.epoch):
            pop_new = []
            for i in range(self.n_new):
                # Create New Harmony Position
                temp = uniform(self.lb, self.ub)
                for j in range(self.problem_size):
                    # Use Harmony Memory
                    if uniform() <= self.c_r:
                        random_index = randint(0, self.pop_size)
                        temp[j] = pop[random_index][self.ID_POS][j]
                    # Pitch Adjustment
                    if uniform() <= self.pa_r:
                        delta = fw * normal(self.lb, self.ub)       # Gaussian(Normal)
                        temp[j] = temp[j] + delta[j]
                temp = self.amend_position_faster(temp)             # Check the bound
                fit = self.get_fitness_position(temp)               # Evaluation
                pop_new.append([temp, fit])

            # Merge Harmony Memory and New Harmonies, Then sort them, Then truncate extra harmonies
            pop = pop + pop_new
            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            pop = pop[:self.pop_size]

            # Update Damp Fret Width
            fw = fw * self.fw_damp

            # Update the best position found so far
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best train: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

