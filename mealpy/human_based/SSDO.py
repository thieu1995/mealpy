#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:17, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import array, mean, sin, cos
from numpy.random import uniform
from copy import deepcopy
from mealpy.root import Root


class BaseSSDO(Root):
    """
    The original version of: Social Ski-Driver (SSD) optimization algorithm
        (Parameters optimization of support vector machines for imbalanced data using social ski driver algorithm)
    Noted:
        https://doi.org/10.1007/s00521-019-04159-z
        https://www.mathworks.com/matlabcentral/fileexchange/71210-social-ski-driver-ssd-optimization-algorithm-2019
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        list_velocity = uniform(self.lb, self.ub, (self.pop_size, self.problem_size))
        list_local_best = array([item[self.ID_POS] for item in pop])

        for epoch in range(self.epoch):
            c = 2 - epoch * (2.0 / self.epoch)  # a decreases linearly from 2 to 0

            # Update Position based on velocity
            for i in range(0, self.pop_size):
                pos_new = pop[i][self.ID_POS] + list_velocity[i]
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    list_local_best[i] = deepcopy(pos_new)
                pop[i] = [pos_new, fit_new]

            ## Calculate the mean of the best three solutions in each dimension. Eq 9
            pop = sorted(pop, key=lambda item: item[self.ID_FIT])
            pop_best_3 = deepcopy(pop[:3])
            pos_list_3 = array([item[self.ID_POS] for item in pop_best_3])
            pos_mean = mean(pos_list_3)

            # Updating velocity vectors
            for i in range(0, self.pop_size):
                r1 = uniform()  # r1, r2 is a random number in [0,1]
                r2 = uniform()
                if r2 <= 0.5:     ## Use Sine function to move
                    vel_new = c * sin(r1) * (list_local_best[i] - pop[i][self.ID_POS]) + sin(r1) * (pos_mean - pop[i][self.ID_POS])
                else:                   ## Use Cosine function to move
                    vel_new = c * cos(r1) * (list_local_best[i] - pop[i][self.ID_POS]) + cos(r1) * (pos_mean - pop[i][self.ID_POS])
                list_velocity[i] = deepcopy(vel_new)

            # Update the global best
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevySSDO(BaseSSDO):
    """
        My levy version of: Social Ski-Driver (SSD) optimization algorithm based on Levy_flight
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseSSDO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        list_velocity = uniform(self.lb, self.ub, (self.pop_size, self.problem_size))
        list_local_best = array([item[self.ID_POS] for item in pop])

        for epoch in range(self.epoch):
            c = 2 - epoch * (2.0 / self.epoch)  # a decreases linearly from 2 to 0

            # Update Position based on velocity
            for i in range(0, self.pop_size):
                ## In real life, there are lots of cases skydrive person death because they can't follow the instructor,
                # or something went wrong with their parasol. Inspired by that I added levy-flight is the a bold moves
                if uniform() < 0.7:
                    pos_new = pop[i][self.ID_POS] + list_velocity[i]
                else:
                    pos_new = self.levy_flight(epoch, pop[i][self.ID_POS], g_best[self.ID_POS], case=1)
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    list_local_best[i] = deepcopy(pos_new)
                pop[i] = [pos_new, fit_new]

            ## Calculate the mean of the best three solutions in each dimension. Eq 9
            pop = sorted(pop, key=lambda item: item[self.ID_FIT])
            pop_best_3 = deepcopy(pop[:3])
            pos_list_3 = array([item[self.ID_POS] for item in pop_best_3])
            pos_mean = mean(pos_list_3)

            # Updating velocity vectors
            for i in range(0, self.pop_size):
                r1 = uniform()  # r1, r2 is a random number in [0,1]
                r2 = uniform()
                if r2 <= 0.5:  ## Use Sine function to move
                    vel_new = c * sin(r1) * (list_local_best[i] - pop[i][self.ID_POS]) + sin(r1) * (pos_mean - pop[i][self.ID_POS])
                else:  ## Use Cosine function to move
                    vel_new = c * cos(r1) * (list_local_best[i] - pop[i][self.ID_POS]) + cos(r1) * (pos_mean - pop[i][self.ID_POS])
                list_velocity[i] = deepcopy(vel_new)

            # Update the global best
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

