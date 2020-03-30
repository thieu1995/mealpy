#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:17, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import array, zeros, mean, sin, cos
from numpy.random import uniform
from copy import deepcopy
from mealpy.root import Root


class BaseSSDO(Root):
    """
    My version of: Social Ski-Driver (SSD) optimization algorithm
        (Parameters optimization of support vector machines for imbalanced data using social ski driver algorithm)
    Noted:
        I changed almost everything, basically not on equations. But the flow of algorithm and the order updating of
            velocity and position.
    """
    ID_POS = 0
    ID_FIT = 1
    ID_VEL = 2  # velocity
    ID_LBS = 3  # local best solution

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size

    def _create_solution__(self, minmax=0):
        solution = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        velocity = zeros(self.problem_size)
        local_best_solution = deepcopy(solution)
        fitness = self._fitness_model__(solution=solution, minmax=minmax)
        return [solution, fitness, velocity, local_best_solution]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            c = 2 - epoch * (2.0 / self.epoch)  # a decreases linearly from 2 to 0

            ## Calculate the mean of the best three solutions in each dimension. Eq 9
            pop = sorted(pop, key=lambda item: item[self.ID_FIT])
            pop_best_3 = deepcopy(pop[:3])
            pos_list_3 = array([item[self.ID_POS] for item in pop_best_3])
            pos_mean = mean(pos_list_3)

            # Updating velocity vectors
            for i in range(0, self.pop_size):
                r1 = uniform()  # r1, r2 is a random number in [0,1]
                r2 = uniform()
                if r2 < 0.5:     ## Use Sine function to move
                    vel_new = c * sin(r1) * (pop[i][self.ID_LBS] - pop[i][self.ID_POS]) + (2-c)*sin(r1) * (pos_mean - pop[i][self.ID_POS])
                else:                   ## Use Cosine function to move
                    vel_new = c * cos(r1) * (pop[i][self.ID_LBS] - pop[i][self.ID_POS]) + (2-c)*cos(r1) * (pos_mean - pop[i][self.ID_POS])
                pop[i][self.ID_VEL] = vel_new

            # Update Position based on velocity
            for i in range(0, self.pop_size):
                temp = pop[i][self.ID_POS] + pop[i][self.ID_VEL]
                temp = self._amend_solution_faster__(temp)
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [ temp, fit, pop[i][self.ID_VEL], pop[i][self.ID_POS] ]

            # Update the global best
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalSSD(BaseSSDO):
    """
    The original version of: Social Ski-Driver (SSD) optimization algorithm
        (Parameters optimization of support vector machines for imbalanced data using social ski driver algorithm)
    Noted:
        https://doi.org/10.1007/s00521-019-04159-z
        https://www.mathworks.com/matlabcentral/fileexchange/71210-social-ski-driver-ssd-optimization-algorithm-2019
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        BaseSSDO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            c = 2 - epoch * (2.0 / self.epoch)  # a decreases linearly from 2 to 0

            # Update Position based on velocity
            for i in range(0, self.pop_size):
                temp = pop[i][self.ID_POS] + pop[i][self.ID_VEL]
                temp = self._amend_solution_faster__(temp)
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i][self.ID_FIT] = fit
                    pop[i][self.ID_LBS] = pop[i][self.ID_POS]

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
                    vel_new = c * sin(r1) * (pop[i][self.ID_LBS] - pop[i][self.ID_POS]) + sin(r1) * (pos_mean - pop[i][self.ID_POS])
                else:                   ## Use Cosine function to move
                    vel_new = c * cos(r1) * (pop[i][self.ID_LBS] - pop[i][self.ID_POS]) + cos(r1) * (pos_mean - pop[i][self.ID_POS])
                pop[i][self.ID_VEL] = vel_new

            # Update the global best
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class LevySSDO(BaseSSDO):
    """
        My modified version of: Social Ski-Driver (SSD) optimization algorithm based on Levy_flight
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        BaseSSDO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size)

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            c = 2 - epoch * (2.0 / self.epoch)  # a decreases linearly from 2 to 0

            ## Calculate the mean of the best three solutions in each dimension. Eq 9
            pop = sorted(pop, key=lambda item: item[self.ID_FIT])
            pop_best_3 = deepcopy(pop[:3])
            pos_list_3 = array([item[self.ID_POS] for item in pop_best_3])
            pos_mean = mean(pos_list_3)

            # Updating velocity vectors
            for i in range(0, self.pop_size):
                r1 = uniform()  # r1, r2 is a random number in [0,1]
                r2 = uniform()
                if r2 < 0.5:  ## Use Sine function to move
                    vel_new = c * sin(r1) * (pop[i][self.ID_LBS] - pop[i][self.ID_POS]) + (2 - c) * sin(r1) * (pos_mean - pop[i][self.ID_POS])
                else:  ## Use Cosine function to move
                    vel_new = c * cos(r1) * (pop[i][self.ID_LBS] - pop[i][self.ID_POS]) + (2 - c) * cos(r1) * (pos_mean - pop[i][self.ID_POS])
                pop[i][self.ID_VEL] = vel_new

            # Update Position based on velocity
            for i in range(0, self.pop_size):
                ## In real life, there are lots of cases skydrive person death because they can't follow the instructor, or something went wrong with their
                # parasol. Inspired by that I added levy-flight is the a bold moves
                if uniform() < 0.5:
                    temp = pop[i][self.ID_POS] + pop[i][self.ID_VEL]
                else:
                    temp = self._levy_flight__(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])
                temp = self._amend_solution_faster__(temp)
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit, pop[i][self.ID_VEL], pop[i][self.ID_POS]]

            # Update the global best
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
