#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:59, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from numpy.random import rand
from copy import deepcopy
from scipy.spatial.distance import cdist
from mealpy.root import Root


class BaseSSA(Root):
    """
    Original version of: Social Spider Algorithm - A social spider algorithm for global optimization
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, r_a=1, p_c=0.7, p_m=0.1):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.r_a = r_a     # the rate of vibration attenuation when propagating over the spider web.
        self.p_c = p_c     # controls the probability of the spiders changing their dimension mask in the random walk step.
        self.p_m = p_m     # the probability of each value in a dimension mask to be one

    def _train__(self):

        g_best = [np.Inf, np.zeros(self.problem_size)]
        self.position = np.random.uniform(self.domain_range[0], self.domain_range[1], (self.pop_size, self.problem_size))
        target_position = self.position.copy()
        target_intensity = np.zeros(self.pop_size)
        mask = np.zeros((self.pop_size, self.problem_size))
        movement = np.zeros((self.pop_size, self.problem_size))
        inactive = np.zeros(self.pop_size)

        epoch = 0
        while (epoch < self.epoch):
            epoch += 1
            spider_fitness = np.array([self._fitness_model__(self.position[i]) for i in range(self.pop_size)])
            base_distance = np.mean(np.std(self.position, 0))
            distance = cdist(self.position, self.position, 'euclidean')

            intensity_source = np.log(1. / (spider_fitness + self.EPSILON) + 1)
            intensity_attenuation = np.exp(-distance / (base_distance * self.r_a))
            intensity_receive = np.tile(intensity_source, self.pop_size).reshape(self.pop_size, self.pop_size) * intensity_attenuation

            max_index = np.argmax(intensity_receive, axis=1)
            keep_target = intensity_receive[np.arange(self.pop_size), max_index] <= target_intensity
            keep_target_matrix = np.repeat(keep_target, self.problem_size).reshape(self.pop_size, self.problem_size)
            inactive = inactive * keep_target + keep_target
            target_intensity = target_intensity * keep_target + intensity_receive[np.arange(self.pop_size), max_index] * (1 - keep_target)
            target_position = target_position * keep_target_matrix + self.position[max_index] * (1 - keep_target_matrix)

            rand_position = self.position[np.floor(rand(self.pop_size * self.problem_size) * self.pop_size).astype(int), \
                                          np.tile(np.arange(self.problem_size), self.pop_size)].reshape(self.pop_size, self.problem_size)
            new_mask = np.ceil(rand(self.pop_size, self.problem_size) + rand() * self.p_m - 1)
            keep_mask = rand(self.pop_size) < self.p_c ** inactive
            inactive = inactive * keep_mask
            keep_mask_matrix = np.repeat(keep_mask, self.problem_size).reshape(self.pop_size, self.problem_size)
            mask = keep_mask_matrix * mask + (1 - keep_mask_matrix) * new_mask

            follow_position = mask * rand_position + (1 - mask) * target_position
            movement = np.repeat(rand(self.pop_size), self.problem_size).reshape(self.pop_size, self.problem_size) * movement + \
                       (follow_position - self.position) * rand(self.pop_size, self.problem_size)
            self.position = self.position + movement

            if np.min(spider_fitness) < g_best[0]:
                g_best = [np.min(spider_fitness), self.position[np.argmin(spider_fitness)].copy()]

            self.loss_train.append(g_best[0])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[0]))

        return g_best[1], g_best[0], self.loss_train


class MySSA(BaseSSA):
    """
    My version: Social Spider Algorithm
    - A social spider algorithm for global optimization
    """
    ID_POS = 0
    ID_FIT = 1
    ID_INT = 2
    ID_TARGET_POS = 3
    ID_PREV_MOVE_VEC = 4
    ID_MASK = 5

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, r_a=1, p_c=0.7, p_m=0.1):
        BaseSSA.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size, r_a, p_c, p_m)

    def _create_solution__(self, minmax=0):
        """  This algorithm has different encoding mechanism, so we need to override this method
                x: The position of s on the web.
                fit: The fitness of the current position of s.
                target_vibration: The target vibration of s in the previous iteration.
                intensity_vibration: intensity of vibration
                movement_vector: The movement that s performed in the previous iteration.
                dimension_mask: The dimension mask 1 that s employed to guide movement in the previous iteration.
                    The dimension mask is a 0-1 binary vector of length problem size.

                n_changed: The number of iterations since s has last changed its target vibration. (No need)

        """
        x = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fit = self._fitness_model__(solution=x, minmax=minmax)
        intensity = np.log(1./(fit + self.EPSILON) + 1)
        target_position = deepcopy(x)
        previous_movement_vector = np.zeros(self.problem_size)
        dimension_mask = np.zeros(self.problem_size)
        return [x, fit, intensity, target_position, previous_movement_vector, dimension_mask]

    def _amend_solution_and_return__(self, solution=None):
        for i in range(self.problem_size):
            if solution[i] < self.domain_range[0]:
                solution[i] = np.random.uniform() * (self.domain_range[0] - solution[i])
            if solution[i] > self.domain_range[1]:
                solution[i] = np.random.uniform() * (solution[i] - self.domain_range[1])
        return solution

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Epoch loop
        for epoch in range(self.epoch):
            all_pos = np.array([it[self.ID_POS] for it in pop])                 ## Matrix (pop_size, problem_size)
            base_distance = np.mean(np.std(all_pos, axis=0))                    ## Number
            dist = cdist(all_pos, all_pos, 'euclidean')

            intensity_source = np.array([it[self.ID_INT] for it in pop])
            intensity_attenuation = np.exp(-dist / (base_distance * self.r_a))  ## vector (pop_size)
            intensity_receive = np.reshape(intensity_source, (1, self.pop_size)) * intensity_attenuation  ## vector (pop_size)
            index_best_pos = np.argmax(intensity_receive, axis=1)

            ## Each individual loop
            for i in range(self.pop_size):

                if pop[index_best_pos[i]][self.ID_INT] > pop[i][self.ID_INT]:
                    pop[i][self.ID_TARGET_POS] = pop[index_best_pos[i]][self.ID_TARGET_POS]

                if np.random.uniform() > self.p_c:      ## changing mask
                    pop[i][self.ID_MASK] = np.array([0 if np.random.uniform() < self.p_m else 1 for _ in range(self.problem_size)])

                #p_fo = deepcopy(pop[i][self.ID_POS])
                # for j in range(self.problem_size):
                #     if pop[i][self.ID_MASK][j] == 0:
                #         p_fo[j] == pop[i][self.ID_TAR_VIB][j]
                #     else:
                #         p_fo[j] == pop[np.random.randint(0, self.pop_size)][self.ID_POS][j]

                p_fo = np.array([ pop[i][self.ID_TARGET_POS][j] if pop[i][self.ID_MASK][j] == 0
                                  else pop[np.random.randint(0, self.pop_size)][self.ID_POS][j]
                                  for j in range(self.problem_size)])
                ## Perform random walk
                temp = pop[i][self.ID_POS] + np.random.uniform() * (pop[i][self.ID_POS] - pop[i][self.ID_PREV_MOVE_VEC]) + \
                       (p_fo - pop[i][self.ID_POS]) * np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)

                temp = self._amend_solution_and_return__(temp)
                fit = self._fitness_model__(temp)
                pop[i][self.ID_PREV_MOVE_VEC] = temp - pop[i][self.ID_POS]
                pop[i][self.ID_INT] = np.log(1./(fit + self.EPSILON) + 1)
                pop[i][self.ID_POS] = temp
                pop[i][self.ID_FIT] = fit

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
