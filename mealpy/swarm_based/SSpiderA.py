#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:59, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from scipy.spatial.distance import cdist
from mealpy.optimizer import Optimizer


class BaseSSpiderA(Optimizer):
    """
        My modified version of: Social Spider Algorithm (BaseSSpiderA)
            (A social spider algorithm for global optimization)
        Link:
            https://doi.org/10.1016/j.asoc.2015.02.014
        Notes:
            + Changes the idea of intensity, which one has better intensity, others will move toward to it
    """
    ID_POS = 0
    ID_FIT = 1
    ID_INT = 2
    ID_TARGET_POS = 3
    ID_PREV_MOVE_VEC = 4
    ID_MASK = 5

    def __init__(self, problem, epoch=10000, pop_size=100, r_a=1, p_c=0.7, p_m=0.1, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_a (float): the rate of vibration attenuation when propagating over the spider web, default=1.0
            p_c (float): controls the probability of the spiders changing their dimension mask in the random walk step, default=0.7
            p_m (float): the probability of each value in a dimension mask to be one, default=0.1
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.r_a = r_a
        self.p_c = p_c
        self.p_m = p_m

    def create_solution(self):
        """
        Returns:
            The position position with 2 element: index of position/location and index of fitness wrapper
            The general format: [position, [target, [obj1, obj2, ...]]]

        ## To get the position, fitness wrapper, target and obj list
        ##      A[self.ID_POS]                  --> Return: position
        ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
        ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
        ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]

        x: The position of s on the web.
                train: The fitness of the current position of s.
                target_vibration: The target vibration of s in the previous iteration.
                intensity_vibration: intensity of vibration
                movement_vector: The movement that s performed in the previous iteration.
                dimension_mask: The dimension mask 1 that s employed to guide movement in the previous iteration.
                    The dimension mask is a 0-1 binary vector of length problem size.

                n_changed: The number of iterations since s has last changed its target vibration. (No need)
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position)
        intensity = np.log(1. / (fitness[self.ID_TAR] + self.EPSILON) + 1)
        target_position = position.copy()
        previous_movement_vector = np.zeros(self.problem.n_dims)
        dimension_mask = np.zeros(self.problem.n_dims)
        return [position, fitness, intensity, target_position, previous_movement_vector, dimension_mask]

    def create_child(self, idx, pop, id_best_intennsity):
        if pop[id_best_intennsity][self.ID_INT] > pop[idx][self.ID_INT]:
            pop[idx][self.ID_TARGET_POS] = pop[id_best_intennsity][self.ID_TARGET_POS]

        if np.random.uniform() > self.p_c:  ## changing mask
            pop[idx][self.ID_MASK] = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.p_m, 0, 1)

        pos_new = np.where(pop[idx][self.ID_MASK] == 0, pop[idx][self.ID_TARGET_POS], pop[np.random.randint(0, self.pop_size)][self.ID_POS])

        ## Perform random walk
        pos_new = pop[idx][self.ID_POS] + np.random.normal() * (pop[idx][self.ID_POS] - pop[idx][self.ID_PREV_MOVE_VEC]) + \
                  (pos_new - pop[idx][self.ID_POS]) * np.random.normal()
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        agent = pop[idx].copy()
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            agent[self.ID_PREV_MOVE_VEC] = pos_new - pop[idx][self.ID_POS]
            agent[self.ID_INT] = np.log(1. / (fit_new[self.ID_TAR] + self.EPSILON) + 1)
            agent[self.ID_POS] = pos_new
            agent[self.ID_FIT] = fit_new
        return agent

        ## Batch size idea
        # if self.batch_idea:
        #     if (i + 1) % self.batch_size == 0:
        #         g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
        # else:
        #     if (i + 1) % self.pop_size == 0:
        #         g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)

    def evolve(self, mode='sequential', epoch=None, pop=None, g_best=None):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        all_pos = np.array([it[self.ID_POS] for it in pop])  ## Matrix (pop_size, problem_size)
        base_distance = np.mean(np.std(all_pos, axis=0))  ## Number
        dist = cdist(all_pos, all_pos, 'euclidean')

        intensity_source = np.array([it[self.ID_INT] for it in pop])
        intensity_attenuation = np.exp(-dist / (base_distance * self.r_a))  ## vector (pop_size)
        intensity_receive = np.dot(np.reshape(intensity_source, (1, self.pop_size)), intensity_attenuation)  ## vector (1, pop_size)
        id_best_intennsity = np.argmax(intensity_receive)

        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, id_best_intennsity=id_best_intennsity), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, id_best_intennsity=id_best_intennsity), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, id_best_intennsity) for idx in pop_idx]
        return child



# class OriginalSSA(Root):
#     """
#         The original version of: Social Spider Algorithm (SSA)
#             (Social Spider Algorithm - A social spider algorithm for global optimization)
#         Link:
#             + Taken from Github: https://github.com/James-Yu/SocialSpiderAlgorithm
#             + Slow convergence
#     """
#
#     def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
#                  r_a=1, p_c=0.7, p_m=0.1, **kwargs):
#         super().__init__(obj_func, lb, ub, verbose, kwargs)
#         self.epoch = epoch
#         self.pop_size = pop_size
#         self.r_a = r_a     # the rate of vibration attenuation when propagating over the spider web.
#         self.p_c = p_c     # controls the probability of the spiders changing their dimension mask in the random walk step.
#         self.p_m = p_m     # the probability of each value in a dimension mask to be one
#
#     def train(self):
#
#         g_best = [np.zeros(self.problem_size), np.Inf]
#         self.position = np.random.uniform(self.lb, self.ub, (self.pop_size, self.problem_size))
#         target_position = self.position.copy()
#         target_intensity = np.zeros(self.pop_size)
#         mask = np.zeros((self.pop_size, self.problem_size))
#         movement = np.zeros((self.pop_size, self.problem_size))
#         inactive = np.zeros(self.pop_size)
#
#         epoch = 0
#         while (epoch < self.epoch):
#             epoch += 1
#             spider_fitness = np.array([self.get_fitness_position(self.position[i]) for i in range(self.pop_size)])
#             base_distance = np.mean(np.std(self.position, 0))
#             distance = cdist(self.position, self.position, 'euclidean')
#
#             intensity_source = np.log(1. / (spider_fitness + self.EPSILON) + 1)
#             intensity_attenuation = np.exp(-distance / (base_distance * self.r_a))
#             intensity_receive = np.tile(intensity_source, self.pop_size).np.reshape(self.pop_size, self.pop_size) * intensity_attenuation
#
#             max_index = np.argmax(intensity_receive, axis=1)
#             keep_target = intensity_receive[np.arange(self.pop_size), max_index] <= target_intensity
#             keep_target_matrix = np.repeat(keep_target, self.problem_size).np.reshape(self.pop_size, self.problem_size)
#             inactive = inactive * keep_target + keep_target
#             target_intensity = target_intensity * keep_target + intensity_receive[np.arange(self.pop_size), max_index] * (1 - keep_target)
#             target_position = target_position * keep_target_matrix + self.position[max_index] * (1 - keep_target_matrix)
#
#             rand_position = self.position[np.floor(rand(self.pop_size * self.problem_size) * self.pop_size).astype(int), \
#                                           np.tile(np.arange(self.problem_size), self.pop_size)].np.reshape(self.pop_size, self.problem_size)
#             new_mask = np.ceil(rand(self.pop_size, self.problem_size) + rand() * self.p_m - 1)
#             keep_mask = rand(self.pop_size) < self.p_c ** inactive
#             inactive = inactive * keep_mask
#             keep_mask_matrix = np.repeat(keep_mask, self.problem_size).np.reshape(self.pop_size, self.problem_size)
#             mask = keep_mask_matrix * mask + (1 - keep_mask_matrix) * new_mask
#
#             follow_position = mask * rand_position + (1 - mask) * target_position
#             movement = np.repeat(rand(self.pop_size), self.problem_size).np.reshape(self.pop_size, self.problem_size) * movement + \
#                        (follow_position - self.position) * rand(self.pop_size, self.problem_size)
#             self.position = self.position + movement
#
#             if min(spider_fitness) < g_best[self.ID_FIT]:
#                 g_best = [self.position[argmin(spider_fitness)].copy(), min(spider_fitness)]
#
#             self.loss_train.append(g_best[self.ID_FIT])
#             if self.verbose:
#                 print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
#         self.solution = g_best
#         return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

