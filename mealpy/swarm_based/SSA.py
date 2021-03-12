#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:59, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import where, argmax, array, log, zeros, mean, exp, reshape, std, argmin, min, repeat, tile, ceil, arange, floor, Inf, dot
from numpy.random import rand, uniform, normal, randint
from copy import deepcopy
from scipy.spatial.distance import cdist
from mealpy.root import Root


class BaseSSA(Root):
    """
        My modified version of: Social Spider Algorithm (SSA)
            (A social spider algorithm for global optimization)
        Notes:
            + Uses batch-size idea
            + Changes the idea of intensity, which one has better intensity, others will move toward to it
    """
    ID_POS = 0
    ID_FIT = 1
    ID_INT = 2
    ID_TARGET_POS = 3
    ID_PREV_MOVE_VEC = 4
    ID_MASK = 5

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 r_a=1, p_c=0.7, p_m=0.1, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.r_a = r_a          # the rate of vibration attenuation when propagating over the spider web.
        self.p_c = p_c          # controls the probability of the spiders changing their dimension mask in the random walk step.
        self.p_m = p_m          # the probability of each value in a dimension mask to be one

    def create_solution(self, minmax=0):
        """  This algorithm has different encoding mechanism, so we need to override this method
                x: The position of s on the web.
                train: The fitness of the current position of s.
                target_vibration: The target vibration of s in the previous iteration.
                intensity_vibration: intensity of vibration
                movement_vector: The movement that s performed in the previous iteration.
                dimension_mask: The dimension mask 1 that s employed to guide movement in the previous iteration.
                    The dimension mask is a 0-1 binary vector of length problem size.

                n_changed: The number of iterations since s has last changed its target vibration. (No need)

        """
        x = uniform(self.lb, self.ub)
        fit = self.get_fitness_position(x, minmax)
        intensity = log(1. / (fit + self.EPSILON) + 1)
        target_position = deepcopy(x)
        previous_movement_vector = zeros(self.problem_size)
        dimension_mask = zeros(self.problem_size)
        return [x, fit, intensity, target_position, previous_movement_vector, dimension_mask]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Epoch loop
        for epoch in range(self.epoch):
            all_pos = array([it[self.ID_POS] for it in pop])  ## Matrix (pop_size, problem_size)
            base_distance = mean(std(all_pos, axis=0))  ## Number
            dist = cdist(all_pos, all_pos, 'euclidean')

            intensity_source = array([it[self.ID_INT] for it in pop])
            intensity_attenuation = exp(-dist / (base_distance * self.r_a))  ## vector (pop_size)
            intensity_receive = dot(reshape(intensity_source, (1, self.pop_size)), intensity_attenuation)  ## vector (1, pop_size)
            index_best_intensity = argmax(intensity_receive)

            ## Each individual loop
            for i in range(self.pop_size):

                if pop[index_best_intensity][self.ID_INT] > pop[i][self.ID_INT]:
                    pop[i][self.ID_TARGET_POS] = pop[index_best_intensity][self.ID_TARGET_POS]

                if uniform() > self.p_c:  ## changing mask
                    pop[i][self.ID_MASK] = where(uniform(0, 1, self.problem_size) < self.p_m, 0, 1)

                pos_new = where(pop[i][self.ID_MASK] == 0, pop[i][self.ID_TARGET_POS], pop[randint(0, self.pop_size)][self.ID_POS])

                ## Perform random walk
                pos_new = pop[i][self.ID_POS] + normal() * (pop[i][self.ID_POS] - pop[i][self.ID_PREV_MOVE_VEC]) + \
                          (pos_new - pop[i][self.ID_POS]) * normal()

                pos_new = self.amend_position_faster(pos_new)

                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i][self.ID_PREV_MOVE_VEC] = pos_new - pop[i][self.ID_POS]
                    pop[i][self.ID_INT] = log(1. / (fit_new + self.EPSILON) + 1)
                    pop[i][self.ID_POS] = pos_new
                    pop[i][self.ID_FIT] = fit_new

                ## Batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalSSA(Root):
    """
        The original version of: Social Spider Algorithm (SSA)
            (Social Spider Algorithm - A social spider algorithm for global optimization)
        Link:
            + Taken from Github: https://github.com/James-Yu/SocialSpiderAlgorithm
            + Slow convergence
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 r_a=1, p_c=0.7, p_m=0.1, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.r_a = r_a     # the rate of vibration attenuation when propagating over the spider web.
        self.p_c = p_c     # controls the probability of the spiders changing their dimension mask in the random walk step.
        self.p_m = p_m     # the probability of each value in a dimension mask to be one

    def train(self):

        g_best = [zeros(self.problem_size), Inf]
        self.position = uniform(self.lb, self.ub, (self.pop_size, self.problem_size))
        target_position = self.position.copy()
        target_intensity = zeros(self.pop_size)
        mask = zeros((self.pop_size, self.problem_size))
        movement = zeros((self.pop_size, self.problem_size))
        inactive = zeros(self.pop_size)

        epoch = 0
        while (epoch < self.epoch):
            epoch += 1
            spider_fitness = array([self.get_fitness_position(self.position[i]) for i in range(self.pop_size)])
            base_distance = mean(std(self.position, 0))
            distance = cdist(self.position, self.position, 'euclidean')

            intensity_source = log(1. / (spider_fitness + self.EPSILON) + 1)
            intensity_attenuation = exp(-distance / (base_distance * self.r_a))
            intensity_receive = tile(intensity_source, self.pop_size).reshape(self.pop_size, self.pop_size) * intensity_attenuation

            max_index = argmax(intensity_receive, axis=1)
            keep_target = intensity_receive[arange(self.pop_size), max_index] <= target_intensity
            keep_target_matrix = repeat(keep_target, self.problem_size).reshape(self.pop_size, self.problem_size)
            inactive = inactive * keep_target + keep_target
            target_intensity = target_intensity * keep_target + intensity_receive[arange(self.pop_size), max_index] * (1 - keep_target)
            target_position = target_position * keep_target_matrix + self.position[max_index] * (1 - keep_target_matrix)

            rand_position = self.position[floor(rand(self.pop_size * self.problem_size) * self.pop_size).astype(int), \
                                          tile(arange(self.problem_size), self.pop_size)].reshape(self.pop_size, self.problem_size)
            new_mask = ceil(rand(self.pop_size, self.problem_size) + rand() * self.p_m - 1)
            keep_mask = rand(self.pop_size) < self.p_c ** inactive
            inactive = inactive * keep_mask
            keep_mask_matrix = repeat(keep_mask, self.problem_size).reshape(self.pop_size, self.problem_size)
            mask = keep_mask_matrix * mask + (1 - keep_mask_matrix) * new_mask

            follow_position = mask * rand_position + (1 - mask) * target_position
            movement = repeat(rand(self.pop_size), self.problem_size).reshape(self.pop_size, self.problem_size) * movement + \
                       (follow_position - self.position) * rand(self.pop_size, self.problem_size)
            self.position = self.position + movement

            if min(spider_fitness) < g_best[self.ID_FIT]:
                g_best = [self.position[argmin(spider_fitness)].copy(), min(spider_fitness)]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

