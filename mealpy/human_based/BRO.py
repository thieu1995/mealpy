#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:17, 09/11/2020                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from scipy.spatial.distance import cdist
from mealpy.optimizer import Optimizer


class BaseBRO(Optimizer):
    """
        My best version of: Battle Royale Optimization (BRO)
            (Battle royale optimization algorithm)
        Link:
            https://doi.org/10.1007/s00521-020-05004-4
    """
    ID_DAM = 2

    def __init__(self, problem, epoch=10000, pop_size=100, threshold=3, **kwargs):
        """
        Args:
            problem ():
            epoch ():
            pop_size ():
            threshold ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.threshold = threshold

        ## Dynamic variable
        shrink = np.ceil(np.log10(self.epoch))
        self.dyn_delta = round(self.epoch / shrink)

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
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position=position)
        damage = 0
        return [position, fitness, damage]

    def find_argmin_distance(self, target_pos=None, pop=None):
        list_pos = np.array([pop[idx][self.ID_POS] for idx in range(0, self.pop_size)])
        target_pos = np.reshape(target_pos, (1, -1))
        dist_list = cdist(list_pos, target_pos, 'euclidean')
        dist_list = np.reshape(dist_list, (-1))
        idx = np.argmin(dist_list[np.nonzero(dist_list)])
        return idx

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
        if mode != "sequential":
            print("BRO algorithm is support sequential only!")
            exit(0)

        for i in range(self.pop_size):
            # Compare ith soldier with nearest one (jth)
            j = self.find_argmin_distance(pop[i][self.ID_POS], pop)
            if pop[i][self.ID_FIT] < pop[j][self.ID_FIT]:
                ## Update Winner based on global best solution
                pos_new = pop[i][self.ID_POS] + np.random.uniform() * np.mean(np.array([pop[i][self.ID_POS], g_best[self.ID_POS]]), axis=0)
                fit_new = self.get_fitness_position(pos_new)
                dam_new = pop[i][self.ID_DAM] - 1  ## Substract damaged hurt -1 to go next battle
                pop[i] = [pos_new, fit_new, dam_new]
                ## Update Loser
                if pop[j][self.ID_DAM] < self.threshold:  ## If loser not dead yet, move it based on general
                    pos_new = np.random.uniform() * (np.maximum(pop[j][self.ID_POS], g_best[self.ID_POS]) -
                                                       np.minimum(pop[j][self.ID_POS], g_best[self.ID_POS])) + \
                                          np.maximum(pop[j][self.ID_POS], g_best[self.ID_POS])
                    dam_new =  pop[j][self.ID_DAM] + 1

                    pop[j][self.ID_FIT] = self.get_fitness_position(pop[j][self.ID_POS])
                else:  ## Loser dead and respawn again
                    pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
                    dam_new = 0
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                pop[j] = [pos_new, fit_new, dam_new]
            else:
                ## Update Loser by following position of Winner
                pop[i] = pop[j].copy()
                ## Update Winner by following position of General to protect the King and General
                pos_new = pop[j][self.ID_POS] + np.random.uniform() * (g_best[self.ID_POS] - pop[j][self.ID_POS])
                fit_new = self.get_fitness_position(pos_new)
                dam_new = 0
                pop[j] = [pos_new, fit_new, dam_new]
        if epoch >= self.dyn_delta:  # max_epoch = 1000 -> delta = 300, 450, >500,....
            pos_list = np.array([pop[idx][self.ID_POS] for idx in range(0, self.pop_size)])
            pos_std = np.std(pos_list, axis=0)
            lb = g_best[self.ID_POS] - pos_std
            ub = g_best[self.ID_POS] + pos_std
            self.problem.lb = np.clip(lb, self.problem.lb, self.problem.ub)
            self.problem.ub = np.clip(ub, self.problem.lb, self.problem.ub)
            self.dyn_delta += np.round(self.dyn_delta / 2)
        return pop


class OriginalBRO(BaseBRO):
    """
        The original version of: Battle Royale Optimization (BRO)
            (Battle royale optimization algorithm)
        Link:
            https://doi.org/10.1007/s00521-020-05004-4
        - Original category: Human-based
    """
    def __init__(self, problem, epoch=10000, pop_size=100, threshold=3, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (Harmony Memory Size), default = 100
            threshold ():
        """
        super().__init__(problem, epoch, pop_size, threshold, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

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
        if mode != "sequential":
            print("BRO algorithm is support sequential only!")
            exit(0)
        for i in range(self.pop_size):
            # Compare ith soldier with nearest one (jth)
            j = self.find_argmin_distance(pop[i][self.ID_POS], pop)
            dam, vic = i, j  ## This error in the algorithm's flow in the paper, But in the matlab code, he changed.
            if pop[i][self.ID_FIT] < pop[j][self.ID_FIT]:
                dam, vic = j, i  ## The mistake also here in the paper.
            if pop[dam][self.ID_DAM] < self.threshold:
                pos_new = pop[dam][self.ID_POS].copy()
                for d in range(0, self.problem.n_dims):
                    pos_new[d] = np.random.uniform() * (max(pop[dam][self.ID_POS][d], g_best[self.ID_POS][d]) -
                                min(pop[dam][self.ID_POS][d], g_best[self.ID_POS][d])) + max(pop[dam][self.ID_POS][d], g_best[self.ID_POS][d])
                pop[dam][self.ID_POS] = self.amend_position_faster(pos_new)
                pop[dam][self.ID_FIT] = self.get_fitness_position(pop[dam][self.ID_POS])
                pop[dam][self.ID_DAM] += 1
                pop[vic][self.ID_DAM] = 0
            else:
                pop[dam] = self.create_solution()
        if epoch >= self.dyn_delta:
            pos_list = np.array([pop[idx][self.ID_POS] for idx in range(0, self.pop_size)])
            pos_std = np.std(pos_list, axis=0)
            lb = g_best[self.ID_POS] - pos_std
            ub = g_best[self.ID_POS] + pos_std
            self.problem.lb = np.clip(lb, self.problem.lb, self.problem.ub)
            self.problem.ub = np.clip(ub, self.problem.lb, self.problem.ub)
            self.dyn_delta += round(self.dyn_delta / 2)
        return pop

