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
from mealpy.optimizer import Optimizer


class BaseMFO(Optimizer):
    """
        My modified version of: Moth-Flame Optimization (MFO)
            (Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm)
        Notes:
            + Changed the flow of algorithm
            + Update the old solution
            + Remove third loop for faster
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size

    def create_child(self, idx, pop, pop_flames, g_best, a, num_flame):
        #   D in Eq.(3.13)
        distance_to_flame = np.abs(pop_flames[idx][self.ID_POS] - pop[idx][self.ID_POS])
        t = (a - 1) * np.random.uniform(0, 1, self.problem.n_dims) + 1
        b = 1

        # Update the position of the moth with respect to its corresponding flame, Eq.(3.12).
        temp_1 = distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + pop_flames[idx][self.ID_POS]

        # Update the position of the moth with respect to one flame Eq.(3.12).
        ## Here is a changed, I used the best position of flames not the position num_flame th (as original code)
        temp_2 = distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + g_best[self.ID_POS]

        list_idx = idx * np.ones(self.problem.n_dims)
        pos_new = np.where(list_idx < num_flame, temp_1, temp_2)

        ## This is the way I make this algorithm working. I tried to run matlab code with large dimension and it doesn't convergence.
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

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
        # Number of flames Eq.(3.14) in the paper (linearly decreased)
        num_flame = round(self.pop_size - (epoch + 1) * ((self.pop_size - 1) / self.epoch))

        # a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a = -1 + (epoch + 1) * ((-1) / self.epoch)

        pop_flames, g_best = self.get_global_best_solution(pop)

        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, pop_flames=pop_flames,
                                                 g_best=g_best, a=a, num_flame=num_flame), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, pop_flames=pop_flames,
                                                 g_best=g_best, a=a, num_flame=num_flame), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, pop_flames, g_best, a, num_flame) for idx in pop_idx]
        return child


class OriginalMFO(BaseMFO):
    """
        The original version of: Moth-flame optimization (MFO)
            (Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm)
        Link:
            + https://www.mathworks.com/matlabcentral/fileexchange/52269-moth-flame-optimization-mfo-algorithm?s_tid=FX_rc1_behav
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    def create_child(self, idx, pop, pop_flames, g_best, a, num_flame):
        pos_new = pop[idx][self.ID_POS]
        for j in range(self.problem.n_dims):
            #   D in Eq.(3.13)
            distance_to_flame = np.abs(pop_flames[idx][self.ID_POS][j] - pop[idx][self.ID_POS][j])
            t = (a - 1) * np.random.uniform() + 1
            b = 1
            if idx <= num_flame:  # Update the position of the moth with respect to its corresponding flame
                # Eq.(3.12)
                pos_new[j] = distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + pop_flames[idx][self.ID_POS][j]
            else:  # Update the position of the moth with respect to one flame
                # Eq.(3.12).
                ## Here is a changed, I used the best position of flames not the position num_flame th (as original code)
                pos_new[j] = distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + pop_flames[num_flame][self.ID_POS][j]
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        return [pos_new, fit_new]

