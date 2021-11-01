#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:01, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class OriginalALO(Optimizer):
    """
    The original version of: Ant Lion Optimizer (ALO)
        (The Ant Lion Optimizer)
    Link:
        https://www.mathworks.com/matlabcentral/fileexchange/49920-ant-lion-optimizer-alo
        http://dx.doi.org/10.1016/j.advengsoft.2015.01.010
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
        self.epoch = epoch
        self.pop_size = pop_size

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

    def _random_walk_around_antlion__(self, solution, current_epoch):
        I = 1  # I is the ratio in Equations (2.10) and (2.11)
        if current_epoch > self.epoch / 10:
            I = 1 + 100 * (current_epoch / self.epoch)
        if current_epoch > self.epoch  / 2:
            I = 1 + 1000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch  * (3 / 4):
            I = 1 + 10000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch  * 0.9:
            I = 1 + 100000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch  * 0.95:
            I = 1 + 1000000 * (current_epoch / self.epoch)

        # Decrease boundaries to converge towards antlion
        lb = self.problem.lb / I  # Equation (2.10) in the paper
        ub = self.problem.ub / I  # Equation (2.10) in the paper

        # Move the interval of [lb ub] around the antlion [lb+anlion ub+antlion]
        if np.random.rand() < 0.5:
            lb = lb + solution # Equation(2.8) in the paper
        else:
            lb = -lb + solution
        if np.random.rand() < 0.5:
            ub = ub + solution  # Equation(2.9) in the paper
        else:
            ub = -ub + solution

        # This function creates n random walks and normalize according to lb and ub vectors,
        temp = []
        for k in range(0, self.problem.n_dims):
            X = np.cumsum(2 * (np.random.rand(self.epoch, 1) > 0.5) - 1)
            a = np.min(X)
            b = np.max(X)
            c = lb[k]       # [a b] - -->[c d]
            d = ub[k]
            X_norm = ((X - a)* (d - c)) / (b - a) + c    # Equation(2.7) in the paper
            temp.append(X_norm)
        return np.array(temp)

    def create_child(self, idx, pop, g_best, epoch, list_fitness):
        # Select ant lions based on their fitness (the better anlion the higher chance of catching ant)
        rolette_index = self.get_index_roulette_wheel_selection(list_fitness)

        # RA is the random walk around the selected antlion by rolette wheel
        RA = self._random_walk_around_antlion__(pop[rolette_index][self.ID_POS], epoch)

        # RE is the random walk around the elite (best antlion so far)
        RE = self._random_walk_around_antlion__(g_best[self.ID_POS], epoch)

        temp = (RA[:, epoch] + RE[:, epoch]) / 2  # Equation(2.13) in the paper

        # Bound checking (bring back the antlions of ants inside search space if they go beyonds the boundaries
        pos_new = self.amend_position_faster(temp)
        fit_new = self.get_fitness_position(pos_new)
        return [pos_new, fit_new]

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
        list_fitness = np.array([item[self.ID_FIT][self.ID_TAR] for item in pop])
        # This for loop simulate random walks
        pop_idx = np.array(range(0, self.pop_size))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, epoch=epoch, list_fitness=list_fitness), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, g_best=g_best, epoch=epoch, list_fitness=list_fitness), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, g_best=g_best, epoch=epoch, list_fitness=list_fitness) for idx in pop_idx]

        # Update antlion positions and fitnesses based of the ants (if an ant becomes fitter than an antlion we
        #   assume it was caught by the antlion and the antlion update goes to its position to build the trap)
        pop = self.get_sorted_strim_population(pop+child, self.pop_size)

        # Keep the elite in the population
        pop[-1] = g_best.copy()
        return pop


class BaseALO(OriginalALO):
    """
    The is my version of: Ant Lion Optimizer (ALO)
        (The Ant Lion Optimizer)
    Link:
        + Using matrix for better performance.
        + Change the flow of updating new position. Make it better then original one
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
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

    def _random_walk_around_antlion__(self, solution, current_epoch):
        I = 1  # I is the ratio in Equations (2.10) and (2.11)
        if current_epoch > self.epoch / 10:
            I = 1 + 100 * (current_epoch / self.epoch)
        if current_epoch > self.epoch  / 2:
            I = 1 + 1000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch  * (3 / 4):
            I = 1 + 10000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch  * 0.9:
            I = 1 + 100000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch  * 0.95:
            I = 1 + 1000000 * (current_epoch / self.epoch)

        # Dicrease boundaries to converge towards antlion
        lb = self.problem.lb / I  # Equation (2.10) in the paper
        ub = self.problem.ub / I  # Equation (2.10) in the paper

        # Move the interval of [lb ub] around the antlion [lb+anlion ub+antlion]. Eq 2.8, 2.9
        lb = lb + solution if np.random.rand() < 0.5 else -lb + solution
        ub = ub + solution if np.random.rand() < 0.5 else -ub + solution

        # This function creates n random walks and normalize according to lb and ub vectors,
        ## Using matrix and vector for better performance
        X = np.array([np.cumsum(2 * (np.random.rand(self.epoch, 1) > 0.5) - 1) for _ in range(0, self.problem.n_dims)])
        a = np.min(X, axis=1)
        b = np.max(X, axis=1)
        temp1 = np.reshape((ub - lb) / (b - a), (self.problem.n_dims, 1))
        temp0 = X - np.reshape(a, (self.problem.n_dims, 1))
        X_norm = temp0 * temp1 + np.reshape(lb, (self.problem.n_dims, 1))
        return X_norm


