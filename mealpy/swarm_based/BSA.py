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


class BaseBSA(Optimizer):
    """
    The original version of: Bird Swarm Algorithm (BSA)
        (A new bio-inspired optimisation algorithm: Bird Swarm Algorithm)
    Link:
        http://doi.org/10.1080/0952813X.2015.1042530
        https://www.mathworks.com/matlabcentral/fileexchange/51256-bird-swarm-algorithm-bsa
    """
    ID_POS = 0
    ID_FIT = 1
    ID_LBP = 2      # local best position
    ID_LBF = 3      # local best fitness

    def __init__(self, problem, epoch=10000, pop_size=100,
                 ff=10, pff=0.8, c_couples=(1.5, 1.5), a_couples=(1.0, 1.0), fl=0.5, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            ff (int): flight frequency - default = 10
            pff (float): the probability of foraging for food - default = 0.8
            c_couples (list): [c1, c2]: Cognitive accelerated coefficient, Social accelerated coefficient same as PSO
            a_couples (list): [a1, a2]: The indirect and direct effect on the birds' vigilance behaviours.
            fl (float): The followed coefficient- default = 0.5
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.ff = ff
        self.pff = pff
        self.c_minmax = c_couples
        self.a_minmax = a_couples
        self.fl = fl

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
        fitness = self.get_fitness_position(position)
        local_position = position.copy()
        local_fitness = fitness.copy()
        return [position, fitness, local_position, local_fitness]

    def _update_solution_(self, solution_old, solution_new):
        solution_old = solution_old.copy()
        pos_new = self.amend_position_faster(solution_new[self.ID_POS])
        fit_new = self.get_fitness_position(pos_new)
        solution_new[self.ID_FIT] = fit_new
        if self.compare_agent(solution_new, solution_old):
            solution_old[self.ID_LBP] = pos_new.copy()
            solution_old[self.ID_LBF] = fit_new.copy()
        solution_old[self.ID_POS] = pos_new.copy()
        solution_old[self.ID_FIT] = fit_new.copy()
        return solution_old

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
            print("BSA algorithm is supported sequential mode only!")
            exit(0)

        pos_list = np.array([item[self.ID_POS] for item in pop])
        fit_list = np.array([item[self.ID_LBF][self.ID_TAR] for item in pop])
        pos_mean = np.mean(pos_list, axis=0)
        fit_sum = np.sum(fit_list)

        if epoch % self.ff != 0:
            for i in range(0, self.pop_size):
                prob = np.random.uniform() * 0.2 + self.pff  # The probability of foraging for food
                if np.random.uniform() < prob:  # Birds forage for food. Eq. 1
                    x_new = pop[i][self.ID_POS] + self.c_minmax[0] * np.random.uniform() * (pop[i][self.ID_LBP] - pop[i][self.ID_POS]) + \
                            self.c_minmax[1] * np.random.uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                else:  # Birds keep vigilance. Eq. 2
                    A1 = self.a_minmax[0] * np.exp(-self.pop_size * pop[i][self.ID_LBF][self.ID_TAR] / (self.EPSILON + fit_sum))
                    k = np.random.choice(list(set(range(0, self.pop_size)) - {i}))
                    t1 = (fit_list[i] - fit_list[k]) / (abs(fit_list[i] - fit_list[k]) + self.EPSILON)
                    A2 = self.a_minmax[1] * np.exp(t1 * self.pop_size * fit_list[k] / (fit_sum + self.EPSILON))
                    x_new = pop[i][self.ID_POS] + A1 * np.random.uniform(0, 1) * (pos_mean - pop[i][self.ID_POS]) + \
                            A2 * np.random.uniform(-1, 1) * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                pop[i] = self._update_solution_(pop[i], [x_new, None, None, None])
        else:
            # Divide the bird swarm into two parts: producers and scroungers.
            min_idx = np.argmin(fit_list)
            max_idx = np.argmax(fit_list)
            choose = 0
            if min_idx < 0.5 * self.pop_size and max_idx < 0.5 * self.pop_size:
                choose = 1
            if min_idx > 0.5 * self.pop_size and max_idx < 0.5 * self.pop_size:
                choose = 2
            if min_idx < 0.5 * self.pop_size and max_idx > 0.5 * self.pop_size:
                choose = 3
            if min_idx > 0.5 * self.pop_size and max_idx > 0.5 * self.pop_size:
                choose = 4

            if choose < 3:  # Producing (Equation 5)
                for i in range(int(self.pop_size / 2 + 1), self.pop_size):
                    x_new = pop[i][self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * pop[i][self.ID_POS]
                    pop[i] = self._update_solution_(pop[i], [x_new, None, None, None])
                if choose == 1:
                    x_new = pop[min_idx][self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * pop[min_idx][self.ID_POS]
                    pop[min_idx] = self._update_solution_(pop[min_idx], [x_new, None, None, None])
                for i in range(0, int(self.pop_size / 2)):
                    if choose == 2 or min_idx != i:
                        FL = np.random.uniform() * 0.4 + self.fl
                        idx = np.random.randint(0.5 * self.pop_size + 1, self.pop_size)
                        x_new = pop[i][self.ID_POS] + (pop[idx][self.ID_POS] - pop[i][self.ID_POS]) * FL
                        pop[min_idx] = self._update_solution_(pop[min_idx], [x_new, None, None, None])
            else:  # Scrounging (Equation 6)
                for i in range(0, int(0.5 * self.pop_size)):
                    x_new = pop[i][self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * pop[i][self.ID_POS]
                    pop[i] = self._update_solution_(pop[i], [x_new, None, None, None])
                if choose == 4:
                    x_new = pop[min_idx][self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * pop[min_idx][self.ID_POS]
                    pop[min_idx] = self._update_solution_(pop[min_idx], [x_new, None, None, None])
                for i in range(int(self.pop_size / 2 + 1), self.pop_size):
                    if choose == 3 or min_idx != i:
                        FL = np.random.uniform() * 0.4 + self.fl
                        idx = np.random.randint(0, 0.5 * self.pop_size)
                        x_new = pop[i][self.ID_POS] + (pop[idx][self.ID_POS] - pop[i][self.ID_POS]) * FL
                        pop[i] = self._update_solution_(pop[i], [x_new, None, None, None])
        return pop
