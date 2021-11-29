#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:59, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
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
        local_position = deepcopy(position)
        local_fitness = deepcopy(fitness)
        return [position, fitness, local_position, local_fitness]

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pos_list = np.array([item[self.ID_POS] for item in self.pop])
        fit_list = np.array([item[self.ID_LBF][self.ID_TAR] for item in self.pop])
        pos_mean = np.mean(pos_list, axis=0)
        fit_sum = np.sum(fit_list)

        if epoch % self.ff != 0:
            pop_new = []
            for i in range(0, self.pop_size):
                agent = deepcopy(self.pop[i])
                prob = np.random.uniform() * 0.2 + self.pff  # The probability of foraging for food
                if np.random.uniform() < prob:  # Birds forage for food. Eq. 1
                    x_new = self.pop[i][self.ID_POS] + self.c_minmax[0] * \
                            np.random.uniform() * (self.pop[i][self.ID_LBP] - self.pop[i][self.ID_POS]) + \
                            self.c_minmax[1] * np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS])
                else:  # Birds keep vigilance. Eq. 2
                    A1 = self.a_minmax[0] * np.exp(-self.pop_size * self.pop[i][self.ID_LBF][self.ID_TAR] / (self.EPSILON + fit_sum))
                    k = np.random.choice(list(set(range(0, self.pop_size)) - {i}))
                    t1 = (fit_list[i] - fit_list[k]) / (abs(fit_list[i] - fit_list[k]) + self.EPSILON)
                    A2 = self.a_minmax[1] * np.exp(t1 * self.pop_size * fit_list[k] / (fit_sum + self.EPSILON))
                    x_new = self.pop[i][self.ID_POS] + A1 * np.random.uniform(0, 1) * (pos_mean - self.pop[i][self.ID_POS]) + \
                            A2 * np.random.uniform(-1, 1) * (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS])
                agent[self.ID_POS] = self.amend_position_faster(x_new)
                pop_new.append(agent)
            pop_new = self.update_fitness_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
        else:
            pop_new = deepcopy(self.pop)
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
                    agent = deepcopy(self.pop[i])
                    x_new = self.pop[i][self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * self.pop[i][self.ID_POS]
                    agent[self.ID_POS] = self.amend_position_faster(x_new)
                    pop_new[i] = agent
                if choose == 1:
                    x_new = self.pop[min_idx][self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * self.pop[min_idx][self.ID_POS]
                    agent = deepcopy(self.pop[min_idx])
                    agent[self.ID_POS] = self.amend_position_faster(x_new)
                    pop_new[min_idx] = agent
                for i in range(0, int(self.pop_size / 2)):
                    if choose == 2 or min_idx != i:
                        agent = deepcopy(self.pop[i])
                        FL = np.random.uniform() * 0.4 + self.fl
                        idx = np.random.randint(0.5 * self.pop_size + 1, self.pop_size)
                        x_new = self.pop[i][self.ID_POS] + (self.pop[idx][self.ID_POS] - self.pop[i][self.ID_POS]) * FL
                        agent[self.ID_POS] = self.amend_position_faster(x_new)
                        pop_new[i] = agent
            else:  # Scrounging (Equation 6)
                for i in range(0, int(0.5 * self.pop_size)):
                    agent = deepcopy(self.pop[i])
                    x_new = self.pop[i][self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * self.pop[i][self.ID_POS]
                    agent[self.ID_POS] = self.amend_position_faster(x_new)
                    pop_new[i] = agent
                if choose == 4:
                    agent = deepcopy(self.pop[min_idx])
                    x_new = self.pop[min_idx][self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * self.pop[min_idx][self.ID_POS]
                    agent[self.ID_POS] = self.amend_position_faster(x_new)
                for i in range(int(self.pop_size / 2 + 1), self.pop_size):
                    if choose == 3 or min_idx != i:
                        agent = deepcopy(self.pop[i])
                        FL = np.random.uniform() * 0.4 + self.fl
                        idx = np.random.randint(0, 0.5 * self.pop_size)
                        x_new = self.pop[i][self.ID_POS] + (self.pop[idx][self.ID_POS] - self.pop[i][self.ID_POS]) * FL
                        agent[self.ID_POS] = self.amend_position_faster(x_new)
                        pop_new[i] = agent
            pop_new = self.update_fitness_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
