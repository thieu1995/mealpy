#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:07, 02/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%
#
# from numpy.random import np.random.uniform, np.random.choice, normal, rand
# from numpy import np.array, max, np.abs, sum, np.mean, np.argmax, min
# from copy import deepcopy
# from mealpy.optimizer import Root

import concurrent.futures as parallel
from functools import partial
import numpy as np
import time
from mealpy.optimizer import Optimizer


class BaseICA(Optimizer):
    """
        The original version of: Imperialist Competitive Algorithm (ICA)
        Link:
            https://ieeexplore.ieee.org/document/4425083
    """

    def __init__(self, problem, epoch=10000, pop_size=100, empire_count=5, selection_pressure=1, assimilation_coeff=1.5,
                 revolution_prob=0.05, revolution_rate=0.1, revolution_step_size=0.1, revolution_step_size_damp=0.99, zeta=0.1, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (n: pop_size, m: clusters), default = 100
            empire_count (): Number of Empires (also Imperialists)
            selection_pressure (): Selection Pressure
            assimilation_coeff (): Assimilation Coefficient (beta in the paper)
            revolution_prob (): Revolution Probability
            revolution_rate (): Revolution Rate       (mu)
            revolution_step_size (): Revolution Step Size  (sigma)
            revolution_step_size_damp (): Revolution Step Size Damp Rate
            zeta (): Colonies Coefficient in Total Objective Value of Empires
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.empire_count = empire_count
        self.selection_pressure = selection_pressure
        self.assimilation_coeff = assimilation_coeff
        self.revolution_prob = revolution_prob
        self.revolution_rate = revolution_rate
        self.revolution_step_size = revolution_step_size
        self.revolution_step_size_damp = revolution_step_size_damp
        self.zeta = zeta

    def revolution_country(self, position, idx_list_variables, n_revoluted):
        pos_new = position + self.revolution_step_size * np.random.normal(0, 1, self.problem.n_dims)
        idx_list = np.random.choice(idx_list_variables, n_revoluted, replace=False)
        position[idx_list] = pos_new[idx_list]      # Change only those selected index
        return position

    def solve(self, mode='sequential'):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        self.termination_start()
        pop = self.create_population(mode, self.pop_size)
        pop, g_best = self.get_global_best_solution(pop)
        self.history.save_initial_best(g_best)

        # Initialization
        n_revoluted_variables = int(round(self.revolution_rate * self.problem.n_dims))
        idx_list_variables = list(range(0, self.problem.n_dims))

        # pop = Empires
        colony_count = self.pop_size - self.empire_count
        pop_empires = pop[:self.empire_count].copy()
        pop_colonies = pop[self.empire_count:].copy()

        cost_empires_list = np.array([solution[self.ID_FIT][self.ID_TAR] for solution in pop_empires])
        cost_empires_list_normalized = cost_empires_list - (np.max(cost_empires_list) + np.min(cost_empires_list))
        prob_empires_list = np.abs(cost_empires_list_normalized / np.sum(cost_empires_list_normalized))
        # Randomly choose colonies to empires
        empires = {}
        idx_already_selected = []
        for i in range(0, self.empire_count - 1):
            empires[i] = []
            n_colonies = int(round(prob_empires_list[i] * colony_count))
            idx_list = np.random.choice(list(set(range(0, colony_count)) - set(idx_already_selected)), n_colonies, replace=False).tolist()
            idx_already_selected += idx_list
            for idx in idx_list:
                empires[i].append(pop_colonies[idx])
        idx_last = list(set(range(0, colony_count)) - set(idx_already_selected))
        empires[self.empire_count - 1] = []
        for idx in idx_last:
            empires[self.empire_count - 1].append(pop_colonies[idx])

        for epoch in range(0, self.epoch):
            time_epoch = time.time()

            # Assimilation
            for idx, colonies in empires.items():
                for idx_colony, colony in enumerate(colonies):
                    pos_new = colony[self.ID_POS] + self.assimilation_coeff * \
                              np.random.uniform(0, 1, self.problem.n_dims) * (pop_empires[idx][self.ID_POS] - colony[self.ID_POS])
                    pos_new = self.amend_position_faster(pos_new)
                    empires[idx][idx_colony][self.ID_POS] = pos_new
                empires[idx] = self.update_fitness_population(mode, empires[idx])
                # empires[idx], g_best = self.update_global_best_solution(empires[idx], self.ID_MIN_PROB, g_best)

            # Revolution
            for idx, colonies in empires.items():
                # Apply revolution to Imperialist
                pos_new = self.revolution_country(pop_empires[idx][self.ID_POS], idx_list_variables, n_revoluted_variables)
                pop_empires[idx][self.ID_POS] = self.amend_position_faster(pos_new)

                # Apply revolution to Colonies
                for idx_colony, colony in enumerate(colonies):
                    if np.random.rand() < self.revolution_prob:
                        pos_new = self.revolution_country(colony[self.ID_POS], idx_list_variables, n_revoluted_variables)
                        empires[idx][idx_colony][self.ID_POS] = self.amend_position_faster(pos_new)
                empires[idx] = self.update_fitness_population(mode, empires[idx])
            pop_empires = self.update_fitness_population(mode, pop_empires)
            _, g_best = self.update_global_best_solution(pop_empires)

            # Intra-Empire Competition
            for idx, colonies in empires.items():
                for idx_colony, colony in enumerate(colonies):
                    if self.compare_agent(colony, pop_empires[idx]):
                        empires[idx][idx_colony], pop_empires[idx] = pop_empires[idx], colony.copy()

            # Update Total Objective Values of Empires
            cost_empires_list = []
            for idx, colonies in empires.items():
                fit_list = np.array([solution[self.ID_FIT][self.ID_TAR] for solution in colonies])
                fit_empire = pop_empires[idx][self.ID_FIT][self.ID_TAR] + self.zeta * np.mean(fit_list)
                cost_empires_list.append(fit_empire)
            cost_empires_list = np.array(cost_empires_list)

            # Find possession probability of each empire based on its total power
            cost_empires_list_normalized = cost_empires_list - (np.max(cost_empires_list) + np.min(cost_empires_list))
            prob_empires_list = np.abs(cost_empires_list_normalized / np.sum(cost_empires_list_normalized))  # Vector P

            uniform_list = np.random.uniform(0, 1, len(prob_empires_list))  # Vector R
            vector_D = prob_empires_list - uniform_list
            idx_empire = np.argmax(vector_D)

            # Find the weakest empire and weakest colony inside it
            idx_weakest_empire = np.argmax(cost_empires_list)
            if len(empires[idx_weakest_empire]) > 0:
                colonies_sorted, best, worst = self.get_special_solutions(empires[idx_weakest_empire])
                empires[idx_empire].append(colonies_sorted.pop(-1))
            else:
                empires[idx_empire].append(pop_empires.pop(idx_weakest_empire))

            # update global best position
            pop, g_best = self.update_global_best_solution(pop)  # We sort the population

            ## Additional information for the framework
            time_epoch = time.time() - time_epoch
            self.history.list_epoch_time.append(time_epoch)
            self.history.list_population.append(pop.copy())
            self.print_epoch(epoch + 1, time_epoch)
            if self.termination_flag:
                if self.termination.mode == 'TB':
                    if time.time() - self.count_terminate >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                elif self.termination.mode == 'FE':
                    self.count_terminate += self.nfe_per_epoch
                    if self.count_terminate >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                elif self.termination.mode == 'MG':
                    if epoch >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                else:  # Early Stopping
                    temp = self.count_terminate + self.history.get_global_repeated_times(self.ID_FIT, self.ID_TAR, self.EPSILON)
                    if temp >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break

        ## Additional information for the framework
        self.save_optimization_process()
        return self.solution[self.ID_POS], self.solution[self.ID_FIT][self.ID_TAR]

