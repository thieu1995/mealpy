#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:07, 02/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice, normal, rand
from numpy import array, max, abs, sum, mean, argmax, min
from copy import deepcopy
from mealpy.root import Root


class BaseICA(Root):
    """
        The original version of: Imperialist Competitive Algorithm (ICA)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 empire_count=5, selection_pressure=1, assimilation_coeff=1.5,
                 revolution_prob=0.05, revolution_rate=0.1, revolution_step_size=0.1,
                 revolution_step_size_damp=0.99, zeta=0.1, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size  # n: pop_size, m: clusters
        self.empire_count = empire_count                # Number of Empires (also Imperialists)
        self.selection_pressure = selection_pressure    # Selection Pressure
        self.assimilation_coeff = assimilation_coeff    # Assimilation Coefficient (beta in the paper)
        self.revolution_prob = revolution_prob          # Revolution Probability
        self.revolution_rate = revolution_rate          # Revolution Rate       (mu)
        self.revolution_step_size = revolution_step_size    # Revolution Step Size  (sigma)
        self.revolution_step_size_damp = revolution_step_size_damp  # Revolution Step Size Damp Rate
        self.zeta = zeta        # Colonies Coefficient in Total Objective Value of Empires

    def revolution_country(self, position, idx_list_variables, n_revoluted):
        pos_new = position + self.revolution_step_size * normal(0, 1, self.problem_size)
        idx_list = choice(idx_list_variables, n_revoluted, replace=False)
        position[idx_list] = pos_new[idx_list]      # Change only those selected index
        return position

    def train(self):
        # Initialization
        n_revoluted_variables = int(round(self.revolution_rate * self.problem_size))
        idx_list_variables = list(range(0, self.problem_size))

        pop = [self.create_solution() for _ in range(0, self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        # pop = Empires
        colony_count = self.pop_size - self.empire_count
        pop_empires = deepcopy(pop[:self.empire_count])
        pop_colonies = deepcopy(pop[self.empire_count:])

        cost_empires_list = array([solution[self.ID_FIT] for solution in pop_empires])
        cost_empires_list_normalized = cost_empires_list - (max(cost_empires_list) + min(cost_empires_list))
        prob_empires_list = abs(cost_empires_list_normalized / sum(cost_empires_list_normalized))
        # Randomly choose colonies to empires
        empires = {}
        idx_already_selected = []
        for i in range(0, self.empire_count-1):
            empires[i] = []
            n_colonies = int(round(prob_empires_list[i] * colony_count))
            idx_list = choice(list(set(range(0, colony_count)) - set(idx_already_selected)), n_colonies, replace=False).tolist()
            idx_already_selected += idx_list
            for idx in idx_list:
                empires[i].append(pop_colonies[idx])
        idx_last = list(set(range(0, colony_count)) - set(idx_already_selected))
        empires[self.empire_count-1] = []
        for idx in idx_last:
            empires[self.empire_count-1].append(pop_colonies[idx])

        # Main loop
        for epoch in range(self.epoch):

            # Assimilation
            for idx, colonies in empires.items():
                for idx_colony, colony in enumerate(colonies):
                    pos_new = colony[self.ID_POS] + self.assimilation_coeff * \
                              uniform(0, 1, self.problem_size) * (pop_empires[idx][self.ID_POS] - colony[self.ID_POS])
                    fit_new = self.get_fitness_position(pos_new)
                    empires[idx][idx_colony] = [pos_new, fit_new]
                g_best = self.update_global_best_solution(empires[idx], self.ID_MIN_PROB, g_best)

            # Revolution
            for idx, colonies in empires.items():

                # Apply revolution to Imperialist
                pos_new = self.revolution_country(pop_empires[idx][self.ID_POS], idx_list_variables, n_revoluted_variables)
                fit = self.get_fitness_position(pos_new)
                if fit < pop_empires[idx][self.ID_FIT]:
                    pop_empires[idx] = [pos_new, fit]
                    if fit < g_best[self.ID_FIT]:
                        g_best = [pos_new, fit]

                # Apply revolution to Colonies
                for idx_colony, colony in enumerate(colonies):
                    if rand() < self.revolution_prob:
                        pos_new = self.revolution_country(colony[self.ID_POS], idx_list_variables, n_revoluted_variables)
                        fit_new = self.get_fitness_position(pos_new)
                        empires[idx][idx_colony] = [pos_new, fit_new]
                g_best = self.update_global_best_solution(empires[idx], self.ID_MIN_PROB, g_best)

            # Intra-Empire Competition
            for idx, colonies in empires.items():
                for idx_colony, colony in enumerate(colonies):
                    if colony[self.ID_FIT] < pop_empires[idx][self.ID_FIT]:
                        empires[idx][idx_colony], pop_empires[idx] = pop_empires[idx], deepcopy(colony)

            # Update Total Objective Values of Empires
            cost_empires_list = []
            for idx, colonies in empires.items():
                fit_list = array([solution[self.ID_FIT] for solution in colonies])
                fit_empire = pop_empires[idx][self.ID_FIT] + self.zeta * mean(fit_list)
                cost_empires_list.append(fit_empire)
            cost_empires_list = array(cost_empires_list)

            # Find possession probability of each empire based on its total power
            cost_empires_list_normalized = cost_empires_list - (max(cost_empires_list) + min(cost_empires_list))
            prob_empires_list = abs(cost_empires_list_normalized / sum(cost_empires_list_normalized))   # Vector P

            uniform_list = uniform(0, 1, len(prob_empires_list))        # Vector R
            vector_D = prob_empires_list - uniform_list
            idx_empire = argmax(vector_D)

            # Find the weakest empire and weakest colony inside it
            idx_weakest_empire = argmax(cost_empires_list)
            if len(empires[idx_weakest_empire]) > 0:
                colonies_sorted = sorted(empires[idx_weakest_empire], key=lambda item:item[self.ID_FIT])
                empires[idx_empire].append(colonies_sorted.pop(-1))
            else:
                empires[idx_empire].append(pop_empires.pop(idx_weakest_empire))

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
