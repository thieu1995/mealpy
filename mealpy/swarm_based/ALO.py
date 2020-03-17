#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:01, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import rand
from numpy import cumsum, array, ones, min, max, reshape
from copy import deepcopy
from mealpy.root import Root


class OriginalALO(Root):
    """
    The original version of: Ant Lion Optimizer (ALO)
        (The Ant Lion Optimizer)
    Link:
        https://www.mathworks.com/matlabcentral/fileexchange/49920-ant-lion-optimizer-alo
        http://dx.doi.org/10.1016/j.advengsoft.2015.01.010
    """
    def __init__(self, root_paras=None, epoch=750, pop_size=100):
        Root.__init__(self, root_paras)
        self.epoch = epoch
        self.pop_size = pop_size

    def _roulette_wheel_selection__(self, weights):
        # The problem with this function is: it will not working with negative fitness values.
        accumulation = cumsum(weights)
        p = rand() * accumulation[-1]
        chosen_index = -1
        for idx in range(0, self.pop_size):
            if accumulation[idx] > p:
                chosen_index = idx
                break
        return chosen_index

    def _random_walk_around_antlion__(self, solution, current_epoch):
        # Make the bounded vector
        lb = self.domain_range[0] * ones(self.problem_size)
        ub = self.domain_range[1] * ones(self.problem_size)

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
        lb = lb / I  # Equation (2.10) in the paper
        ub = ub / I  # Equation (2.10) in the paper

        # Move the interval of [lb ub] around the antlion [lb+anlion ub+antlion]
        if rand() < 0.5:
            lb = lb + solution # Equation(2.8) in the paper
        else:
            lb = -lb + solution
        if rand() < 0.5:
            ub = ub + solution  # Equation(2.9) in the paper
        else:
            ub = -ub + solution

        # This function creates n random walks and normalize according to lb and ub vectors,
        temp = []
        for k in range(0, self.problem_size):
            X = cumsum(2 * (rand(self.epoch, 1) > 0.5) - 1)
            a = min(X)
            b = max(X)
            c = lb[k]       # [a b] - -->[c d]
            d = ub[k]
            X_norm = ((X - a)* (d - c)) / (b - a) + c    # Equation(2.7) in the paper
            temp.append(X_norm)
        return array(temp)


    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop_new, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            fit_list = array([item[self.ID_FIT] for item in pop])
            # This for loop simulate random walks
            for i in range(0, self.pop_size):
                # Select ant lions based on their fitness (the better anlion the higher chance of catching ant)
                rolette_index = self._roulette_wheel_selection__(1. / fit_list)
                if rolette_index == -1:
                    rolette_index = 1

                # RA is the random walk around the selected antlion by rolette wheel
                RA = self._random_walk_around_antlion__(pop[rolette_index][self.ID_POS], epoch)

                # RE is the random walk around the elite (best antlion so far)
                RE = self._random_walk_around_antlion__(g_best[self.ID_POS], epoch)

                temp = (RA[:, epoch] + RE[:, epoch]) / 2        # Equation(2.13) in the paper
                # Bound checking (bring back the antlions of ants inside search space if they go beyonds the boundaries
                temp = self._amend_solution_faster__(temp)
                fit = self._fitness_model__(temp)
                pop_new[i] = [temp, fit]

            # Update antlion positions and fitnesses based of the ants (if an ant becomes fitter than an antlion we
            #   assume it was caught by the antlion and the antlion update goes to its position to build the trap)
            pop = pop + pop_new
            pop = sorted(pop, key=lambda item: item[self.ID_FIT])
            pop = pop[:self.pop_size]

            # Update the position of elite if any antlinons becomes fitter than it
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)

            # Keep the elite in the population
            pop[self.ID_MIN_PROB] = deepcopy(g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class BaseALO(OriginalALO):
    """
    The is my version of: Ant Lion Optimizer (ALO)
        (The Ant Lion Optimizer)
    Link:
        Using matrix for better performance. Change the flow of updating new solution. Make it better then original one
    """
    def __init__(self, root_paras=None, alo_paras=None):
        OriginalALO.__init__(self, root_paras, alo_paras)

    def _random_walk_around_antlion__(self, solution, current_epoch):
        # Make the bounded vector
        lb = self.domain_range[0] * ones(self.problem_size)
        ub = self.domain_range[1] * ones(self.problem_size)

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
        lb = lb / I  # Equation (2.10) in the paper
        ub = ub / I  # Equation (2.10) in the paper

        # Move the interval of [lb ub] around the antlion [lb+anlion ub+antlion]. Eq 2.8, 2.9
        lb = lb + solution if rand() < 0.5 else -lb + solution
        ub = ub + solution if rand() < 0.5 else -ub + solution

        # This function creates n random walks and normalize according to lb and ub vectors,
        ## Using matrix and vector for better performance
        X = array([cumsum(2 * (rand(self.epoch, 1) > 0.5) - 1) for _ in range(0, self.problem_size)])
        a = min(X, axis=1)
        b = max(X, axis=1)
        temp1 = reshape((ub - lb) / (b - a), (self.problem_size, 1))
        temp0 = X - reshape(a, (self.problem_size, 1))
        X_norm = temp0 * temp1 + reshape(lb, (self.problem_size, 1))
        return X_norm

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop_new, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            fit_list = array([item[self.ID_FIT] for item in pop])
            # This for loop simulate random walks
            for i in range(0, self.pop_size):
                # Select ant lions based on their fitness (the better anlion the higher chance of catching ant)
                rolette_index = self._roulette_wheel_selection__(1. / fit_list)
                if rolette_index == -1:
                    rolette_index = 1

                # RA is the random walk around the selected antlion by rolette wheel
                RA = self._random_walk_around_antlion__(pop[rolette_index][self.ID_POS], epoch)

                # RE is the random walk around the elite (best antlion so far)
                RE = self._random_walk_around_antlion__(g_best[self.ID_POS], epoch)

                temp = (RA[:, epoch] + RE[:, epoch]) / 2        # Equation(2.13) in the paper
                # Bound checking (bring back the antlions of ants inside search space if they go beyonds the boundaries
                temp = self._amend_solution_faster__(temp)
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop_new[i] = [temp, fit]

            # Change the flow of how solution updated
            pop = deepcopy(pop_new)
            # Update the position of elite if any antlinons becomes fitter than it
            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            # Keep the elite in the population
            pop[self.ID_MIN_PROB] = deepcopy(g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

