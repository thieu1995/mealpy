#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:00, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, normal
from numpy import zeros, exp, mean
from mealpy.root import Root


class BaseBA(Root):
    """
    My modified version of: Bat-Inspired Algorithm (A little bit better than both 2 original version)
    - No need A parameter
    - I changed the process.
        + 1st: We proceed exploration phase (using frequency)
        + 2nd: If new position has better fitness we replace the old position
        + 3rd: Otherwise, we proceed exploitation phase (using finding around the best position so far)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, r=0.95, pf=(0, 10), **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs=kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.r = r              # (r_min, r_max): pulse rate / emission rate
        self.pf = pf            # (pf_min, pf_max): pulse frequency

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)
        list_velocity = zeros((self.pop_size, self.problem_size))

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                pf = self.pf[0] + (self.pf[1] - self.pf[0]) * uniform()                                 # Eq. 2
                v = uniform() * list_velocity[i] + (g_best[self.ID_POS] - pop[i][self.ID_POS]) * pf     # Eq. 3
                x = pop[i][self.ID_POS] + v                                                             # Eq. 4
                x = self.amend_position_faster(x)
                fit = self.get_fitness_position(x)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [x, fit, v]
                else:
                    if uniform() > self.r:
                        x = g_best[self.ID_POS] + 0.01 * uniform(self.lb, self.ub)
                        x = self.amend_position_faster(x)
                        fit = self.get_fitness_position(x)
                        if fit < pop[i][self.ID_FIT]:
                            pop[i] = [x, fit, v]
                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalBA(Root):
    """
        The original version of: Bat-Inspired Algorithm (BA)
            A New Metaheuristic Bat-Inspired Algorithm
        Notes:
            The value of A and r parameters are constant
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, A=0.8, r=0.95, pf=(0, 10), **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs=kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.r = r          # (r_min, r_max): pulse rate / emission rate
        self.pf = pf        # (pf_min, pf_max): pulse frequency
        self.A = A          # (A_min, A_max): loudness
        self.r = r          # (r_min, r_max): pulse rate / emission rate

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)
        list_velocity = zeros((self.pop_size, self.problem_size))

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                pf = self.pf[0] + (self.pf[1] - self.pf[0]) * uniform()                     # Eq. 2
                v = list_velocity[i] + (pop[i][self.ID_POS] - g_best[self.ID_POS]) * pf     # Eq. 3
                x = pop[i][self.ID_POS] + v                                                 # Eq. 4

                ## Local Search around g_best_position position
                if uniform() > self.r:
                    x = g_best[self.ID_POS] + 0.001 * normal(self.problem_size)  # gauss
                x = self.amend_position_faster(x)
                fit = self.get_fitness_position(x)

                ## Replace the old position by the new one when its has better fitness.
                ##  and then update loudness and emission rate
                if fit < pop[i][self.ID_FIT] and uniform() < self.A:
                    pop[i] = [x, fit]

                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class BasicBA(Root):
    """
        The original versino of: Bat-Inspired Algorithm (BA)
            A New Metaheuristic Bat-Inspired Algorithm
        Notes:
            The value of A and r are changing after each iteration
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 A=(0.2, 0.8), r=(0.2, 0.95), pf=(0, 10), **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs=kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.r = r          # (r_min, r_max): pulse rate / emission rate
        self.pf = pf        # (pf_min, pf_max): pulse frequency
        self.A = A          # (A_min, A_max): loudness
        self.r = r          # (r_min, r_max): pulse rate / emission rate

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)
        list_velocity = zeros((self.pop_size, self.problem_size))

        alpha = gamma = 0.975  # original paper
        ## Parallel in each iterations. All bats move together each time.
        a = uniform(self.A[0], self.A[1])
        r_0 = uniform(self.r[0], self.r[1])
        r = r_0

        mean_a = mean(a)
        for epoch in range(self.epoch):
            for i in range(self.pop_size):

                pf = self.pf[0] + (self.pf[1] - self.pf[0]) * uniform()                     # Eq. 2
                v = list_velocity[i] + (pop[i][self.ID_POS] - g_best[self.ID_POS]) * pf     # Eq. 3
                x = pop[i][self.ID_POS] + v                                                 # Eq. 4

                ## Local Search around g_best_position position
                if uniform() > r:
                    x = g_best[self.ID_POS] + mean_a * uniform(self.problem_size)
                x = self.amend_position_faster(x)
                fit = self.get_fitness_position(x)

                ## Replace the old position by the new one when its has better fitness.
                ##  and then update loudness and emission rate
                if fit < pop[i][self.ID_FIT] and uniform() < a:
                    pop[i] = [x, fit]
                    a = alpha * a
                    r = r_0 * (1 - exp(-gamma * (epoch + 1)))

                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train