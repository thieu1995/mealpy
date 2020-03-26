#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:00, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, normal
from numpy import zeros, exp, mean
from mealpy.root import Root


class BaseBA(Root):
    """
    This is my version of: Bat-Inspired Algorithm (A little bit better then both 2 original version)
    - No need A parameter
    - I changed the process.
        + 1st: We proceed exploration phase (using frequency)
        + 2nd: If new solution has better fitness we replace the old solution
        + 3rd: Otherwise, we proceed exploitation phase (using finding around the best solution so far)
    """
    ID_POS = 0  # position
    ID_FIT = 1  # fitness
    ID_VEL = 2  # velocity

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, r=0.95, pf=(0, 10)):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.r = r  # (r_min, r_max): pulse rate / emission rate
        self.pf = pf  # (pf_min, pf_max): pulse frequency

    def _create_solution__(self, minmax=0):
        """  This algorithm has different encoding mechanism, so we need to override this method
                x: current position
                fit: fitness
                v: velocity of this bird (same number of dimension of x)
        """
        x = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fit = self._fitness_model__(solution=x, minmax=minmax)
        v = zeros(self.problem_size)
        return [x, fit, v]

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                pf = self.pf[0] + (self.pf[1] - self.pf[0]) * uniform()  # Eq. 2
                v = uniform() * pop[i][self.ID_VEL] + (g_best[self.ID_POS] - pop[i][self.ID_POS]) * pf  # Eq. 3
                x = pop[i][self.ID_POS] + v  # Eq. 4
                x = self._amend_solution_faster__(x)
                fit = self._fitness_model__(x)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [x, fit, v]
                else:
                    if uniform() > self.r:
                        x = g_best[self.ID_POS] + 0.001 * normal(self.domain_range[0], self.domain_range[1],
                                                                 self.problem_size)
                        x = self._amend_solution_faster__(x)
                        fit = self._fitness_model__(x)
                        if fit < pop[i][self.ID_FIT]:
                            pop[i] = [x, fit, v]

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalBA(BaseBA):
    """
    This is the original version of: Bat-Inspired Algorithm
    - The value of A and r parameters are constant
    - A New Metaheuristic Bat-Inspired Algorithm
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, A=0.8, r=0.95, pf=(0, 10)):
        BaseBA.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size, r, pf)
        self.A = A  # (A_min, A_max): loudness
        self.r = r  # (r_min, r_max): pulse rate / emission rate

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                pf = self.pf[0] + (self.pf[1] - self.pf[0]) * uniform()  # Eq. 2
                v = pop[i][self.ID_VEL] + (pop[i][self.ID_POS] - g_best[self.ID_POS]) * pf  # Eq. 3
                x = pop[i][self.ID_POS] + v  # Eq. 4

                ## Local Search around g_best solution
                if uniform() > self.r:
                    x = g_best[self.ID_POS] + 0.001 * normal(self.problem_size)  # gauss
                x = self._amend_solution_faster__(x)
                fit = self._fitness_model__(x)

                ## Replace the old solution by the new one when its has better fitness.
                ##  and then update loudness and emission rate
                if fit < pop[i][self.ID_FIT] and uniform() < self.A:
                    pop[i] = [x, fit, v]

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class BasicBA(OriginalBA):
    """
    This is a basic version of: Bat-Inspired Algorithm
    - The value of A and r are changing after each iteration
    - A New Metaheuristic Bat-Inspired Algorithm
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, A=(0.2, 0.8), r=(0.2, 0.95), pf=(0, 10)):
        OriginalBA.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size, A, r, pf)
        self.A = A  # (A_min, A_max): loudness
        self.r = r  # (r_min, r_max): pulse rate / emission rate
        self.pf = pf

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)

        alpha = gamma = 0.975  # original paper
        ## Parallel in each iterations. All bats move together each time.
        a = uniform(self.A[0], self.A[1])
        r_0 = uniform(self.r[0], self.r[1])
        r = r_0

        mean_a = mean(a)
        for epoch in range(self.epoch):
            for i in range(self.pop_size):

                pf = self.pf[0] + (self.pf[1] - self.pf[0]) * uniform()  # Eq. 2
                v = pop[i][self.ID_VEL] + (pop[i][self.ID_POS] - g_best[self.ID_POS]) * pf  # Eq. 3
                x = pop[i][self.ID_POS] + v  # Eq. 4

                ## Local Search around g_best solution
                if uniform() > r:
                    x = g_best[self.ID_POS] + mean_a * uniform(self.problem_size)
                x = self._amend_solution_faster__(x)
                fit = self._fitness_model__(x)

                ## Replace the old solution by the new one when its has better fitness.
                ##  and then update loudness and emission rate
                if fit < pop[i][self.ID_FIT] and uniform() < a:
                    pop[i] = [x, fit, v]
                    a = alpha * a
                    r = r_0 * (1 - exp(-gamma * (epoch + 1)))

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train