#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice, randint
from numpy import sum, sort, array_equal, ones, array
from copy import deepcopy
from mealpy.root import Root


class BaseEOA(Root):
    """
    The original version of: Earthworm Optimisation Algorithm (EOA)
        (Earthworm optimisation algorithm: a bio-inspired metaheuristic algorithm for global optimisation problems)
    Link:
        http://doi.org/10.1504/IJBIC.2015.10004283
        https://www.mathworks.com/matlabcentral/fileexchange/53479-earthworm-optimization-algorithm-ewa
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True,
                 epoch=750, pop_size=100, p_c=0.9, p_m=0.01, n_best=2, alpha=0.98, beta=1, gamma=0.9):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.p_c = p_c              # default = 0.9, crossover probability
        self.p_m = p_m              # default = 0.01 initial mutation probability
        self.n_best = n_best        # default = 2, how many of the best earthworm to keep from one generation to the next
        self.alpha = alpha      # default = 0.98, similarity factor
        self.beta = beta        # default = 1, the initial proportional factor
        self.gamma = gamma      # default = 0.9, a constant that is similiar to cooling factor of a cooling schedule in the simulated annealing.

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop_sorted = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        pop_best = deepcopy(pop_sorted[:self.n_best])
        g_best = deepcopy(pop_sorted[self.ID_MIN_PROB])
        beta = self.beta

        for epoch in range(self.epoch):
            beta = self.gamma * beta
            ## Begin the Earthworm Optimization Algorithm process
            for i in range(0, self.pop_size):
                ### Reproduction 1: the first way of reproducing
                x_t1 = ones(self.problem_size) * (self.domain_range[1] + self.domain_range[0]) - self.alpha * pop[i][self.ID_POS]

                ### Reproduction 2: the second way of reproducing
                if i >= self.n_best:
                    ### Select two parents to mate and create two children
                    idx = int(self.pop_size * 0.2)
                    if uniform() < 0.5:       ## 80% parents selected from best population
                        idx1, idx2 = choice(range(0, idx), 2, replace=False)
                    else:                               ## 20% left parents selected from worst population (make more diversity)
                        idx1, idx2 = choice(range(idx, self.pop_size), 2, replace=False)

                    ### Uniform crossover
                    # x_child = deepcopy(pop[idx1][self.ID_POS])
                    # for j in range(0, self.problem_size):
                    #     if self.p_c > uniform():
                    #         x_child[j] = pop[idx1][self.ID_POS][j] if uniform() < 0.5 else pop[idx2][self.ID_POS][j]
                    #     else:
                    #         x_child[j] = pop[idx2][self.ID_POS][j] if uniform() < 0.5 else pop[idx1][self.ID_POS][j]
                    r = uniform()
                    x_child = r * pop[idx2][self.ID_POS] + (1 - r) * pop[idx1][self.ID_POS]
                else:
                    r1 = randint(0, self.pop_size)
                    x_child = pop[r1][self.ID_POS]
                x_t1 = beta * x_t1 + (1.0-beta) * x_child
                x_t1 = self._amend_solution_faster__(x_t1)
                fit_t1 = self._fitness_model__(x_t1)
                if pop[i][self.ID_FIT] > fit_t1:
                    pop[i] = [x_t1, fit_t1]

            pos_list = array([item[self.ID_POS] for item in pop])
            ## Cauchy mutation (CM)
            cauchy_w = deepcopy(g_best[self.ID_POS])
            for i in range(0, self.n_best, self.pop_size):     # Don't allow the elites to be mutated
                for j in range(0, self.problem_size):
                    if self.p_m > uniform():
                        cauchy_w[j] = sum(pos_list[:j]) / self.pop_size
                x_t1 = (cauchy_w + g_best[self.ID_POS]) / 2
                x_t1 = self._amend_solution_faster__(x_t1)
                fit_t1 = self._fitness_model__(x_t1)
                if fit_t1 < pop[i][self.ID_FIT]:
                    pop[i] = [x_t1, fit_t1]

            ## Elitism Strategy: Replace the worst with the previous generation's elites.
            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            for i in range(0, self.n_best):
                pop[self.pop_size-i-1] = deepcopy(pop_best[i])

            ## Make sure the population does not have duplicates.
            for i in range(0, self.pop_size):
                temp1 = sort(pop[i][self.ID_POS])
                for j in range(0, self.pop_size):
                    temp2 = sort(pop[j][self.ID_POS])
                    if array_equal(temp1, temp2):
                        pop[j][self.ID_POS] = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)

            ## Update the pop best
            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            current_pop_best = deepcopy(pop[:self.n_best])
            for i in range(0, self.n_best):
                if current_pop_best[i][self.ID_FIT] < pop_best[i][self.ID_FIT]:
                    pop_best[i] = deepcopy(current_pop_best[i])

            g_best = deepcopy(pop_best[0])
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
