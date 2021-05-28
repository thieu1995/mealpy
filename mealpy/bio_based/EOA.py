#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice, randint
from numpy import where, array, mean
from copy import deepcopy
from mealpy.root import Root


class BaseEOA(Root):
    """
    My modified version of: Earthworm Optimisation Algorithm (EOA)
        (Earthworm optimisation algorithm: a bio-inspired metaheuristic algorithm for global optimisation problems)
    Link:
        http://doi.org/10.1504/IJBIC.2015.10004283
        https://www.mathworks.com/matlabcentral/fileexchange/53479-earthworm-optimization-algorithm-ewa
    Notes:
        + The original version from matlab code above will not working well, even with small dimensions.
        + I changed updating process
        + Changed cauchy process using x_mean
        + Used global best solution
        + Remove third loop for faster
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 p_c=0.9, p_m=0.01, n_best=2, alpha=0.98, beta=1, gamma=0.9, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.p_c = p_c              # default = 0.9, crossover probability
        self.p_m = p_m              # default = 0.01 initial mutation probability
        self.n_best = n_best        # default = 2, how many of the best earthworm to keep from one generation to the next
        self.alpha = alpha      # default = 0.98, similarity factor
        self.beta = beta        # default = 1, the initial proportional factor
        self.gamma = gamma      # default = 0.9, a constant that is similar to cooling factor of a cooling schedule in the simulated annealing.

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop_sorted = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        pop_best = deepcopy(pop_sorted[:self.n_best])
        g_best = deepcopy(pop_sorted[self.ID_MIN_PROB])
        beta = self.beta

        for epoch in range(self.epoch):
            beta = self.gamma * beta
            ## Begin the Earthworm Optimization Algorithm process
            for i in range(0, self.pop_size):
                ### Reproduction 1: the first way of reproducing
                x_t1 = self.lb + self.ub - self.alpha * pop[i][self.ID_POS]

                ### Reproduction 2: the second way of reproducing
                if i >= self.n_best:                    ### Select two parents to mate and create two children
                    idx = int(self.pop_size * 0.2)
                    if uniform() < 0.5:                 ## 80% parents selected from best population
                        idx1, idx2 = choice(range(0, idx), 2, replace=False)
                    else:                               ## 20% left parents selected from worst population (make more diversity)
                        idx1, idx2 = choice(range(idx, self.pop_size), 2, replace=False)
                    r = uniform()
                    x_child = r * pop[idx2][self.ID_POS] + (1 - r) * pop[idx1][self.ID_POS]
                else:
                    r1 = randint(0, self.pop_size)
                    x_child = pop[r1][self.ID_POS]
                x_t1 = beta * x_t1 + (1.0-beta) * x_child
                x_t1 = self.amend_position_faster(x_t1)
                fit_t1 = self.get_fitness_position(x_t1)
                if fit_t1 < pop[i][self.ID_FIT]:
                    pop[i] = [x_t1, fit_t1]

            pos_list = array([item[self.ID_POS] for item in pop])
            x_mean = mean(pos_list, axis=0)
            ## Cauchy mutation (CM)
            cauchy_w = deepcopy(g_best[self.ID_POS])
            for i in range(self.n_best, self.pop_size):     # Don't allow the elites to be mutated
                cauchy_w = where(uniform(0, 1, self.problem_size) < self.p_m, x_mean, cauchy_w)
                x_t1 = (cauchy_w + g_best[self.ID_POS]) / 2
                x_t1 = self.amend_position_faster(x_t1)
                fit_t1 = self.get_fitness_position(x_t1)
                if fit_t1 < pop[i][self.ID_FIT]:
                    pop[i] = [x_t1, fit_t1]

            ## Elitism Strategy: Replace the worst with the previous generation's elites.
            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            for i in range(0, self.n_best):
                pop[self.pop_size-i-1] = deepcopy(pop_best[i])

            ## Make sure the population does not have duplicates.
            new_set = set()
            for idx, obj in enumerate(pop):
                if tuple(obj[self.ID_POS].tolist()) in new_set:
                    pop[idx] = self.create_solution()
                else:
                    new_set.add(tuple(obj[self.ID_POS].tolist()))

            ## Update the pop best
            pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
            current_pop_best = deepcopy(pop[:self.n_best])
            for i in range(0, self.n_best):
                if current_pop_best[i][self.ID_FIT] < pop_best[i][self.ID_FIT]:
                    pop_best[i] = deepcopy(current_pop_best[i])

            g_best = deepcopy(pop_best[0])
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
