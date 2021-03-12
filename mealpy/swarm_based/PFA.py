#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:51, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import exp, sqrt, sum
from numpy.random import uniform
from copy import deepcopy
from mealpy.root import Root


class BasePFA(Root):
    """
        The original version of: Pathfinder algorithm (PFA)
            (A new meta-heuristic optimizer: Pathfinder algorithm)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        # Init pop and calculate fitness
        pop = [self.create_solution() for _ in range(self.pop_size)]

        # Find the pathfinder
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        gbest_present = deepcopy(g_best)

        for epoch in range(self.epoch):
            alpha, beta = uniform(1, 2, 2)
            A = uniform(self.lb, self.ub) * exp(-2 * (epoch + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present[self.ID_POS] + 2 * uniform() * (gbest_present[self.ID_POS] - g_best[self.ID_POS]) + A
            temp = self.amend_position_faster(temp)
            fit = self.get_fitness_position(temp)
            g_best = deepcopy(gbest_present)
            if fit < gbest_present[self.ID_FIT]:
                gbest_present = [temp, fit]
            pop[0] = deepcopy(gbest_present)

            ## Update positions of members, check the bound and calculate new fitness
            for i in range(1, self.pop_size):
                temp = deepcopy(pop[i][self.ID_POS])
                pos_new = deepcopy(pop[i][self.ID_POS])

                t1 = beta * uniform() * (gbest_present[self.ID_POS] - temp)
                for k in range(1, self.pop_size):
                    dist = sqrt(sum((pop[k][self.ID_POS] - temp)**2)) / self.problem_size
                    t2 = alpha * uniform() * (pop[k][self.ID_POS] - temp)
                    ## First stabilize the distance
                    t3 = uniform() * (1 - (epoch + 1) * 1.0 / self.epoch) * (dist / (self.ub - self.lb))
                    pos_new += t2 + t3
                ## Second stabilize the population size
                pos_new = (pos_new + t1) / self.pop_size

                ## Update members
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit_new]

                ### Batch size idea
                ## Update the best position found so far (current pathfinder)
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        pop, gbest_present = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, gbest_present)
                else:
                    if (i + 1) % self.pop_size == 0:
                        pop, gbest_present = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, gbest_present)
            self.loss_train.append(gbest_present[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, gbest_present[self.ID_FIT]))
        self.solution = gbest_present
        return gbest_present[self.ID_POS], gbest_present[self.ID_FIT], self.loss_train


class OPFA(BasePFA):
    """
        My opposition-based learning version of: Pathfinder algorithm (PFA)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BasePFA.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def train(self):
        # Init pop and calculate fitness
        pop = [self.create_solution(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        gbest_present = deepcopy(g_best)

        for epoch in range(self.epoch):
            alpha, beta = uniform(1, 2, 2)
            A = uniform(self.lb, self.ub) * exp(-2 * (epoch + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present[self.ID_POS] + 2 * uniform() * (gbest_present[self.ID_POS] - g_best[self.ID_POS]) + A
            temp = self.amend_position_faster(temp)
            fit = self.get_fitness_position(temp)
            g_best = deepcopy(gbest_present)
            if fit < gbest_present[self.ID_FIT]:
                gbest_present = [temp, fit]
            pop[0] = deepcopy(gbest_present)

            ## Update positions of members, check the bound and calculate new fitness
            for i in range(1, self.pop_size):
                temp = deepcopy(pop[i][self.ID_POS])
                pos_new = deepcopy(pop[i][self.ID_POS])

                t1 = beta * uniform() * (gbest_present[self.ID_POS] - temp)
                for k in range(1, self.pop_size):
                    dist = sqrt(sum((pop[k][self.ID_POS] - temp) ** 2)) / self.problem_size
                    t2 = alpha * uniform() * (pop[k][self.ID_POS] - temp)
                    ## First stabilize the distance
                    t3 = uniform() * (1 - (epoch + 1) * 1.0 / self.epoch) * (dist / (self.ub - self.lb))
                    pos_new += t2 + t3
                ## Second stabilize the population size
                pos_new = (pos_new + t1) / self.pop_size

                ## Update members
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit_new]
                else:
                    C_op = self.create_opposition_position(pos_new, gbest_present[self.ID_POS])
                    fit_op = self.get_fitness_position(C_op)
                    if fit_op < pop[i][self.ID_FIT]:
                        pop[i] = [C_op, fit_op]

                ## Update the best position found so far (current pathfinder)
                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        gbest_present = self.update_global_best_solution(pop, self.ID_MIN_PROB, gbest_present)
                else:
                    if (i + 1) % self.pop_size == 0:
                        gbest_present = self.update_global_best_solution(pop, self.ID_MIN_PROB, gbest_present)
            self.loss_train.append(gbest_present[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, gbest_present[self.ID_FIT]))
        self.solution = gbest_present
        return gbest_present[self.ID_POS], gbest_present[self.ID_FIT], self.loss_train


class ImprovedPFA(BasePFA):
    """
        My improved version of: Pathfinder algorithm (PFA)
            + use opposition-based learning
            + use levy-flight trajectory
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BasePFA.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def train(self):
        # Init pop and calculate fitness
        pop = [self.create_solution(minmax=0) for _ in range(self.pop_size)]

        # Find the pathfinder
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        gbest_present = deepcopy(g_best)

        for epoch in range(0, self.epoch):
            alpha, beta = uniform(1, 2, 2)
            A = uniform(self.lb, self.ub) * exp(-2 * (epoch + 1) / self.epoch)

            ## Update the position of pathfinder and check the bound
            temp = gbest_present[self.ID_POS] + 2 * uniform() * (gbest_present[self.ID_POS] - g_best[self.ID_POS]) + A
            temp = self.amend_position_faster(temp)
            fit = self.get_fitness_position(temp)
            g_best = deepcopy(gbest_present)
            if fit < gbest_present[self.ID_FIT]:
                gbest_present = [temp, fit]
            pop[0] = deepcopy(gbest_present)

            ## Update positions of members, check the bound and calculate new fitness
            for i in range(1, self.pop_size):
                temp = deepcopy(pop[i][self.ID_POS])
                pos_new = deepcopy(pop[i][self.ID_POS])

                t1 = beta * uniform() * (gbest_present[self.ID_POS] - temp)
                for k in range(1, self.pop_size):
                    dist = sqrt(sum((pop[k][self.ID_POS] - temp) ** 2)) / self.problem_size
                    t2 = alpha * uniform() * (pop[k][self.ID_POS] - temp)
                    ## First stabilize the distance
                    t3 = uniform() * (1 - (epoch + 1) * 1.0 / self.epoch) * (dist / (self.ub - self.lb))
                    pos_new += t2 + t3
                ## Second stabilize the population size
                pos_new = (pos_new + t1) / self.pop_size

                ## Update members
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit_new]
                else:
                    C_op = self.create_opposition_position(pos_new, gbest_present[self.ID_POS])
                    fit_op = self.get_fitness_position(C_op)
                    if fit_op < pop[i][self.ID_FIT]:
                        pop[i] = [C_op, fit_op]
                    else:
                        ## Using Levy-flight to boost algorithm's convergence speed
                        pos_new = self.levy_flight(epoch, pop[i][self.ID_POS], gbest_present[self.ID_POS])
                        fit_new = self.get_fitness_position(pos_new)
                        pop[i] = [pos_new, fit_new]
                        if fit_new < pop[i][self.ID_FIT]:
                            pop[i] = [pos_new, fit_new]
            ## Make sure the population does not have duplicates.
            new_set = set()
            for idx, obj in enumerate(pop):
                if tuple(obj[self.ID_POS].tolist()) in new_set:
                    pos_new = self.levy_flight(epoch, pop[idx][self.ID_POS], gbest_present[self.ID_POS])
                    fit_new = self.get_fitness_position(pos_new)
                    pop[idx] = [pos_new, fit_new]
                else:
                    new_set.add(tuple(obj[self.ID_POS].tolist()))

            ## Update the best position found so far (current pathfinder)
            gbest_present = self.update_global_best_solution(pop, self.ID_MIN_PROB, gbest_present)
            self.loss_train.append(gbest_present[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, gbest_present[self.ID_FIT]))
        self.solution = gbest_present
        return gbest_present[self.ID_POS], gbest_present[self.ID_FIT], self.loss_train

