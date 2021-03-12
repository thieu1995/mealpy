#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:48, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import where, sum
from numpy.random import uniform, choice, normal, randint, random
from copy import deepcopy
from mealpy.root import Root


class BaseDE(Root):
    """
        The original version of: Differential Evolution (DE)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 wf=0.8, cr=0.9, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.weighting_factor = wf
        self.crossover_rate = cr

    def _mutation__(self, p0, p1, p2, p3):
        ### Remove third loop here
        pos_new = deepcopy(p0)
        temp = p1 + self.weighting_factor * (p2 - p3)
        pos_new = where(uniform(0, 1, self.problem_size) < self.crossover_rate, temp, pos_new)
        return self.amend_position_faster(pos_new)

    def _create_children__(self, pop):
        pop_child = deepcopy(pop)
        for i in range(0, self.pop_size):
            # Choose 3 random element and different to i
            temp = choice(list(set(range(0, self.pop_size)) - {i}), 3, replace=False)
            #create new child and append in children array
            child = self._mutation__(pop[i][self.ID_POS], pop[temp[0]][self.ID_POS], pop[temp[1]][self.ID_POS], pop[temp[2]][self.ID_POS])
            fit = self.get_fitness_position(child)
            pop_child[i] = [child, fit]
        return pop_child

    ### Survivor Selection
    def _greedy_selection__(self, pop_old=None, pop_new=None):
        pop = [pop_new[i] if pop_new[i][self.ID_FIT] < pop_old[i][self.ID_FIT] else pop_old[i] for i in range(self.pop_size)]
        return pop

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # create children
            pop_child = self._create_children__(pop)
            # create new pop by comparing fitness of corresponding each member in pop and children
            pop = self._greedy_selection__(pop, pop_child)

            # update global best position
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class DESAP(Root):
    """
        The original version of: Differential Evolution with Self-Adaptive Populations
        Link:
            Exploring dynamic self-adaptive populations in differential evolution
    """
    ID_CR = 2
    ID_MR = 3
    ID_PS = 4

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 wf=0.8, cr=0.9, F=1, branch="ABS", **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.weighting_factor = wf
        self.crossover_rate = cr
        self.F = F
        self.M = pop_size
        self.branch = branch            # absolute (ABS) or relative (REL)

    def create_solution(self, minmax=0):
        position = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position, minmax=minmax)
        crossover_rate = uniform(0, 1)
        mutation_rate = uniform(0, 1)
        if self.branch == "ABS":
            pop_size = int(10 * self.problem_size + normal(0, 1))
        elif self.branch == "REL":
            pop_size = int(10 * self.problem_size + uniform(-0.5, 0.5))
        return [position, fitness, crossover_rate, mutation_rate, pop_size]

    def edit_to_range(self, var=None, lower=0, upper=1, func_value=None):
        while var <= lower or var >= upper:
            if var <= lower:
                var += func_value()
            if var >= upper:
                var -= func_value()
        return var

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)
        m_new = self.pop_size

        for epoch in range(self.epoch):

            for i in range(0, self.pop_size):
            ### create children
                # Choose 3 random element and different to i
                idxs = choice(list(set(range(0, self.pop_size)) - {i}), 3, replace=False)
                j = randint(0, self.pop_size)
                self.F = uniform(0, 1)

                sol_new = deepcopy(pop[idxs[0]])
                ## Crossover
                if uniform(0, 1) < pop[i][self.ID_CR] or i == j:
                    pos_new = pop[idxs[0]][self.ID_POS] + self.F * (pop[idxs[1]][self.ID_POS] - pop[idxs[2]][self.ID_POS])
                    cr_new = pop[idxs[0]][self.ID_CR] + self.F * (pop[idxs[1]][self.ID_CR] - pop[idxs[2]][self.ID_CR])
                    mr_new = pop[idxs[0]][self.ID_MR] + self.F * (pop[idxs[1]][self.ID_MR] - pop[idxs[2]][self.ID_MR])
                    if self.branch == "ABS":
                        ps_new = pop[idxs[0]][self.ID_PS] + int(self.F * (pop[idxs[1]][self.ID_PS] - pop[idxs[2]][self.ID_PS]))
                    elif self.branch == "REL":
                        ps_new = pop[idxs[0]][self.ID_PS] + self.F * (pop[idxs[1]][self.ID_PS] - pop[idxs[2]][self.ID_PS])
                    pos_new = self.amend_position_faster(pos_new)
                    fit_new = self.get_fitness_solution(pos_new)
                    cr_new = self.edit_to_range(cr_new, 0, 1, random)
                    mr_new = self.edit_to_range(mr_new, 0, 1, random)
                    sol_new = [pos_new, fit_new, cr_new, mr_new, ps_new]

                ## Mutation
                if uniform(0, 1) < pop[idxs[0]][self.ID_MR]:
                    pos_new = pop[i][self.ID_POS] + normal(0, pop[idxs[0]][self.ID_MR])
                    cr_new = normal(0, 1)
                    mr_new = normal(0, 1)
                    if self.branch == "ABS":
                        ps_new = pop[i][self.ID_PS] + int(normal(0.5, 1))
                    elif self.branch == "REL":
                        ps_new = pop[i][self.ID_PS] + normal(0, pop[idxs[0]][self.ID_MR])
                    pos_new = self.amend_position_faster(pos_new)
                    fit_new = self.get_fitness_solution(pos_new)
                    sol_new = [pos_new, fit_new, cr_new, mr_new, ps_new]
                pop[i] = deepcopy(sol_new)

            # Calculate new population size
            total = sum([pop[i][self.ID_PS] for i in range(0, self.pop_size)])
            if self.branch == "ABS":
                m_new = int(total / self.pop_size)
            elif self.branch == "REL":
                m_new = int(self.pop_size + total)
            if m_new <= 4:
                m_new = self.M + int(uniform(0, 4))
            elif m_new > 4 * self.M:
                m_new = self.M - int(uniform(0, 4))

            ## Change population by population size
            if m_new <= self.pop_size:
                pop = pop[:m_new]
            else:
                pop_sorted = sorted(pop, key=lambda x: x[self.ID_FIT])
                best = deepcopy(pop_sorted[0])
                pop_best = [best for i in range(0, m_new - self.pop_size)]
                pop = pop + pop_best
            self.pop_size = m_new

            # update global best position
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

