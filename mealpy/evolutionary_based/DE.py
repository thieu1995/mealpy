#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:48, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import where, sum, any, mean, array, clip, ones, abs
from numpy.random import uniform, choice, normal, randint, random
from copy import deepcopy
from scipy.stats import cauchy
from mealpy.root import Root

"""
BaseDE: - the very first DE algorithm (Novel mutation strategy for enhancing SHADE and LSHADE algorithms for global numerical optimization)
    strategy = 0: DE/current-to-rand/1/bin
             = 1: DE/best/1/bin             
             = 2: DE/best/2/bin
             = 3: DE/rand/2/bin
             = 4: DE/current-to-best/1/bin
             = 5: DE/current-to-rand/1/bin
"""


class BaseDE(Root):
    """
        The original version of: Differential Evolution (DE)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 wf=0.8, cr=0.9, strategy=0, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.weighting_factor = wf
        self.crossover_rate = cr
        self.strategy = strategy

    def _mutation__(self, current_pos, new_pos):
        pos_new = where(uniform(0, 1, self.problem_size) < self.crossover_rate, current_pos, new_pos)
        return self.amend_position_faster(pos_new)

    def _create_children__(self, pop, g_best):
        pop_child = deepcopy(pop)
        if self.strategy == 0:
            for i in range(0, self.pop_size):
                # Choose 3 random element and different to i
                idx_list = choice(list(set(range(0, self.pop_size)) - {i}), 3, replace=False)
                pos_new = pop[idx_list[0]][self.ID_POS] + self.weighting_factor * (pop[idx_list[1]][self.ID_POS] - pop[idx_list[2]][self.ID_POS])
                pos_new = self._mutation__(pop[i][self.ID_POS], pos_new)
                fit = self.get_fitness_position(pos_new)
                pop_child[i] = [pos_new, fit]
            return pop_child
        elif self.strategy == 1:
            for i in range(0, self.pop_size):
                idx_list = choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                pos_new = g_best[self.ID_POS] + self.weighting_factor * (pop[idx_list[0]][self.ID_POS] - pop[idx_list[1]][self.ID_POS])
                pos_new = self._mutation__(pop[i][self.ID_POS], pos_new)
                fit = self.get_fitness_position(pos_new)
                pop_child[i] = [pos_new, fit]
            return pop_child
        elif self.strategy == 2:
            for i in range(0, self.pop_size):
                idx_list = choice(list(set(range(0, self.pop_size)) - {i}), 4, replace=False)
                pos_new = g_best[self.ID_POS] + self.weighting_factor * (pop[idx_list[0]][self.ID_POS] - pop[idx_list[1]][self.ID_POS]) + \
                          self.weighting_factor * (pop[idx_list[2]][self.ID_POS] - pop[idx_list[3]][self.ID_POS])
                pos_new = self._mutation__(pop[i][self.ID_POS], pos_new)
                fit = self.get_fitness_position(pos_new)
                pop_child[i] = [pos_new, fit]
            return pop_child
        elif self.strategy == 3:
            for i in range(0, self.pop_size):
                idx_list = choice(list(set(range(0, self.pop_size)) - {i}), 5, replace=False)
                pos_new = pop[idx_list[0]][self.ID_POS] + self.weighting_factor * (pop[idx_list[1]][self.ID_POS] - pop[idx_list[2]][self.ID_POS]) + \
                          self.weighting_factor * (pop[idx_list[3]][self.ID_POS] - pop[idx_list[4]][self.ID_POS])
                pos_new = self._mutation__(pop[i][self.ID_POS], pos_new)
                fit = self.get_fitness_position(pos_new)
                pop_child[i] = [pos_new, fit]
            return pop_child
        elif self.strategy == 4:
            for i in range(0, self.pop_size):
                idx_list = choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                pos_new = pop[i][self.ID_POS] + self.weighting_factor * (g_best[self.ID_POS] - pop[i][self.ID_POS]) + \
                          self.weighting_factor * (pop[idx_list[0]][self.ID_POS] - pop[idx_list[1]][self.ID_POS])
                pos_new = self._mutation__(pop[i][self.ID_POS], pos_new)
                fit = self.get_fitness_position(pos_new)
                pop_child[i] = [pos_new, fit]
            return pop_child
        elif self.strategy == 5:
            for i in range(0, self.pop_size):
                idx_list = choice(list(set(range(0, self.pop_size)) - {i}), 3, replace=False)
                pos_new = pop[i][self.ID_POS] + self.weighting_factor * (pop[idx_list[0]][self.ID_POS] - pop[i][self.ID_POS]) + \
                          self.weighting_factor * (pop[idx_list[1]][self.ID_POS] - pop[idx_list[2]][self.ID_POS])
                pos_new = self._mutation__(pop[i][self.ID_POS], pos_new)
                fit = self.get_fitness_position(pos_new)
                pop_child[i] = [pos_new, fit]
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
            pop_child = self._create_children__(pop, g_best)
            # create new pop by comparing fitness of corresponding each member in pop and children
            pop = self._greedy_selection__(pop, pop_child)

            # update global best position
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class JADE(Root):
    """
        The original version of: Differential Evolution (JADE)
        Link:
            JADE: Adaptive Differential Evolution with Optional External Archive
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 miu_f=0.5, miu_cr=0.5, p=0.1, c=0.1, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.miu_f = miu_f              # the initial f, location is changed then that f is good
        self.miu_cr = miu_cr            # the initial cr,
        self.p = p # uniform(0.05, 0.2) # the x_best is select from the top 100p % solutions
        self.c = c # uniform(1/20, 1/5) # the adaptation parameter control value of f and cr

    ### Survivor Selection
    def lehmer_mean(self, list_objects):
        return sum(list_objects**2) / sum(list_objects)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)
        miu_cr = self.miu_cr
        miu_f = self.miu_f
        archive_pop = list()

        for epoch in range(self.epoch):
            list_f = list()
            list_cr = list()

            sorted_pop = sorted(pop, key=lambda x:x[self.ID_FIT])
            for i in range(0, self.pop_size):
                ## Calculate adaptive parameter cr and f
                cr = normal(miu_cr, 0.1)
                cr = clip(cr, 0, 1)
                while True:
                    f = cauchy.rvs(miu_f, 0.1)
                    if f < 0:
                        continue
                    elif f > 1:
                        f = 1
                    break
                top = int(self.pop_size * self.p)
                x_best = sorted_pop[randint(0, top)]
                x_r1 = pop[choice(list(set(range(0, self.pop_size)) - {i}))]
                new_pop = pop + archive_pop
                while True:
                    x_r2 = new_pop[randint(0, len(new_pop))]
                    if any(x_r2[self.ID_POS] - x_r1[self.ID_POS]) and any(x_r2[self.ID_POS] - pop[i][self.ID_POS]):
                        break
                x_new = pop[i][self.ID_POS] + f * (x_best[self.ID_POS] - pop[i][self.ID_POS]) + f * (x_r1[self.ID_POS] - x_r2[self.ID_POS])
                pos_new = where(uniform(0, 1, self.problem_size) < cr, x_new, pop[i][self.ID_POS])
                j_rand = randint(0, self.problem_size)
                pos_new[j_rand] = x_new[j_rand]
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    archive_pop.append(pop[i])
                    list_cr.append(cr)
                    list_f.append(f)
                    pop[i] = [pos_new, fit_new]

            # Randomly remove solution
            temp = len(archive_pop) - self.pop_size
            if temp > 0:
                idx_list = choice(range(0, len(archive_pop)), len(archive_pop) - self.pop_size, replace=False)
                archive_pop_new = []
                for idx, solution in enumerate(archive_pop):
                    if idx not in idx_list:
                        archive_pop_new.append(solution)
                archive_pop = deepcopy(archive_pop_new)

            # Update miu_cr and miu_f
            miu_cr = (1 - self.c) * miu_cr + self.c * mean(array(list_cr))
            miu_f = (1 - self.c) * miu_f + self.c * self.lehmer_mean(array(list_f))

            # update global best position
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class SHADE(Root):
    """
        The original version of: Success-History Adaptation Differential Evolution (SHADE)
        Link:
            Success-History Based Parameter Adaptation for Differential Evolution
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 miu_f=0.5, miu_cr=0.5, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.miu_f = miu_f  # list the initial f,
        self.miu_cr = miu_cr  # list the initial cr,

    ### Survivor Selection
    def weighted_lehmer_mean(self, list_objects, list_weights):
        up = list_weights * list_objects**2
        down = list_weights * list_objects
        return sum(up) / sum(down)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)
        miu_cr = self.miu_cr * ones(self.pop_size)
        miu_f = self.miu_f * ones(self.pop_size)
        archive_pop = list()
        k = 0

        for epoch in range(self.epoch):
            list_f = list()
            list_cr = list()
            list_f_index = list()
            list_cr_index = list()

            list_f_new = ones(self.pop_size)
            list_cr_new = ones(self.pop_size)
            pop_new = deepcopy(pop) # Save all new elements --> Use to update the list_cr and list_f
            pop_old = deepcopy(pop) # Save all old elements --> Use to update cr value
            sorted_pop = sorted(pop, key=lambda x: x[self.ID_FIT])
            for i in range(0, self.pop_size):
                ## Calculate adaptive parameter cr and f
                idx_rand = randint(0, self.pop_size)
                cr = normal(miu_cr[idx_rand], 0.1)
                cr = clip(cr, 0, 1)
                while True:
                    f = cauchy.rvs(miu_f[idx_rand], 0.1)
                    if f < 0:
                        continue
                    elif f > 1:
                        f = 1
                    break
                list_cr_new[i] = cr
                list_f_new[i] = f
                p = uniform(2/self.pop_size, 0.2)
                top = int(self.pop_size * p)
                x_best = sorted_pop[randint(0, top)]
                x_r1 = pop[choice(list(set(range(0, self.pop_size)) - {i}))]
                new_pop = pop + archive_pop
                while True:
                    x_r2 = new_pop[randint(0, len(new_pop))]
                    if any(x_r2[self.ID_POS] - x_r1[self.ID_POS]) and any(x_r2[self.ID_POS] - pop[i][self.ID_POS]):
                        break
                x_new = pop[i][self.ID_POS] + f * (x_best[self.ID_POS] - pop[i][self.ID_POS]) + f * (x_r1[self.ID_POS] - x_r2[self.ID_POS])
                pos_new = where(uniform(0, 1, self.problem_size) < cr, x_new, pop[i][self.ID_POS])
                j_rand = randint(0, self.problem_size)
                pos_new[j_rand] = x_new[j_rand]
                fit_new = self.get_fitness_position(pos_new)
                pop_new[i] = [pos_new, fit_new]

            for i in range(0, self.pop_size):
                if pop_new[i][self.ID_FIT] < pop[i][self.ID_FIT]:
                    list_cr.append(list_cr_new[i])
                    list_f.append(list_f_new[i])
                    list_f_index.append(i)
                    list_cr_index.append(i)
                    pop[i] = pop_new[i]
                    archive_pop.append(deepcopy(pop[i]))

            # Randomly remove solution
            temp = len(archive_pop) - self.pop_size
            if temp > 0:
                idx_list = choice(range(0, len(archive_pop)), len(archive_pop) - self.pop_size, replace=False)
                archive_pop_new = []
                for idx, solution in enumerate(archive_pop):
                    if idx not in idx_list:
                        archive_pop_new.append(solution)
                archive_pop = deepcopy(archive_pop_new)

            # Update miu_cr and miu_f
            if len(list_f) != 0 and len(list_cr) != 0:
                # Eq.13, 14, 10
                list_fit_old = ones(len(list_cr_index))
                list_fit_new = ones(len(list_cr_index))
                idx_increase = 0
                for i in range(0, self.pop_size):
                    if i in list_cr_index:
                        list_fit_old[idx_increase] = pop_old[i][self.ID_FIT]
                        list_fit_new[idx_increase] = pop_new[i][self.ID_FIT]
                        idx_increase += 1
                list_weights = abs(list_fit_new - list_fit_old) / sum(abs(list_fit_new - list_fit_old))
                miu_cr[k] = sum(list_weights * array(list_cr))
                miu_f[k] = self.weighted_lehmer_mean(array(list_f), list_weights)
                k += 1
                if k >= self.pop_size:
                    k = 0

            # update global best position
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class SAP_DE(Root):
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
        super().__init__(obj_func, lb, ub, verbose, kwargs)
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
                    fit_new = self.get_fitness_position(pos_new)
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
                    fit_new = self.get_fitness_position(pos_new)
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

