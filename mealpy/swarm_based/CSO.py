#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:09, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, random
from numpy import where
from random import sample, choice
from copy import deepcopy
from mealpy.root import Root


class BaseCSO(Root):
    """
        The original version of: Cat Swarm Optimization (CSO)
    """
    ID_POS = 0      # position of the cat
    ID_FIT = 1      # fitness
    ID_VEL = 2      # velocity
    ID_FLAG = 3     # status

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 mixture_ratio=0.15, smp=10, spc=False, cdc=0.8, srd=0.15, c1=0.4, w_minmax=(0.4, 0.9), selected_strategy=1, **kwargs):
        """
        # mixture_ratio - joining seeking mode with tracing mode
        # smp - seeking memory pool, 10 clones  (larger is better but time-consuming)
        # spc - self-position considering
        # cdc - counts of dimension to change  (larger is more diversity but slow convergence)
        # srd - seeking range of the selected dimension (smaller is better but slow convergence)
        # w_minmax - same in PSO
        # c1 - same in PSO
        # selected_strategy : 0: best fitness, 1: tournament, 2: roulette wheel, else: random  (decrease by quality)
        """
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs=kwargs)
        self.epoch =  epoch
        self.pop_size = pop_size
        self.mixture_ratio = mixture_ratio
        self.smp = smp
        self.spc = spc
        self.cdc = cdc
        self.srd = srd
        self.c1 = c1         # Still using c1 and r1 but not c2, r2
        self.w_min = w_minmax[0]
        self.w_max = w_minmax[1]
        self.selected_strategy = selected_strategy

    def create_solution(self, minmax=0):
        """
                x: current position of cat
                v: vector v of cat (same amount of dimension as x)
                flag: the stage of cat, seeking (looking/finding around) or tracing (chasing/catching)
        """
        x = uniform(self.lb, self.ub)
        v = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(x, minmax)
        flag = False                        # False: seeking mode , True: tracing mode
        if random() < self.mixture_ratio:
            flag = True
        return [x, fitness, v, flag]

    def _seeking_mode__(self, cat):
        candidate_cats = []
        clone_cats = [deepcopy(cat) for _ in range(self.smp)]
        if self.spc:
            candidate_cats.append(deepcopy(cat))
            clone_cats = [deepcopy(cat) for _ in range(self.smp - 1)]

        for clone in clone_cats:
            idx = sample(range(0, self.problem_size), int(self.cdc * self.problem_size))
            pos_new1 = clone[self.ID_POS] * (1 + self.srd)
            pos_new2 = clone[self.ID_POS] * (1 - self.srd)

            pos_new = where(uniform(0, 1, self.problem_size) < 0.5, pos_new1, pos_new2)
            pos_new[idx] = clone[self.ID_POS][idx]
            pos_new = self.amend_position_faster(pos_new)
            fit_new = self.get_fitness_position(pos_new)
            candidate_cats.append([pos_new, fit_new, clone[self.ID_VEL], clone[self.ID_FLAG]])

        if self.selected_strategy == 0:                # Best fitness-self
            cat = sorted(candidate_cats, key=lambda cat: cat[self.ID_FIT])[0]
        elif self.selected_strategy == 1:              # Tournament
            k_way = 4
            idx = sample(range(0, self.smp), k_way)
            cats_k_way = [candidate_cats[_] for _ in idx]
            cat = sorted(cats_k_way, key=lambda cat: cat[self.ID_FIT])[0]
        elif self.selected_strategy == 2:              ### Roul-wheel selection
            list_fitness = [candidate_cats[u][self.ID_FIT] for u in range(0, len(candidate_cats))]
            idx = self.get_index_roulette_wheel_selection(list_fitness)
            cat = candidate_cats[idx]
        else:
            cat = choice(candidate_cats)                # Random
        return cat


    def train(self):
        pop = [self.create_solution() for _ in range(0, self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
            for i in range(0, self.pop_size):
                if pop[i][self.ID_FLAG]:            # tracing mode
                    pos_new = pop[i][self.ID_POS] + w * pop[i][self.ID_VEL] + uniform() * self.c1 * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                    fit_new = self.get_fitness_position(pos_new)
                    pop[i][self.ID_POS] = pos_new
                    pop[i][self.ID_FIT] = fit_new
                else:
                    pop[i] = self._seeking_mode__(pop[i])

                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            for i in range(0, self.pop_size):
                if uniform() < self.mixture_ratio:
                    pop[i][self.ID_FLAG] = True
                else:
                    pop[i][self.ID_FLAG] = False

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
