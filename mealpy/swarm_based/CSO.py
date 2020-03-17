#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:09, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, random
from random import sample, choice
from copy import deepcopy
from mealpy.root import Root


class BaseCSO(Root):
    """
        This is basic version of Cat Swarm Optimization
    """
    ID_POS = 0      # position of the cat
    ID_FIT = 1      # fitness
    ID_VEL = 2      # velocity
    ID_FLAG = 3     # status

    def __init__(self, root_paras=None, epoch=750, pop_size=100, mixture_ratio=0.15, smp=20, spc=False, cdc=0.8,
                 srd=0.15, c1=0.4, w_minmax=(0.4, 0.9), selected_strategy=0):
        """
        # mixture_ratio - joining seeking mode with tracing mode
        # smp - seeking memory pool, 10 clones  (lon cang tot, nhung ton time hon)
        # spc - self-position considering
        # cdc - counts of dimension to change  (lon cang tot)
        # srd - seeking range of the selected dimension (nho thi tot nhung search lau hon)
        # w_minmax - same in PSO
        # c1 - same in PSO
        # selected_strategy : 0: best fitness, 1: tournament, 2: roulette wheel, 3: random  (decrease by quality)
        """
        Root.__init__(self, root_paras)
        self.epoch =  epoch
        self.pop_size = pop_size
        self.mixture_ratio = mixture_ratio
        self.smp = smp
        self.spc = spc
        self.cdc = cdc
        self.srd = srd
        self.c1 = c1  # Still using c1 and r1 but not c2, r2
        self.w_min = w_minmax[0]
        self.w_max = w_minmax[1]
        self.selected_strategy = selected_strategy

    def _create_solution__(self, minmax=0):
        """
                x: current position of cat
                v: vector v of cat (same amount of dimension as x)
                flag: the stage of cat, seeking (looking/finding around) or tracing (chasing/catching)
        """
        x = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        v = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fitness = self._fitness_model__(x, minmax)
        flag = False        # False: seeking mode , True: tracing mode
        if random() < self.mixture_ratio:
            flag = True
        return [x, fitness, v, flag]


    def _get_index_roulette_wheel_selection__(self, list_fitness, sum_fitness, fitness_min):
        r = uniform(fitness_min, sum_fitness)
        for idx, f in enumerate(list_fitness):
            r += f
            if r > sum_fitness:
                return idx

    def _seeking_mode__(self, cat):
        candidate_cats = []
        clone_cats = [deepcopy(cat) for _ in range(self.smp)]
        if self.spc:
            candidate_cats.append(deepcopy(cat))
            clone_cats = [deepcopy(cat) for _ in range(self.smp - 1)]

        for clone in clone_cats:
            idx = sample(range(0, self.problem_size), int(self.cdc * self.problem_size))
            for u in idx:
                if uniform() < 0.5:
                    # temp = clone[self.ID_POS][u] * (1 + srd)
                    # if temp > 1.0:
                    #     clone[self.ID_POS][u] = 1.0
                    clone[self.ID_POS][u] += clone[self.ID_POS][u] * self.srd
                else:
                    # temp = clone[self.ID_POS][u] * (1 - srd)
                    # if temp < -1.0:
                    #     clone[self.ID_POS][u] = -1.0
                    clone[self.ID_POS][u] -= clone[self.ID_POS][u] * self.srd
            clone[self.ID_FIT] = self._fitness_model__(clone[self.ID_POS])
            candidate_cats.append(clone)

        fit1 = candidate_cats[0][self.ID_FIT]
        flag_equal = True
        for candidate in candidate_cats:
            if candidate[self.ID_FIT]!= fit1:
                flag_equal = False
                break

        if flag_equal == True:
            cat = choice(candidate_cats)            # Random choice one cat from list cats
        else:
            if self.selected_strategy == 0:                # Best fitness-self
                cat = sorted(candidate_cats, key=lambda cat: cat[self.ID_FIT])[0]

            elif self.selected_strategy == 1:              # Tournament
                k_way = 4
                idx = sample(range(0, self.smp), k_way)
                cats_k_way = [candidate_cats[_] for _ in idx]
                cat = sorted(cats_k_way, key=lambda cat: cat[self.ID_FIT])[0]

            elif self.selected_strategy == 2:              ### Roul-wheel selection
                fitness_list = [candidate_cats[u][self.ID_FIT] for u in range(0, len(candidate_cats))]
                fitness_sum = sum(fitness_list)
                fitness_min = min(fitness_list)
                idx = self._get_index_roulette_wheel_selection__(fitness_list, fitness_sum, fitness_min)
                cat = candidate_cats[idx]

            elif self.selected_strategy == 3:
                cat = choice(candidate_cats)                # Random
            else:
                print("Out of my abilities")
        return cat


    def _tracing_mode__(self, cat, cat_best, w):
        temp = cat[self.ID_POS] + w * cat[self.ID_VEL] + uniform() * self.c1 * (cat_best[self.ID_POS] - cat[self.ID_POS])
        temp = np.clip(temp, self.domain_range[0], self.domain_range[1])
        cat[self.ID_POS] = temp
        return cat

    def _train__(self):
        pop = [self._create_solution__() for _ in range(0, self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
            for i in range(0, self.pop_size):
                if pop[i][self.ID_FLAG]:
                    pop[i] = self._tracing_mode__(pop[i], g_best, w)
                else:
                    pop[i] = self._seeking_mode__(pop[i])
            for i in range(0, self.pop_size):
                if uniform() < self.mixture_ratio:
                    pop[i][self.ID_FLAG] = True
                else:
                    pop[i][self.ID_FLAG] = False

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
