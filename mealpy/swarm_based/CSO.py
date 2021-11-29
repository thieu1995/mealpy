#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:09, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseCSO(Optimizer):
    """
        The original version of: Cat Swarm Optimization (CSO)
        Link:
            https://link.springer.com/chapter/10.1007/978-3-540-36668-3_94
            https://www.hindawi.com/journals/cin/2020/4854895/
    """
    ID_POS = 0      # position of the cat
    ID_FIT = 1      # fitness
    ID_VEL = 2      # velocity
    ID_FLAG = 3     # status

    def __init__(self, problem, epoch=10000, pop_size=100, mixture_ratio=0.15, smp=5, spc=False,
                 cdc=0.8, srd=0.15, c1=0.4, w_minmax=(0.4, 0.9), selected_strategy=1, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            mixture_ratio (): joining seeking mode with tracing mode
            smp (): seeking memory pool, 10 clones  (larger is better but time-consuming)
            spc (): self-position considering
            cdc (): counts of dimension to change  (larger is more diversity but slow convergence)
            srd (): seeking range of the selected dimension (smaller is better but slow convergence)
            c1 (): same in PSO
            w_minmax (): same in PSO
            selected_strategy ():  0: best fitness, 1: tournament, 2: roulette wheel, else: random  (decrease by quality)
            **kwargs ():
        """

        super().__init__(problem, kwargs)
        self.epoch = epoch
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

    def create_solution(self):
        """
        Returns:
            The position position with 2 element: index of position/location and index of fitness wrapper
            The general format: [position, [target, [obj1, obj2, ...]]]

        ## To get the position, fitness wrapper, target and obj list
        ##      A[self.ID_POS]                  --> Return: position
        ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
        ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
        ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]

        x: current position of cat
        v: vector v of cat (same amount of dimension as x)
        flag: the stage of cat, seeking (looking/finding around) or tracing (chasing/catching)
        # False: seeking mode , True: tracing mode
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position=position)
        velocity = np.random.uniform(self.problem.lb, self.problem.ub)
        flag = True if np.random.uniform() < self.mixture_ratio else False
        return [position, fitness, velocity, flag]

    def _seeking_mode__(self, cat):
        candidate_cats = []
        clone_cats = self.create_population(self.smp)
        if self.spc:
            candidate_cats.append(deepcopy(cat))
            clone_cats = [deepcopy(cat) for _ in range(self.smp - 1)]

        for clone in clone_cats:
            idx = np.random.choice(range(0, self.problem.n_dims), int(self.cdc * self.problem.n_dims), replace=False)
            pos_new1 = clone[self.ID_POS] * (1 + self.srd)
            pos_new2 = clone[self.ID_POS] * (1 - self.srd)

            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < 0.5, pos_new1, pos_new2)
            pos_new[idx] = clone[self.ID_POS][idx]
            pos_new = self.amend_position_faster(pos_new)
            candidate_cats.append([pos_new, None, clone[self.ID_VEL], clone[self.ID_FLAG]])
        candidate_cats = self.update_fitness_population(candidate_cats)

        if self.selected_strategy == 0:                # Best fitness-self
            _, cat = self.get_global_best_solution(candidate_cats)
        elif self.selected_strategy == 1:              # Tournament
            k_way = 4
            idx = np.random.choice(range(0, self.smp), k_way, replace=False)
            cats_k_way = [candidate_cats[_] for _ in idx]
            _, cat = self.get_global_best_solution(cats_k_way)
        elif self.selected_strategy == 2:              ### Roul-wheel selection
            list_fitness = [candidate_cats[u][self.ID_FIT][self.ID_TAR] for u in range(0, len(candidate_cats))]
            idx = self.get_index_roulette_wheel_selection(list_fitness)
            cat = candidate_cats[idx]
        else:
            idx = np.random.choice(range(0, len(candidate_cats)))
            cat = candidate_cats[idx]               # Random
        return cat[self.ID_POS]

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = deepcopy(self.pop[idx])
            # tracing mode
            if self.pop[idx][self.ID_FLAG]:
                pos_new = self.pop[idx][self.ID_POS] + w * self.pop[idx][self.ID_VEL] + \
                          np.random.uniform() * self.c1 * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                pos_new = self.amend_position_faster(pos_new)
            else:
                pos_new = self._seeking_mode__(self.pop[idx])
            agent[self.ID_POS] = pos_new
            agent[self.ID_FLAG] = True if np.random.uniform() < self.mixture_ratio else False
            pop_new.append(agent)
        self.pop = self.update_fitness_population(pop_new)
