#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:09, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# -------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseCSO(Optimizer):
    """
    The original version of: Cat Swarm Optimization (CSO)

    Links:
        1. https://link.springer.com/chapter/10.1007/978-3-540-36668-3_94
        2. https://www.hindawi.com/journals/cin/2020/4854895/

    Hyper-parameters should fine tuned in approximate range to get faster convergence toward the global optimum:
        + mixture_ratio (float): joining seeking mode with tracing mode, default=0.15
        + smp (int): seeking memory pool, default=5 clones (larger is better but time-consuming)
        + spc (bool): self-position considering, default=False
        + cdc (float): counts of dimension to change (larger is more diversity but slow convergence), default=0.8
        + srd (float): seeking range of the selected dimension (smaller is better but slow convergence), default=0.15
        + c1 (float): same in PSO, default=0.4
        + w_minmax (list, tuple): same in PSO, default=(0.4, 0.9)
        + selected_strategy (int):  0: best fitness, 1: tournament, 2: roulette wheel, else: random (decrease by quality)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.CSO import BaseCSO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> mixture_ratio = 0.15
    >>> smp = 5
    >>> spc = False
    >>> cdc = 0.8
    >>> srd = 0.15
    >>> c1 = 0.4
    >>> w_minmax = [0.4, 0.9]
    >>> selected_strategy = 1
    >>> model = BaseCSO(problem_dict1, epoch, pop_size, mixture_ratio, smp, spc, cdc, srd, c1, w_minmax, selected_strategy)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Chu, S.C., Tsai, P.W. and Pan, J.S., 2006, August. Cat swarm optimization. In Pacific Rim
    international conference on artificial intelligence (pp. 854-858). Springer, Berlin, Heidelberg.
    """

    ID_POS = 0  # position of the cat
    ID_TAR = 1  # fitness
    ID_VEL = 2  # velocity
    ID_FLAG = 3  # status

    def __init__(self, problem, epoch=10000, pop_size=100, mixture_ratio=0.15, smp=5, spc=False,
                 cdc=0.8, srd=0.15, c1=0.4, w_minmax=(0.4, 0.9), selected_strategy=1, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            mixture_ratio (float): joining seeking mode with tracing mode
            smp (int): seeking memory pool, 10 clones  (larger is better but time-consuming)
            spc (bool): self-position considering
            cdc (float): counts of dimension to change  (larger is more diversity but slow convergence)
            srd (float): seeking range of the selected dimension (smaller is better but slow convergence)
            c1 (float): same in PSO
            w_minmax (list, tuple): same in PSO
            selected_strategy (int):  0: best fitness, 1: tournament, 2: roulette wheel, else: random (decrease by quality)
        """

        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.mixture_ratio = self.validator.check_float("mixture_ratio", mixture_ratio, (0, 1.0))
        self.smp = self.validator.check_int("smp", smp, [2, 20])
        self.spc = self.validator.check_bool("spc", spc, [True, False])
        self.cdc = self.validator.check_float("cdc", cdc, (0, 1.0))
        self.srd = self.validator.check_float("srd", srd, (0, 1.0))
        self.c1 = self.validator.check_float("c1", c1, (0, 3.0))
        self.w_minmax = self.validator.check_tuple_float("w (min, max)", w_minmax, ([0.1, 0.49], [0.5, 2.0]))
        self.w_min, self.w_max = self.w_minmax
        self.selected_strategy = self.validator.check_int("selected_strategy", selected_strategy, [0, 4])
        self.sort_flag = False

    def create_solution(self, lb=None, ub=None):
        """
        + x: current position of cat
        + v: vector v of cat (same amount of dimension as x)
        + flag: the stage of cat, seeking (looking/finding around) or tracing (chasing/catching) => False: seeking mode , True: tracing mode

        To get the position, fitness wrapper, target and obj list
            + A[self.ID_POS]                  --> Return: position
            + A[self.ID_TAR]                  --> Return: [target, [obj1, obj2, ...]]
            + A[self.ID_TAR][self.ID_FIT]     --> Return: target
            + A[self.ID_TAR][self.ID_OBJ]     --> Return: [obj1, obj2, ...]

        Returns:
            list: wrapper of solution with format [position, target, velocity, flag]
        """
        position = self.generate_position(lb, ub)
        position = self.amend_position(position, lb, ub)
        target = self.get_target_wrapper(position)
        velocity = np.random.uniform(lb, ub)
        flag = True if np.random.uniform() < self.mixture_ratio else False
        return [position, target, velocity, flag]

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
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            candidate_cats.append([pos_new, None, clone[self.ID_VEL], clone[self.ID_FLAG]])
        candidate_cats = self.update_target_wrapper_population(candidate_cats)

        if self.selected_strategy == 0:  # Best fitness-self
            _, cat = self.get_global_best_solution(candidate_cats)
        elif self.selected_strategy == 1:  # Tournament
            k_way = 4
            idx = np.random.choice(range(0, self.smp), k_way, replace=False)
            cats_k_way = [candidate_cats[_] for _ in idx]
            _, cat = self.get_global_best_solution(cats_k_way)
        elif self.selected_strategy == 2:  ### Roul-wheel selection
            list_fitness = [candidate_cats[u][self.ID_TAR][self.ID_FIT] for u in range(0, len(candidate_cats))]
            idx = self.get_index_roulette_wheel_selection(list_fitness)
            cat = candidate_cats[idx]
        else:
            idx = np.random.choice(range(0, len(candidate_cats)))
            cat = candidate_cats[idx]  # Random
        return cat[self.ID_POS]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

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
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            else:
                pos_new = self._seeking_mode__(self.pop[idx])
            agent[self.ID_POS] = pos_new
            agent[self.ID_FLAG] = True if np.random.uniform() < self.mixture_ratio else False
            pop_new.append(agent)
        self.pop = self.update_target_wrapper_population(pop_new)
        self.nfe_per_epoch = self.pop_size
