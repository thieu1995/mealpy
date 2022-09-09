# !/usr/bin/env python
# Created by "Thieu" at 15:37, 19/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalHGS(Optimizer):
    """
    The original version of: Hunger Games Search (HGS)

    Links:
        https://aliasgharheidari.com/HGS.html

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + PUP (float): [0.01, 0.2], The probability of updating position (L in the paper), default = 0.08
        + LH (float): [1000, 20000], Largest hunger / threshold, default = 10000

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.HGS import OriginalHGS
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
    >>> PUP = 0.08
    >>> LH = 10000
    >>> model = OriginalHGS(epoch, pop_size, PUP, LH)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, Y., Chen, H., Heidari, A.A. and Gandomi, A.H., 2021. Hunger games search: Visions, conception, implementation,
    deep analysis, perspectives, and towards performance shifts. Expert Systems with Applications, 177, p.114864.
    """

    ID_HUN = 2  # ID for Hunger value

    def __init__(self, epoch=10000, pop_size=100, PUP=0.08, LH=10000, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            PUP (float): The probability of updating position (L in the paper), default = 0.08
            LH (float): Largest hunger / threshold, default = 10000
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.PUP = self.validator.check_float("PUP", PUP, (0, 1.0))
        self.LH = self.validator.check_float("LH", LH, [1000, 20000])
        self.set_parameters(["epoch", "pop_size", "PUP", "LH"])
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, target, hunger]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        hunger = 1.0
        return [position, target, hunger]

    def sech__(self, x):
        if np.abs(x) > 50:
            return 0.5
        return 2 / (np.exp(x) + np.exp(-x))

    def update_hunger_value__(self, pop=None, g_best=None, g_worst=None):
        # min_index = pop.index(min(pop, key=lambda x: x[self.ID_TAR][self.ID_FIT]))
        # Eq (2.8) and (2.9)
        for i in range(0, self.pop_size):
            r = np.random.rand()
            # space: since we pass lower bound and upper bound as list. Better take the np.mean of them.
            space = np.mean(self.problem.ub - self.problem.lb)
            H = (pop[i][self.ID_TAR][self.ID_FIT] - g_best[self.ID_TAR][self.ID_FIT]) / \
                (g_worst[self.ID_TAR][self.ID_FIT] - g_best[self.ID_TAR][self.ID_FIT] + self.EPSILON) * r * 2 * space
            if H < self.LH:
                H = self.LH * (1 + r)
            pop[i][self.ID_HUN] += H

            if g_best[self.ID_TAR][self.ID_FIT] == pop[i][self.ID_TAR][self.ID_FIT]:
                pop[i][self.ID_HUN] = 0
        return pop

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Eq. (2.2)
        ### Find the current best and current worst
        _, best, worst = self.get_special_solutions(self.pop, best=1, worst=1)
        g_best, g_worst = best[0], worst[0]
        pop = self.update_hunger_value__(self.pop, g_best, g_worst)

        ## Eq. (2.4)
        shrink = 2 * (1 - (epoch + 1) / self.epoch)
        total_hunger = np.sum([pop[idx][self.ID_HUN] for idx in range(0, self.pop_size)])

        pop_new = []
        for idx in range(0, self.pop_size):
            agent = deepcopy(self.pop[idx])
            #### Variation control
            E = self.sech__(self.pop[idx][self.ID_TAR][self.ID_FIT] - g_best[self.ID_TAR][self.ID_FIT])

            # R is a ranging controller added to limit the range of activity, in which the range of R is gradually reduced to 0
            R = 2 * shrink * np.random.rand() - shrink  # Eq. (2.3)

            ## Calculate the hungry weight of each position
            if np.random.rand() < self.PUP:
                W1 = self.pop[idx][self.ID_HUN] * self.pop_size / (total_hunger + self.EPSILON) * np.random.rand()
            else:
                W1 = 1
            W2 = (1 - np.exp(-np.abs(self.pop[idx][self.ID_HUN] - total_hunger))) * np.random.rand() * 2

            ### Udpate position of individual Eq. (2.1)
            r1 = np.random.rand()
            r2 = np.random.rand()
            if r1 < self.PUP:
                pos_new = self.pop[idx][self.ID_POS] * (1 + np.random.normal(0, 1))
            else:
                if r2 > E:
                    pos_new = W1 * g_best[self.ID_POS] + R * W2 * np.abs(g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                else:
                    pos_new = W1 * g_best[self.ID_POS] - R * W2 * np.abs(g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            agent[self.ID_POS] = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent[self.ID_TAR] = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], agent)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
