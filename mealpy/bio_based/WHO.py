# !/usr/bin/env python
# Created by "Thieu" at 12:51, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseWHO(Optimizer):
    """
    The original version of: Wildebeest Herd Optimization (WHO)

    Links:
        1. https://doi.org/10.3233/JIFS-190495

    Notes
    ~~~~~
    Before updated old position, I check whether new position is better or not.

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + n_s (int): [2, 4], number of exploration step
        + n_e (int): [2, 4], number of exploitation step
        + eta (float): [0.05, 0.5], learning rate
        + p_hi (float): [0.7, 0.95], the probability of wildebeest move to another position based on herd instinct
        + local_move (tuple, list): (alpha 1, beta 1) -> ([0.5, 0.9], [0.1, 0.5]), control local movement
        + global_move (tuple, list): (alpha 2, beta 2) -> ([0.1, 0.5], [0.5, 0.9]), control global movement
        + delta (tuple, list): (delta_w, delta_c) -> ([1.0, 2.0], [1.0, 2.0]), (dist to worst, dist to best)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.WHO import BaseWHO
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
    >>> n_s = 3
    >>> n_e = 3
    >>> eta = 0.15
    >>> p_hi = 0.9
    >>> local_move = (0.9, 0.3)
    >>> global_move = (0.2, 0.8)
    >>> delta = (2.0, 2.0)
    >>> model = BaseWHO(problem_dict1, epoch, pop_size, n_s, n_e, eta, p_hi, local_move, global_move, delta,)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Amali, D. and Dinakaran, M., 2019. Wildebeest herd optimization: a new global optimization algorithm inspired
    by wildebeest herding behaviour. Journal of Intelligent & Fuzzy Systems, 37(6), pp.8063-8076.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, n_s=3, n_e=3, eta=0.15, p_hi=0.9, local_move=(0.9, 0.3),
                 global_move=(0.2, 0.8), delta=(2.0, 2.0), **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_s (int): default = 3, number of exploration step
            n_e (int): default = 3, number of exploitation step
            eta (float): default = 0.15, learning rate
            p_hi (float): default = 0.9, the probability of wildebeest move to another position based on herd instinct
            local_move (tuple, list): default = (0.9, 0.3), (alpha 1, beta 1) - control local movement
            global_move (tuple, list): default = (0.2, 0.8), (alpha 2, beta 2) - control global movement
            delta (tuple, list): default = (2.0, 2.0) , (delta_w, delta_c) - (dist to worst, dist to best)
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_s = self.validator.check_int("n_s", n_s, [2, 10])
        self.n_e = self.validator.check_int("n_e", n_e, [2, 10])
        self.eta = self.validator.check_float("eta", eta, (0, 1.0))
        self.p_hi = self.validator.check_float("p_hi", p_hi, (0, 1.0))
        self.local_move = self.validator.check_tuple_float("local_move (alpha 1, beta 1)", local_move, ((0, 2.0), (0, 2.0)))
        self.global_move = self.validator.check_tuple_float("global_move (alpha 2, beta 2)", global_move, ((0, 2.0), (0, 2.0)))
        self.delta = self.validator.check_tuple_float("delta (delta_w, delta_c)", delta, ((0.5, 5.0), (0.5, 5.0)))

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        ## Begin the Wildebeest Herd Optimization process
        pop_new = []
        for i in range(0, self.pop_size):
            ### 1. Local movement (Milling behaviour)
            local_list = []
            for j in range(0, self.n_s):
                temp = self.pop[i][self.ID_POS] + self.eta * np.random.uniform() * np.random.uniform(self.problem.lb, self.problem.ub)
                pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
                local_list.append([pos_new, None])
            local_list = self.update_target_wrapper_population(local_list)
            _, best_local = self.get_global_best_solution(local_list)
            temp = self.local_move[0] * best_local[self.ID_POS] + self.local_move[1] * (self.pop[i][self.ID_POS] - best_local[self.ID_POS])
            pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
        pop_new = self.update_target_wrapper_population(pop_new)
        pop_new = self.greedy_selection_population(self.pop, pop_new)
        nfe_epoch += self.pop_size

        for i in range(0, self.pop_size):
            ### 2. Herd instinct
            idr = np.random.choice(range(0, self.pop_size))
            if self.compare_agent(pop_new[idr], pop_new[i]) and np.random.rand() < self.p_hi:
                temp = self.global_move[0] * pop_new[i][self.ID_POS] + self.global_move[1] * pop_new[idr][self.ID_POS]
                pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
                target = self.get_target_wrapper(pos_new)
                nfe_epoch += 1
                if self.compare_agent([pos_new, target], pop_new[i]):
                    pop_new[i] = [pos_new, target]

        _, best, worst = self.get_special_solutions(pop_new, best=1, worst=1)
        g_best, g_worst = best[0], worst[0]

        pop_child = []
        for i in range(0, self.pop_size):
            dist_to_worst = np.linalg.norm(pop_new[i][self.ID_POS] - g_worst[self.ID_POS])
            dist_to_best = np.linalg.norm(pop_new[i][self.ID_POS] - g_best[self.ID_POS])

            ### 3. Starvation avoidance
            if dist_to_worst < self.delta[0]:
                temp = pop_new[i][self.ID_POS] + np.random.uniform() * (self.problem.ub - self.problem.lb) * \
                       np.random.uniform(self.problem.lb, self.problem.ub)
                pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
                pop_child.append([pos_new, None])

            ### 4. Population pressure
            if 1.0 < dist_to_best and dist_to_best < self.delta[1]:
                temp = g_best[self.ID_POS] + self.eta * np.random.uniform(self.problem.lb, self.problem.ub)
                pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
                pop_child.append([pos_new, None])

            ### 5. Herd social memory
            for j in range(0, self.n_e):
                temp = g_best[self.ID_POS] + 0.1 * np.random.uniform(self.problem.lb, self.problem.ub)
                pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
                pop_child.append([pos_new, None])

        nfe_epoch += len(pop_child)
        self.nfe_per_epoch = nfe_epoch
        pop_child = self.update_target_wrapper_population(pop_child)
        pop_child = self.get_sorted_strim_population(pop_child, self.pop_size)
        self.pop = self.greedy_selection_population(pop_new, pop_child)
