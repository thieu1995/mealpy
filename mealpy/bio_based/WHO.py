#!/usr/bin/env python
# Created by "Thieu" at 12:51, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalWHO(Optimizer):
    """
    The original version of: Wildebeest Herd Optimization (WHO)

    Links:
        1. https://doi.org/10.3233/JIFS-190495

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_explore_step (int): [2, 10] -> better [2, 4], number of exploration step
        + n_exploit_step (int): [2, 10] -> better [2, 4], number of exploitation step
        + eta (float): (0, 1.0) -> better [0.05, 0.5], learning rate
        + p_hi (float): (0, 1.0) -> better [0.7, 0.95], the probability of wildebeest move to another position based on herd instinct
        + local_alpha (float): (0, 3.0) -> better [0.5, 0.9], control local movement (alpha 1)
        + local_beta (float): (0, 3.0) -> better [0.1, 0.5], control local movement (beta 1)
        + global_alpha (float): (0, 3.0) -> better [0.1, 0.5], control global movement (alpha 2)
        + global_beta (float): (0, 3.0), control global movement (beta 2)
        + delta_w (float): (0.5, 5.0) -> better [1.0, 2.0], dist to worst
        + delta_c (float): (0.5, 5.0) -> better [1.0, 2.0], dist to best

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.WHO import OriginalWHO
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
    >>> n_explore_step = 3
    >>> n_exploit_step = 3
    >>> eta = 0.15
    >>> p_hi = 0.9
    >>> local_alpha=0.9
    >>> local_beta=0.3
    >>> global_alpha=0.2
    >>> global_beta=0.8
    >>> delta_w=2.0
    >>> delta_c=2.0
    >>> model = OriginalWHO(epoch, pop_size, n_explore_step, n_exploit_step, eta, p_hi, local_alpha, local_beta,
    >>>                     global_alpha, global_beta, delta_w, delta_c)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Amali, D. and Dinakaran, M., 2019. Wildebeest herd optimization: a new global optimization algorithm inspired
    by wildebeest herding behaviour. Journal of Intelligent & Fuzzy Systems, 37(6), pp.8063-8076.
    """

    def __init__(self, epoch=10000, pop_size=100, n_explore_step=3, n_exploit_step=3, eta=0.15, p_hi=0.9,
                 local_alpha=0.9, local_beta=0.3, global_alpha=0.2, global_beta=0.8, delta_w=2.0, delta_c=2.0, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_explore_step (int): default = 3, number of exploration step
            n_exploit_step (int): default = 3, number of exploitation step
            eta (float): default = 0.15, learning rate
            p_hi (float): default = 0.9, the probability of wildebeest move to another position based on herd instinct
            local_alpha (float): control local movement (alpha 1)
            local_beta (float): control local movement (beta 1)
            global_alpha (float): control global movement (alpha 2)
            global_beta (float): control global movement (beta 2)
            delta_w (float): dist to worst
            delta_c (float): dist to best
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_explore_step = self.validator.check_int("n_explore_step", n_explore_step, [2, 10])
        self.n_exploit_step = self.validator.check_int("n_exploit_step", n_exploit_step, [2, 10])
        self.eta = self.validator.check_float("eta", eta, (0, 1.0))
        self.p_hi = self.validator.check_float("p_hi", p_hi, (0, 1.0))
        self.local_alpha = self.validator.check_float("local_alpha", local_alpha, (0, 3.0))
        self.local_beta = self.validator.check_float("local_beta", local_beta, (0, 3.0))
        self.global_alpha = self.validator.check_float("global_alpha", global_alpha, (0, 3.0))
        self.global_beta = self.validator.check_float("global_beta", global_beta, (0, 3.0))
        self.delta_w = self.validator.check_float("delta_w", delta_w, (0.5, 5.0))
        self.delta_c = self.validator.check_float("delta_c", delta_c, (0.5, 5.0))
        self.set_parameters(["epoch", "pop_size", "n_explore_step", "n_exploit_step",
                             "eta", "p_hi", "local_alpha", "local_beta", "global_alpha", "global_beta", "delta_w", "delta_c"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Begin the Wildebeest Herd Optimization process
        pop_new = []
        for idx in range(0, self.pop_size):
            ### 1. Local movement (Milling behaviour)
            local_list = []
            for j in range(0, self.n_explore_step):
                temp = self.pop[idx][self.ID_POS] + self.eta * np.random.uniform() * np.random.uniform(self.problem.lb, self.problem.ub)
                pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
                local_list.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    local_list[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
            local_list = self.update_target_wrapper_population(local_list)
            _, best_local = self.get_global_best_solution(local_list)
            temp = self.local_alpha * best_local[self.ID_POS] + self.local_beta * (self.pop[idx][self.ID_POS] - best_local[self.ID_POS])
            pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        for idx in range(0, self.pop_size):
            ### 2. Herd instinct
            idr = np.random.choice(range(0, self.pop_size))
            if self.compare_agent(self.pop[idr], self.pop[idx]) and np.random.rand() < self.p_hi:
                temp = self.global_alpha * self.pop[idx][self.ID_POS] + self.global_beta * self.pop[idr][self.ID_POS]
                pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
                target = self.get_target_wrapper(pos_new)
                if self.compare_agent([pos_new, target], self.pop[idx]):
                    self.pop[idx] = [pos_new, target]

        _, best, worst = self.get_special_solutions(self.pop, best=1, worst=1)
        g_best, g_worst = best[0], worst[0]

        pop_child = []
        for idx in range(0, self.pop_size):
            dist_to_worst = np.linalg.norm(self.pop[idx][self.ID_POS] - g_worst[self.ID_POS])
            dist_to_best = np.linalg.norm(self.pop[idx][self.ID_POS] - g_best[self.ID_POS])

            ### 3. Starvation avoidance
            if dist_to_worst < self.delta_w:
                temp = self.pop[idx][self.ID_POS] + np.random.uniform() * (self.problem.ub - self.problem.lb) * \
                       np.random.uniform(self.problem.lb, self.problem.ub)
                pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
                pop_child.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    target = self.get_target_wrapper(pos_new)
                    self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])

            ### 4. Population pressure
            if 1.0 < dist_to_best and dist_to_best < self.delta_c:
                temp = g_best[self.ID_POS] + self.eta * np.random.uniform(self.problem.lb, self.problem.ub)
                pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
                pop_child.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    target = self.get_target_wrapper(pos_new)
                    self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])

            ### 5. Herd social memory
            for j in range(0, self.n_exploit_step):
                temp = g_best[self.ID_POS] + 0.1 * np.random.uniform(self.problem.lb, self.problem.ub)
                pos_new = self.amend_position(temp, self.problem.lb, self.problem.ub)
                pop_child.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    target = self.get_target_wrapper(pos_new)
                    self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            pop_child = self.get_sorted_strim_population(pop_child, self.pop_size)
            self.pop = self.greedy_selection_population(pop_child, self.pop)
