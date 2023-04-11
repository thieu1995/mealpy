#!/usr/bin/env python
# Created by "Thieu" at 18:09, 13/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalEVO(Optimizer):
    """
    The original version of: Energy Valley Optimizer (EVO)

    Links:
        1. https://www.nature.com/articles/s41598-022-27344-y
        2. https://www.mathworks.com/matlabcentral/fileexchange/123130-energy-valley-optimizer-a-novel-metaheuristic-algorithm

    Notes:
        1. The algorithm is straightforward and does not require any specialized knowledge or techniques.
        2. The algorithm may not perform optimally due to slow convergence and no good operations, which could be improved by implementing better strategies and operations.
        3. The problem is that it is stuck at a local optimal around 1/2 of the max generations because fitness distance is being used as a factor in the equations.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.EVO import OriginalEVO
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
    >>> model = OriginalEVO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Azizi, M., Aickelin, U., A. Khorshidi, H., & Baghalzadeh Shishehgarkhaneh, M. (2023). Energy valley optimizer: a novel
    metaheuristic algorithm for global and engineering optimization. Scientific Reports, 13(1), 226.
    """
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_list = np.array([agent[self.ID_POS] for agent in self.pop])
            fit_list = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in self.pop])
            dis = np.sqrt(np.sum((self.pop[idx][self.ID_POS] - pos_list)**2, axis=1))
            idx_dis_sort = np.argsort(dis)
            CnPtIdx = np.random.choice(list(set(range(2, self.pop_size)) - {idx}))
            x_team = pos_list[idx_dis_sort[1:CnPtIdx], :]
            x_avg_team = np.mean(x_team, axis=0)
            x_avg_pop = np.mean(pos_list, axis=0)
            eb = np.mean(fit_list)
            sl = (fit_list[idx] - self.g_best[self.ID_TAR][self.ID_FIT]) / (self.g_worst[self.ID_TAR][self.ID_FIT] - self.g_best[self.ID_TAR][self.ID_FIT] + self.EPSILON)

            pos_new1 = self.pop[idx][self.ID_POS].copy()
            pos_new2 = self.pop[idx][self.ID_POS].copy()
            if self.compare_agent([None, [eb]], self.pop[idx]):
                if np.random.rand() > sl:
                    a1_idx = np.random.randint(self.problem.n_dims)
                    a2_idx = np.random.randint(0, self.problem.n_dims, size=a1_idx)
                    pos_new1[a2_idx] = self.g_best[self.ID_POS][a2_idx]
                    g1_idx = np.random.randint(self.problem.n_dims)
                    g2_idx = np.random.randint(0, self.problem.n_dims, size=g1_idx)
                    pos_new2[g2_idx] = x_avg_team[g2_idx]
                else:
                    ir = np.random.uniform(0, 1, 2)
                    jr = np.random.uniform(0, 1, self.problem.n_dims)
                    pos_new1 += jr * (ir[0] * self.g_best[self.ID_POS] - ir[1] * x_avg_pop) / sl
                    ir = np.random.uniform(0, 1, 2)
                    jr = np.random.uniform(0, 1, self.problem.n_dims)
                    pos_new2 += jr * (ir[0] * self.g_best[self.ID_POS] - ir[1] * x_avg_team)
                pos_new1 = self.amend_position(pos_new1, self.problem.lb, self.problem.ub)
                pos_new2 = self.amend_position(pos_new2, self.problem.lb, self.problem.ub)
                pop_new.append([pos_new1, None])
                pop_new.append([pos_new2, None])
            else:
                pos_new = pos_new1 + np.random.randn() * sl * np.random.uniform(self.problem.lb, self.problem.ub, self.problem.n_dims)
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                pop_new.append([pos_new, None])
        if self.mode not in self.AVAILABLE_MODES:
            for idx in range(0, len(pop_new)):
                pop_new[idx][self.ID_TAR] = self.get_target_wrapper(pop_new[idx][self.ID_POS])
        pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = self.get_sorted_strim_population(self.pop + pop_new, self.pop_size)
