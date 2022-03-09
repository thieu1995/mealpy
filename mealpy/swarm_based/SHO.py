# !/usr/bin/env python
# Created by "Thieu" at 10:55, 02/12/2019 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseSHO(Optimizer):
    """
    My changed version of: Spotted Hyena Optimizer (SHO)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2017.05.014

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + h_factor (float): default = 5, coefficient linearly decreased from 5 to 0
        + rand_v (list): (uniform min, uniform max), random vector, default = [0.5, 1]
        + N_tried (int): default = 10,

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.SHO import BaseSHO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> h_factor = 5
    >>> rand_v = [0.5, 1]
    >>> N_tried = 10
    >>> model = BaseSHO(problem_dict1, epoch, pop_size, h_factor, rand_v, N_tried)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Dhiman, G. and Kumar, V., 2017. Spotted hyena optimizer: a novel bio-inspired based metaheuristic
    technique for engineering applications. Advances in Engineering Software, 114, pp.48-70.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, h_factor=5, rand_v=(0.5, 1), N_tried=10, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            h_factor (float): default = 5, coefficient linearly decreased from 5 to 0
            rand_v (list): (uniform min, uniform max), random vector, default = [0.5, 1]
            N_tried (int): default = 10,
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.h_factor = h_factor
        self.rand_v = rand_v
        self.N_tried = N_tried

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        pop_new = []
        for idx in range(0, self.pop_size):
            h = self.h_factor - (epoch + 1.0) * (self.h_factor / self.epoch)
            rd1 = np.random.uniform(0, 1, self.problem.n_dims)
            rd2 = np.random.uniform(0, 1, self.problem.n_dims)
            B = 2 * rd1
            E = 2 * h * rd2 - h

            if np.random.rand() < 0.5:
                D_h = np.abs(np.dot(B, self.g_best[self.ID_POS]) - self.pop[idx][self.ID_POS])
                pos_new = self.g_best[self.ID_POS] - np.dot(E, D_h)
            else:
                N = 1
                for i in range(0, self.N_tried):
                    pos_temp = self.g_best[self.ID_POS] + np.random.uniform(self.rand_v[0], self.rand_v[1]) * \
                              np.random.uniform(self.problem.lb, self.problem.ub)
                    pos_temp = self.amend_position(pos_temp)
                    fit_new = self.get_fitness_position(pos_temp)
                    if self.compare_agent([pos_temp, fit_new], self.g_best):
                        N += 1
                        nfe_epoch += 1
                        break
                    N += 1
                circle_list = []
                idx_list = np.random.choice(range(0, self.pop_size), N, replace=False)
                for j in range(0, N):
                    D_h = np.abs(np.dot(B, self.g_best[self.ID_POS]) - self.pop[idx_list[j]][self.ID_POS])
                    p_k = self.g_best[self.ID_POS] - np.dot(E, D_h)
                    circle_list.append(p_k)
                pos_new = np.mean(np.array(circle_list), axis=0)
            pos_new = self.amend_position(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
        nfe_epoch += self.pop_size
        self.nfe_per_epoch = nfe_epoch
