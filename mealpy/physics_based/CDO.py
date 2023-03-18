#!/usr/bin/env python
# Created by "Thieu" at 21:45, 13/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalCDO(Optimizer):
    """
    The original version of: Chernobyl Disaster Optimizer (CDO)

    Links:
        1. https://link.springer.com/article/10.1007/s00521-023-08261-1
        2. https://www.mathworks.com/matlabcentral/fileexchange/124351-chernobyl-disaster-optimizer-cdo

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.CDO import OriginalCDO
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
    >>> model = OriginalCDO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Shehadeh, H. A. (2023). Chernobyl disaster optimizer (CDO): a novel meta-heuristic method
    for global optimization. Neural Computing and Applications, 1-17.
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
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        _, (b1, b2, b3), _ = self.get_special_solutions(self.pop, best=3, worst=1)
        a = 3 - (epoch+1)*3/self.epoch
        a1 = np.log10((16000-1) * np.random.rand()+16000)
        a2 = np.log10((270000-1)*np.random.rand() + 270000)
        a3 = np.log10((300000-1)*np.random.rand() + 300000)
        pop_new = []
        for idx in range(0, self.pop_size):
            r1 = np.random.rand(self.problem.n_dims)
            r2 = np.random.rand(self.problem.n_dims)
            pa = np.pi * r1*r1 / (0.25 * a1) - a*np.random.rand(self.problem.n_dims)
            c1 = r2 * r2 * np.pi
            alpha = np.abs(c1*b1[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_a = 0.25 * (b1[self.ID_POS] - pa * alpha)

            r3 = np.random.rand(self.problem.n_dims)
            r4 = np.random.rand(self.problem.n_dims)
            pb = np.pi * r3 * r3 / (0.5 * a2) - a * np.random.rand(self.problem.n_dims)
            c2 = r4 * r4 * np.pi
            beta = np.abs(c2 * b2[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_b = 0.5 * (b2[self.ID_POS] - pb * beta)

            r5 = np.random.rand(self.problem.n_dims)
            r6 = np.random.rand(self.problem.n_dims)
            pc = np.pi * r5 * r5 / a3 - a * np.random.rand(self.problem.n_dims)
            c3 = r6 * r6 * np.pi
            gama = np.abs(c3 * b3[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_c = b3[self.ID_POS] - pc * gama

            pos_new = (pos_a + pos_b + pos_c) / 3
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = pop_new
