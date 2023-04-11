#!/usr/bin/env python
# Created by "Thieu" at 18:09, 13/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalRIME(Optimizer):
    """
    The original version of: physical phenomenon of RIME-ice  (RIME)

    Links:
        1. https://doi.org/10.1016/j.neucom.2023.02.010
        2. https://www.mathworks.com/matlabcentral/fileexchange/124610-rime-a-physics-based-optimization

    Notes (parameters):
        1. w (float): Soft-rime parameters, default=5.0
        2. The algorithm is straightforward and does not require any specialized knowledge or techniques.
        3. The algorithm may exhibit slow convergence and may not perform optimally.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.RIME import OriginalRIME
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
    >>> w = 5.0
    >>> model = OriginalRIME(epoch, pop_size, w)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Su, H., Zhao, D., Heidari, A. A., Liu, L., Zhang, X., Mafarja, M., & Chen, H. (2023). RIME: A physics-based optimization. Neurocomputing.
    """
    def __init__(self, epoch=10000, pop_size=100, w=5., **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            w (float): Soft-rime parameters, default=5.0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.w = self.validator.check_float("w", w, (0., 100.))
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        rime_factor = (np.random.rand() - 0.5)*2*np.cos(np.pi*(epoch+1)/(self.epoch/10)) * (1 - np.round((epoch+1)*self.w/self.epoch) / self.w)
        ee = np.sqrt((epoch+1)/self.epoch)
        fits = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in self.pop]).reshape((1, -1))
        fits_norm = fits / np.linalg.norm(fits, axis=1, keepdims=True)
        LB = self.problem.lb
        UB = self.problem.ub
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx][self.ID_POS].copy()
            for jdx in range(0, self.problem.n_dims):
                # Soft-rime search strategy
                if np.random.rand() < ee:
                    pos_new[jdx] = self.g_best[self.ID_POS][jdx] + rime_factor*(LB[jdx] + np.random.rand() * (UB[jdx] - LB[jdx]))
                # Hard-rime puncture mechanism
                if np.random.rand() < fits_norm[0, idx]:
                    pos_new[jdx] = self.g_best[self.ID_POS][jdx]
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
