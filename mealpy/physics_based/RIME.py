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
        1. sr (float): Soft-rime parameters, default=5.0
        2. The algorithm is straightforward and does not require any specialized knowledge or techniques.
        3. The algorithm may exhibit slow convergence and may not perform optimally.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, RIME
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = RIME.OriginalRIME(epoch=1000, pop_size=50, sr = 5.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Su, H., Zhao, D., Heidari, A. A., Liu, L., Zhang, X., Mafarja, M., & Chen, H. (2023). RIME: A physics-based optimization. Neurocomputing.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, sr: float = 5., **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            sr (float): Soft-rime parameters, default=5.0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.sr = self.validator.check_float("sr", sr, (0., 100.))
        self.set_parameters(["epoch", "pop_size", "sr"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        rime_factor = (self.generator.random() - 0.5)*2*np.cos(np.pi*epoch/(self.epoch/10)) * (1 - np.round(epoch*self.sr/self.epoch) / self.sr)
        ee = np.sqrt((epoch+1)/self.epoch)
        fits = np.array([agent.target.fitness for agent in self.pop]).reshape((1, -1))
        fits_norm = fits / np.linalg.norm(fits, axis=1, keepdims=True)
        LB = self.problem.lb
        UB = self.problem.ub
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx].solution.copy()
            for jdx in range(0, self.problem.n_dims):
                # Soft-rime search strategy
                if self.generator.random() < ee:
                    pos_new[jdx] = self.g_best.solution[jdx] + rime_factor*(LB[jdx] + self.generator.random() * (UB[jdx] - LB[jdx]))
                # Hard-rime puncture mechanism
                if self.generator.random() < fits_norm[0, idx]:
                    pos_new[jdx] = self.g_best.solution[jdx]
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
