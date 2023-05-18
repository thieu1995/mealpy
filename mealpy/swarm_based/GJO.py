#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalGJO(Optimizer):
    """
    The original version of: Golden jackal optimization (GJO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S095741742200358X
        2. https://www.mathworks.com/matlabcentral/fileexchange/108889-golden-jackal-optimization-algorithm

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.GJO import OriginalGJO
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
    >>> model = OriginalGJO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Chopra, N., & Ansari, M. M. (2022). Golden jackal optimization: A novel nature-inspired
    optimizer for engineering applications. Expert Systems with Applications, 198, 116924.
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
        E1 = 1.5*(1-((1+epoch)/self.epoch))
        RL = self.get_levy_flight_step(beta=1.5, multiplier=0.05, size=(self.pop_size, self.problem.n_dims), case=-1)
        _, (male, female), _ = self.get_special_solutions(self.pop, best=2, worst=1)
        pop_new = []
        for idx in range(0, self.pop_size):
            male_pos = male[self.ID_POS].copy()
            female_pos = female[self.ID_POS].copy()

            for jdx in range(0, self.problem.n_dims):
                r1 = np.random.rand()
                E0 = 2*r1 - 1
                E = E1 * E0
                if np.abs(E) < 1:       # EXPLOITATION
                    t1 = np.abs( (RL[idx, jdx] * male[self.ID_POS][jdx] - self.pop[idx][self.ID_POS][jdx]) )
                    male_pos[jdx] = male[self.ID_POS][jdx] - E*t1
                    t2 = np.abs( (RL[idx, jdx] * female[self.ID_POS][jdx] - self.pop[idx][self.ID_POS][jdx]) )
                    female_pos[jdx] = female[self.ID_POS][jdx] - E*t2
                else:                   # EXPLORATION
                    t1 = np.abs((male[self.ID_POS][jdx] - RL[idx, jdx] * self.pop[idx][self.ID_POS][jdx]))
                    male_pos[jdx] = male[self.ID_POS][jdx] - E * t1
                    t2 = np.abs((female[self.ID_POS][jdx] - RL[idx, jdx] * self.pop[idx][self.ID_POS][jdx]))
                    female_pos[jdx] = female[self.ID_POS][jdx] - E * t2
            pos_new = (male_pos + female_pos) / 2
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = [pos_new, target]
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_wrapper_population(pop_new)
