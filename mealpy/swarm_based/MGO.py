#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalMGO(Optimizer):
    """
    The original version of: Mountain Gazelle Optimizer (MGO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0965997822001831
        2. https://www.mathworks.com/matlabcentral/fileexchange/118680-mountain-gazelle-optimizer

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, MGO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = MGO.OriginalMGO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdollahzadeh, B., Gharehchopogh, F. S., Khodadadi, N., & Mirjalili, S. (2022). Mountain gazelle optimizer: a new
    nature-inspired metaheuristic algorithm for global optimization problems. Advances in Engineering Software, 174, 103282.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def coefficient_vector__(self, n_dims, epoch, max_epoch):
        a2 = -1. + epoch * (-1. / max_epoch)
        u = self.generator.standard_normal(n_dims)
        v = self.generator.standard_normal(n_dims)
        cofi = np.zeros((4, n_dims))
        cofi[0, :] = self.generator.random(n_dims)
        cofi[1, :] = (a2 + 1) + self.generator.random()
        cofi[2, :] = a2 * self.generator.standard_normal(n_dims)
        cofi[3, :] = u * np.power(v, 2) * np.cos((self.generator.random() * 2) * u)
        return cofi

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            idxs_rand = self.generator.permutation(self.pop_size)[:int(np.ceil(self.pop_size/3))]
            pos_list = np.array([ self.pop[mm].solution for mm in idxs_rand ])
            idx_rand = self.generator.integers(int(np.ceil(self.pop_size / 3)), self.pop_size)
            M = self.pop[idx_rand].solution * np.floor(self.generator.normal()) + np.mean(pos_list, axis=0) * np.ceil(self.generator.normal())

            # Calculate the vector of coefficients
            cofi = self.coefficient_vector__(self.problem.n_dims, epoch+1, self.epoch)
            A = self.generator.standard_normal(self.problem.n_dims) * np.exp(2 - (epoch+1) * (2. / self.epoch))
            D = (np.abs(self.pop[idx].solution) + np.abs(self.g_best.solution))*(2 * self.generator.random() - 1)

            # Update the location
            x2 = self.g_best.solution - np.abs((self.generator.integers(1, 3)*M - self.generator.integers(1, 3)*self.pop[idx].solution) * A) * cofi[self.generator.integers(0, 4), :]
            x3 = M + cofi[self.generator.integers(0, 4), :] + (self.generator.integers(1, 3)*self.g_best.solution - self.generator.integers(1, 3)*self.pop[self.generator.integers(self.pop_size)].solution)*cofi[self.generator.integers(0, 4), :]
            x4 = self.pop[idx].solution - D + (self.generator.integers(1, 3)*self.g_best.solution - self.generator.integers(1, 3)*M) * cofi[self.generator.integers(0, 4), :]

            x1 = self.problem.generate_solution()
            x1 = self.correct_solution(x1)
            x2 = self.correct_solution(x2)
            x3 = self.correct_solution(x3)
            x4 = self.correct_solution(x4)

            agent1 = self.generate_empty_agent(x1)
            agent2 = self.generate_empty_agent(x2)
            agent3 = self.generate_empty_agent(x3)
            agent4 = self.generate_empty_agent(x4)

            pop_new += [agent1, agent2, agent3, agent4]
            if self.mode not in self.AVAILABLE_MODES:
                for jdx in range(-4, 0):
                    pop_new[jdx].target = self.get_target(pop_new[jdx].solution)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_new, self.pop_size, self.problem.minmax)
