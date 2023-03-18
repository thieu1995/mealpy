#!/usr/bin/env python
# Created by "Thieu" at 19:38, 10/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from scipy.stats import qmc
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalPSS(Optimizer):
    """
    The original version of: Pareto-like Sequential Sampling (PSS)

    Links:
        1. https://doi.org/10.1007/s00500-021-05853-8
        2. https://github.com/eesd-epfl/pareto-optimizer

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + acceptance_rate (float): [0.7-0.96], the probability of accepting a solution in the normal range, default=0.9
        + sampling_method (str): 'LHS': Latin-Hypercube or 'MC': 'MonteCarlo', default="LHS"

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.math_based.PSS import OriginalPSS
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
    >>> acceptance_rate = 0.8
    >>> sampling_method = "LHS"
    >>> model = OriginalPSS(epoch, pop_size, acceptance_rate, sampling_method)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Shaqfa, M. and Beyer, K., 2021. Pareto-like sequential sampling heuristic for global optimisation. Soft Computing, 25(14), pp.9077-9096.
    """

    def __init__(self, epoch=10000, pop_size=100, acceptance_rate=0.9, sampling_method="LHS", **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            acceptance_rate (float): the probability of accepting a solution in the normal range, default = 0.9
            sampling_method (str): 'LHS': Latin-Hypercube or 'MC': 'MonteCarlo', default = "LHS"
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.acceptance_rate = self.validator.check_float("acceptance_rate", acceptance_rate, (0, 1.0))
        self.sampling_method = self.validator.check_str("sampling_method", sampling_method, ["MC", "LHS"])
        self.set_parameters(["epoch", "pop_size", "acceptance_rate", "sampling_method"])
        self.sort_flag = False

    def initialize_variables(self):
        self.step = 10e-10
        self.steps = np.ones(self.problem.n_dims) * self.step
        self.new_solution = True

    def create_population(self, pop_size=None):
        if self.sampling_method == "MC":
            pop = np.random.rand(self.pop_size, self.problem.n_dims)
        else:       # Default: "LHS"
            sampler = qmc.LatinHypercube(d=self.problem.n_dims)
            pop = sampler.random(n=pop_size)
        return pop

    def initialization(self):
        lb_pop = np.repeat(np.reshape(self.problem.lb, (1, -1)), self.pop_size, axis=0)
        ub_pop = np.repeat(np.reshape(self.problem.ub, (1, -1)), self.pop_size, axis=0)
        steps_mat = np.repeat(np.reshape(self.steps, (1, -1)), self.pop_size, axis=0)

        random_pop = self.create_population(self.pop_size)
        pop = np.round((lb_pop + random_pop * (ub_pop - lb_pop)) / steps_mat) * steps_mat
        self.pop = []
        for pos in pop:
            pos_new = self.amend_position(pos, self.problem.lb, self.problem.ub)
            target = self.get_target_wrapper(pos_new)
            self.pop.append([pos_new, target])

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        pop_rand = self.create_population(self.pop_size)
        for idx in range(0, self.pop_size):
            pos_new = deepcopy(self.pop[idx][self.ID_POS]).astype(float)
            for k in range(self.problem.n_dims):
                # Update the ranges
                deviation = np.random.uniform(0, self.g_best[self.ID_POS][k])
                if self.new_solution:
                    # The deviation is positive dynamic real number
                    deviation = abs(0.5 * (1. - self.acceptance_rate) * (self.problem.ub[k] - self.problem.lb[k])) * (1 - ((epoch+1) / self.epoch))

                reduced_lb = self.g_best[self.ID_POS][k] - deviation
                reduced_lb = np.amax([reduced_lb, self.problem.lb[k]])

                reduced_ub = reduced_lb + deviation * 2.
                reduced_ub = np.amin([reduced_ub, self.problem.ub[k]])

                # Choose new solution
                if np.random.rand() <= self.acceptance_rate:
                    # choose a solution from the prominent domain
                    pos_new[k] = reduced_lb + pop_rand[idx, k] * (reduced_ub - reduced_lb)
                else:
                    # choose a solution from the overall domain
                    pos_new[k] = self.problem.lb[k] + pop_rand[idx, k] * (self.problem.ub[k] - self.problem.lb[k])

                # Round for the step size
                pos_new = np.round(pos_new / self.steps) * self.steps
            # Check the bound
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = pop_new
        _, current_best = self.get_global_best_solution(pop_new)
        if self.compare_agent(current_best, self.g_best):
            self.new_solution = True
        else:
            self.new_solution = False
