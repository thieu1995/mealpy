#!/usr/bin/env python
# Created by "Thieu" at 19:38, 10/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from scipy.stats import qmc
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
    >>> from mealpy import FloatVar, PSS
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
    >>> model = PSS.OriginalPSS(epoch=1000, pop_size=50, acceptance_rate = 0.8, sampling_method = "LHS")
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Shaqfa, M. and Beyer, K., 2021. Pareto-like sequential sampling heuristic for global optimisation. Soft Computing, 25(14), pp.9077-9096.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, acceptance_rate: float = 0.9, sampling_method: str = "LHS", **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            acceptance_rate (float): the probability of accepting a solution in the normal range, default = 0.9
            sampling_method (str): 'LHS': Latin-Hypercube or 'MC': 'MonteCarlo', default = "LHS"
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
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
            pop = self.generator.random(self.pop_size, self.problem.n_dims)
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
            pos_new = self.correct_solution(pos)
            agent = self.generate_agent(pos_new)
            self.pop.append(agent)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        pop_rand = self.create_population(self.pop_size)
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx].solution.copy()
            for k in range(self.problem.n_dims):
                # Update the ranges
                deviation = self.generator.uniform(min(0, self.g_best.solution[k]), max(0, self.g_best.solution[k]))
                if self.new_solution:
                    # The deviation is positive dynamic real number
                    deviation = abs(0.5 * (1. - self.acceptance_rate) * (self.problem.ub[k] - self.problem.lb[k])) * (1 - (epoch / self.epoch))
                reduced_lb = self.g_best.solution[k] - deviation
                reduced_lb = np.amax([reduced_lb, self.problem.lb[k]])
                reduced_ub = reduced_lb + deviation * 2.
                reduced_ub = np.amin([reduced_ub, self.problem.ub[k]])
                # Choose new solution
                if self.generator.random() <= self.acceptance_rate:
                    # choose a solution from the prominent domain
                    pos_new[k] = reduced_lb + pop_rand[idx, k] * (reduced_ub - reduced_lb)
                else:
                    # choose a solution from the overall domain
                    pos_new[k] = self.problem.lb[k] + pop_rand[idx, k] * (self.problem.ub[k] - self.problem.lb[k])
                # Round for the step size
                pos_new = np.round(pos_new / self.steps) * self.steps
            # Check the bound
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        self.pop = self.update_target_for_population(pop_new)
        current_best = self.get_best_agent(pop_new, self.problem.minmax)
        if self.compare_target(current_best.target, self.g_best.target, self.problem.minmax):
            self.new_solution = True
        else:
            self.new_solution = False
