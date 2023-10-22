#!/usr/bin/env python
# Created by "Thieu" at 12:17, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalIWO(Optimizer):
    """
    The original version of: Invasive Weed Optimization (IWO)

    Links:
        1. https://pdfs.semanticscholar.org/734c/66e3757620d3d4016410057ee92f72a9853d.pdf

    Notes:
        Better to use normal distribution instead of uniform distribution,
        updating population by sorting both parent population and child population

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + seed_min (int): [1, 3], Number of Seeds (min)
        + seed_max (int): [4, pop_size/2], Number of Seeds (max)
        + exponent (int): [2, 4], Variance Reduction Exponent
        + sigma_start (float): [0.5, 5.0], The initial value of Standard Deviation
        + sigma_end (float): (0, 0.5), The final value of Standard Deviation

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, EOA
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
    >>> model = EOA.OriginalEOA(epoch=1000, pop_size=50, seed_min = 3, seed_max = 9, exponent = 3, sigma_start = 0.6, sigma_end = 0.01)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mehrabian, A.R. and Lucas, C., 2006. A novel numerical optimization algorithm inspired from weed colonization.
    Ecological informatics, 1(4), pp.355-366.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, seed_min: int = 2, seed_max: int = 10,
                 exponent: int = 2, sigma_start: float = 1.0, sigma_end: float = 0.01, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            seed_min (int): Number of Seeds (min)
            seed_max (int): Number of seeds (max)
            exponent (int): Variance Reduction Exponent
            sigma_start (float): The initial value of standard deviation
            sigma_end (float): The final value of standard deviation
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.seed_min = self.validator.check_int("seed_min", seed_min, [1, 3])
        self.seed_max = self.validator.check_int("seed_max", seed_max, [4, int(self.pop_size/2)])
        self.exponent = self.validator.check_int("exponent", exponent, [2, 4])
        self.sigma_start = self.validator.check_float("sigma_start", sigma_start, [0.5, 5.0])
        self.sigma_end = self.validator.check_float("sigma_end", sigma_end, (0, 0.5))
        self.set_parameters(["epoch", "pop_size", "seed_min", "seed_max", "exponent", "sigma_start", "sigma_end"])
        self.sort_flag = True

    def evolve(self, epoch=None):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Update Standard Deviation
        sigma = ((self.epoch - epoch) / (self.epoch - 1)) ** self.exponent * (self.sigma_start - self.sigma_end) + self.sigma_end
        pop, list_best, list_worst = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)
        best, worst = list_best[0], list_worst[0]
        pop_new = []
        for idx in range(0, self.pop_size):
            temp = best.target.fitness - worst.target.fitness
            if temp == 0:
                ratio = self.generator.random()
            else:
                ratio = (pop[idx].target.fitness - worst.target.fitness) / temp
            s = int(np.ceil(self.seed_min + (self.seed_max - self.seed_min) * ratio))
            if s > int(np.sqrt(self.pop_size)):
                s = int(np.sqrt(self.pop_size))
            pop_local = []
            for jdx in range(s):
                # Initialize Offspring and Generate Random Location
                pos_new = pop[idx].solution + sigma * self.generator.normal(0, 1, self.problem.n_dims)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_local.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_local[-1].target = self.get_target(pos_new)
            if self.mode in self.AVAILABLE_MODES:
                pop_local = self.update_target_for_population(pop_local)
            pop_new += pop_local
        self.pop = self.get_sorted_and_trimmed_population(pop_new, self.pop_size, self.problem.minmax)
