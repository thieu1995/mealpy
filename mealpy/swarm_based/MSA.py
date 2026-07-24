#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo


class OriginalMSA(Optimizer):
    """
    The original version: Moth Search Algorithm (MSA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations. Default is 10000.
    pop_size : int
        Population size. Default is 100.
    n_best : int
        How many of the best moths to keep from one generation to the next, in range [3, 10]. Default is 5.
    partition : float
        The proportional of first partition, in range [0.3, 0.8]. Default is 0.5.
    max_step_size : float
        Max step size used in Levy-flight technique, in range [0.5, 2.0]. Default is 1.0.

    Links
    -----
    1. https://www.mathworks.com/matlabcentral/fileexchange/59010-moth-search-ms-algorithm
    2. https://doi.org/10.1007/s12293-016-0212-3

    References
    ----------
    1. Wang, G.G., 2018. Moth search algorithm: a bio-inspired metaheuristic algorithm for
       global optimization problems. Memetic Computing, 10(2), pp.151-164.

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, MSA
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
    >>> model = MSA.OriginalMSA(epoch=1000, pop_size=50, n_best = 5, partition = 0.5, max_step_size = 1.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Moth Search Algorithm", year=2018, difficulty="medium", kind="original")

    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_best: int = 5, partition: float = 0.5, max_step_size: float = 1.0, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_best (int): how many of the best moths to keep from one generation to the next, default=5
            partition (float): The proportional of first partition, default=0.5
            max_step_size (float): Max step size used in Levy-flight technique, default=1.0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_best = self.validator.check_int("n_best", n_best, [2, int(self.pop_size/2)])
        self.partition = self.validator.check_float("partition", partition, (0, 1.0))
        self.max_step_size = self.validator.check_float("max_step_size", max_step_size, [0, self.epoch])
        self.set_parameters(["epoch", "pop_size", "n_best", "partition", "max_step_size"])
        self.sort_flag = True
        # np1 in paper
        self.n_moth1 = int(np.ceil(self.partition * self.pop_size))
        # np2 in paper, we actually don't need this variable
        self.n_moth2 = self.pop_size - self.n_moth1
        # you can change this ratio so as to get much better performance
        self.golden_ratio = (np.sqrt(5) - 1) / 2.0

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_best = [agent.copy() for agent in self.pop[:self.n_best]]
        pop_new = []
        for idx in range(0, self.pop_size):
            # Migration operator
            if idx < self.n_moth1:
                scale = self.max_step_size / epoch      # Smaller step for local walk
                pos_new = (self.pop[idx].solution + self.generator.random(self.problem.n_dims) *
                           self.get_levy_flight_step(beta=1.5, multiplier=scale, size=self.problem.n_dims, case=-1))
            else:
                # Flying in a straight line
                temp_case1 = self.pop[idx].solution + self.generator.random(self.problem.n_dims) * \
                             self.golden_ratio * (self.g_best.solution - self.pop[idx].solution)
                temp_case2 = self.pop[idx].solution + self.generator.random(self.problem.n_dims) * \
                             (1.0 / self.golden_ratio) * (self.g_best.solution - self.pop[idx].solution)
                pos_new = np.where(self.generator.random(self.problem.n_dims) < 0.5, temp_case2, temp_case1)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        self.pop, _ = self.get_sorted_population(self.pop, self.problem.minmax)
        # Replace the worst with the previous generation's elites.
        for idx in range(0, self.n_best):
            self.pop[-1 - idx] = pop_best[idx].copy()
