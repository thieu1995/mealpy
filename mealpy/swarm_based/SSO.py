#!/usr/bin/env python
# Created by "Thieu" at 11:38, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSSO(Optimizer):
    """
    The original version of: Salp Swarm Optimization (SSO)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2017.07.002

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SSO
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
    >>> model = SSO.OriginalSSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., Gandomi, A.H., Mirjalili, S.Z., Saremi, S., Faris, H. and Mirjalili, S.M., 2017.
    Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems. Advances in Engineering Software, 114, pp.163-191.
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

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Eq. (3.2) in the paper
        c1 = 2 * np.exp(-((4 * epoch / self.epoch) ** 2))
        pop_new = []
        for idx in range(0, self.pop_size):
            if idx < self.pop_size / 2:
                c2_list = self.generator.random(self.problem.n_dims)
                c3_list = self.generator.random(self.problem.n_dims)
                pos_new_1 = self.g_best.solution + c1 * ((self.problem.ub - self.problem.lb) * c2_list + self.problem.lb)
                pos_new_2 = self.g_best.solution - c1 * ((self.problem.ub - self.problem.lb) * c2_list + self.problem.lb)
                pos_new = np.where(c3_list < 0.5, pos_new_1, pos_new_2)
            else:
                # Eq. (3.4) in the paper
                pos_new = (self.pop[idx].solution + self.pop[idx - 1].solution) / 2
            # Check if salps go out of the search space and bring it back then re-calculate its fitness value
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
