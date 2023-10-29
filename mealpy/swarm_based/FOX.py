#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalFOX(Optimizer):
    """
    The original version of: Fox Optimizer (FOX)

    Links:
        1. https://link.springer.com/article/10.1007/s10489-022-03533-0
        2. https://www.mathworks.com/matlabcentral/fileexchange/121592-fox-a-fox-inspired-optimization-algorithm

    Notes (parameters):
        1. c1 (float): the probability of jumping (c1 in the paper), default = 0.18
        2. c2 (float): the probability of jumping (c2 in the paper), default = 0.82

    Notes:
        1. The equation used to calculate the distance_S_travel value in the Matlab code seems to be lacking in meaning.
        2. The if-else conditions used with p > 0.18 seem to lack a clear justification. The authors seem to have simply chosen the best value based on their experiments without explaining the rationale behind it.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, FOX
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
    >>> model = FOX.OriginalFOX(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mohammed, H., & Rashid, T. (2023). FOX: a FOX-inspired optimization algorithm. Applied Intelligence, 53(1), 1030-1050.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, c1: float = 0.18, c2: float = 0.82, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c1 (float): the probability of jumping (c1 in the paper), default = 0.18
            c2 (float): the probability of jumping (c2 in the paper), default = 0.82
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, (-100., 100.))      # c1 in the paper
        self.c2 = self.validator.check_float("c2", c2, (-100., 100.))      # c2 in the paper
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def initialize_variables(self):
        self.mint = 10000000

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        aa = 2 * (1 - (1.0 / self.epoch))
        pop_new = []
        for idx in range(0, self.pop_size):
            if self.generator.random() >= 0.5:
                t1 = self.generator.random(self.problem.n_dims)
                sps = self.g_best.solution / t1
                dis = 0.5 * sps * t1
                tt = np.mean(t1)
                t = tt / 2
                jump = 0.5 * 9.81 * t ** 2
                if self.generator.random() > 0.18:
                    pos_new = dis * jump * self.c1
                else:
                    pos_new = dis * jump * self.c2
                if self.mint > tt:
                    self.mint = tt
            else:
                pos_new = self.g_best.solution + self.generator.standard_normal(self.problem.n_dims) * (self.mint * aa)
            pos_new =self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = agent
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)
