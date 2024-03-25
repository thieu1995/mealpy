#!/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalWDO(Optimizer):
    """
    The original version of: Wind Driven Optimization (WDO)

    Notes
        + pop is the set of "air parcel" - "position"
        + air parcel: is the set of gas atoms. Each atom represents a dimension in position and has its own velocity
        + pressure represented by fitness value
        + https://ieeexplore.ieee.org/abstract/document/6407788

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + RT (int): [2, 3, 4], RT coefficient, default = 3
        + g_c (float): [0.1, 0.5], gravitational constant, default = 0.2
        + alp (float): [0.3, 0.8], constants in the update equation, default=0.4
        + c_e (float): [0.1, 0.9], coriolis effect, default=0.4
        + max_v (float): [0.1, 0.9], maximum allowed speed, default=0.3

    Examples
    ~~~~~~~~

    >>> import numpy as np
    >>> from mealpy import FloatVar, WDO
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
    >>> model = WDO.OriginalWDO(epoch=1000, pop_size=50, RT = 3, g_c = 0.2, alp = 0.4, c_e = 0.4, max_v = 0.3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Bayraktar, Z., Komurcu, M., Bossard, J.A. and Werner, D.H., 2013. The wind driven optimization
    technique and its application in electromagnetics. IEEE transactions on antennas and
    propagation, 61(5), pp.2745-2757.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, RT: int = 3, g_c: float = 0.2,
                 alp: float = 0.4, c_e: float = 0.4, max_v: float = 0.3, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            RT (int): RT coefficient, default = 3
            g_c (float): gravitational constant, default = 0.2
            alp (float): constants in the update equation, default=0.4
            c_e (float): coriolis effect, default=0.4
            max_v (float): maximum allowed speed, default=0.3
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.RT = self.validator.check_int("RT", RT, [1, 4])
        self.g_c = self.validator.check_float("g_c", g_c, (0, 1.0))
        self.alp = self.validator.check_float("alp", alp, (0, 1.0))
        self.c_e = self.validator.check_float("c_e", c_e, (0, 1.0))
        self.max_v = self.validator.check_float("max_v", max_v, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "RT", "g_c", "alp", "c_e", "max_v"])
        self.sort_flag = False

    def initialize_variables(self):
        self.dyn_list_velocity = self.max_v * self.generator.uniform(self.problem.lb, self.problem.ub, (self.pop_size, self.problem.n_dims))

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            rand_dim = self.generator.integers(0, self.problem.n_dims)
            temp = self.dyn_list_velocity[idx][rand_dim] * np.ones(self.problem.n_dims)
            vel = (1 - self.alp) * self.dyn_list_velocity[idx] - self.g_c * self.pop[idx].solution + \
                  (1 - 1.0 / (idx + 1)) * self.RT * (self.g_best.solution - self.pop[idx].solution) + self.c_e * temp / (idx + 1)
            vel = np.clip(vel, -self.max_v, self.max_v)
            # Update air parcel positions, check the bound and calculate pressure (fitness)
            self.dyn_list_velocity[idx] = vel
            pos = self.pop[idx].solution + vel
            pos_new = self.correct_solution(pos)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
