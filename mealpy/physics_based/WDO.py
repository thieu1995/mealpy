# !/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseWDO(Optimizer):
    """
    The original version of: Wind Driven Optimization (WDO)

    Links:
        1. https://ieeexplore.ieee.org/abstract/document/6407788

    Notes
    ~~~~~
    + pop is the set of "air parcel" - "position"
    + air parcel: is the set of gas atoms. Each atom represents a dimension in position and has its own velocity
    + pressure represented by fitness value

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + RT (int): [2, 3, 4], RT coefficient, default = 3
        + g_c (float): [0.1, 0.5], gravitational constant, default = 0.2
        + alp (float): [0.3, 0.8], constants in the update equation, default=0.4
        + c_e (float): [0.1, 0.9], coriolis effect, default=0.4
        + max_v (float): [0.1, 0.9], maximum allowed speed, default=0.3

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.WDO import BaseWDO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> RT = 3
    >>> g_c = 0.2
    >>> alp = 0.4
    >>> c_e = 0.4
    >>> max_v = 0.3
    >>> model = BaseWDO(problem_dict1, epoch, pop_size, RT, g_c, alp, c_e, max_v)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Bayraktar, Z., Komurcu, M., Bossard, J.A. and Werner, D.H., 2013. The wind driven optimization
    technique and its application in electromagnetics. IEEE transactions on antennas and
    propagation, 61(5), pp.2745-2757.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, RT=3, g_c=0.2, alp=0.4, c_e=0.4, max_v=0.3, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            RT (int): RT coefficient, default = 3
            g_c (float): gravitational constant, default = 0.2
            alp (float): constants in the update equation, default=0.4
            c_e (float): coriolis effect, default=0.4
            max_v (float): maximum allowed speed, default=0.3
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.RT = RT
        self.g_c = g_c
        self.alp = alp
        self.c_e = c_e
        self.max_v = max_v

        ## Dynamic variable
        self.dyn_list_velocity = self.max_v * np.random.uniform(self.problem.lb, self.problem.ub, (self.pop_size, self.problem.n_dims))

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            rand_dim = np.random.randint(0, self.problem.n_dims)
            temp = self.dyn_list_velocity[idx][rand_dim] * np.ones(self.problem.n_dims)
            vel = (1 - self.alp) * self.dyn_list_velocity[idx] - self.g_c * self.pop[idx][self.ID_POS] + \
                  (1 - 1.0 / (idx + 1)) * self.RT * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + self.c_e * temp / (idx + 1)
            vel = np.clip(vel, -self.max_v, self.max_v)

            # Update air parcel positions, check the bound and calculate pressure (fitness)
            self.dyn_list_velocity[idx] = vel
            pos = self.pop[idx][self.ID_POS] + vel
            pos_new = self.amend_position(pos)
            pop_new.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_new)
