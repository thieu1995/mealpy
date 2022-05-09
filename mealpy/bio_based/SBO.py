#!/usr/bin/env python
# Created by "Thieu" at 12:48, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseSBO(Optimizer):
    """
    My changed version of: Satin Bowerbird Optimizer (SBO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2017.01.006

    Notes
    ~~~~~
    The original version is not good enough, I remove all third loop for faster training, remove equation (1, 2) in the paper,
    calculate probability by roulette-wheel. My version can also handle negative value

    Hyper-parameters should fine tuned in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [0.5, 2.0], the greatest step size
        + p_m (float): [0.01, 0.2], mutation probability
        + psw (float): [0.01, 0.1], proportion of space width (z in the paper)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.SBO import BaseSBO
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
    >>> alpha = 0.9
    >>> p_m =0.05
    >>> psw = 0.02
    >>> model = BaseSBO(problem_dict1, epoch, pop_size, alpha, p_m, psw)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, problem, epoch=10000, pop_size=100, alpha=0.94, p_m=0.05, psw=0.02, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): the greatest step size, default=0.94
            p_m (float): mutation probability, default=0.05
            psw (float): proportion of space width (z in the paper), default=0.02
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.alpha = self.validator.check_float("alpha", alpha, [0.5, 3.0])
        self.p_m = self.validator.check_float("p_m", p_m, (0, 1.0))
        self.psw = self.validator.check_float("psw", psw, (0, 1.0))

        # (percent of the difference between the upper and lower limit (Eq. 7))
        self.sigma = self.psw * (self.problem.ub - self.problem.lb)
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Calculate the probability of bowers using my equation
        fit_list = np.array([item[self.ID_TAR][self.ID_FIT] for item in self.pop])
        pop_new = []
        for i in range(0, self.pop_size):
            ### Select a bower using roulette wheel
            idx = self.get_index_roulette_wheel_selection(fit_list)
            ### Calculating Step Size
            lamda = self.alpha * np.random.uniform()
            pos_new = self.pop[i][self.ID_POS] + lamda * ((self.pop[idx][self.ID_POS] + self.g_best[self.ID_POS]) / 2 - self.pop[i][self.ID_POS])
            ### Mutation
            temp = self.pop[i][self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * self.sigma
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.p_m, temp, pos_new)
            ### In-bound position
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
        self.pop = self.update_target_wrapper_population(pop_new)


class OriginalSBO(BaseSBO):
    """
    The original version of: Satin Bowerbird Optimizer (SBO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2017.01.006
        2. https://www.mathworks.com/matlabcentral/fileexchange/62009-satin-bowerbird-optimizer-sbo-2017

    Hyper-parameters should fine tuned in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [0.5, 0.99], the greatest step size
        + p_m (float): [0.01, 0.2], mutation probability
        + psw (float): [0.01, 0.1], proportion of space width (z in the paper)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.SBO import OriginalSBO
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
    >>> alpha = 0.9
    >>> p_m=0.05
    >>> psw = 0.02
    >>> model = OriginalSBO(problem_dict1, epoch, pop_size, alpha, p_m, psw)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Moosavi, S.H.S. and Bardsiri, V.K., 2017. Satin bowerbird optimizer: A new optimization algorithm
    to optimize ANFIS for software development effort estimation. Engineering Applications of Artificial Intelligence, 60, pp.1-15.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, alpha=0.94, p_m=0.05, psw=0.02, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): the greatest step size, default=0.94
            p_m (float): mutation probability, default=0.05
            psw (float): proportion of space width (z in the paper), default=0.02
        """
        super().__init__(problem, epoch, pop_size, alpha, p_m, psw, **kwargs)

    def _roulette_wheel_selection__(self, fitness_list=None) -> int:
        """
        Roulette Wheel Selection in the original version, this version can't handle the negative fitness values

        Args:
            fitness_list (list): Fitness of population

        Returns:
            f (int): The index of selected solution
        """
        r = np.random.uniform()
        c = np.cumsum(fitness_list)
        f = np.where(r < c)[0][0]
        return f

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Calculate the probability of bowers using Eqs. (1) and (2)
        fx_list = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in self.pop])
        fit_list = deepcopy(fx_list)
        for i in range(0, self.pop_size):
            if fx_list[i] < 0:
                fit_list[i] = 1.0 + np.abs(fx_list[i])
            else:
                fit_list[i] = 1.0 / (1.0 + np.abs(fx_list[i]))
        fit_sum = np.sum(fit_list)
        ## Calculating the probability of each bower
        prob_list = fit_list / fit_sum
        pop_new = []
        for i in range(0, self.pop_size):
            pos_new = deepcopy(self.pop[i][self.ID_POS])
            for j in range(0, self.problem.n_dims):
                ### Select a bower using roulette wheel
                idx = self._roulette_wheel_selection__(prob_list)
                ### Calculating Step Size
                lamda = self.alpha / (1 + prob_list[idx])
                pos_new[j] = self.pop[i][self.ID_POS][j] + lamda * \
                             ((self.pop[idx][self.ID_POS][j] + self.g_best[self.ID_POS][j]) / 2 - self.pop[i][self.ID_POS][j])
                ### Mutation
                if np.random.uniform() < self.p_m:
                    pos_new[j] = self.pop[i][self.ID_POS][j] + np.random.normal(0, 1) * self.sigma[j]
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
        self.pop = self.update_target_wrapper_population(pop_new)
