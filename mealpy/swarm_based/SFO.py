# !/usr/bin/env python
# Created by "Thieu" at 14:51, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseSFO(Optimizer):
    """
    The original version of: SailFish Optimizer (SFO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2019.01.001

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
        + AP (float): coefficient for decreasing the value of Attack Power linearly from AP to 0
        + epxilon (float): should be 0.0001, 0.001

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.SFO import BaseSFO
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
    >>> pp = 0.1
    >>> AP = 4
    >>> epxilon = 0.0001
    >>> model = BaseSFO(problem_dict1, epoch, pop_size, pp, AP, epxilon)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Shadravan, S., Naji, H.R. and Bardsiri, V.K., 2019. The Sailfish Optimizer: A novel
    nature-inspired metaheuristic algorithm for solving constrained engineering optimization
    problems. Engineering Applications of Artificial Intelligence, 80, pp.20-34.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, pp=0.1, AP=4, epxilon=0.0001, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, SailFish pop size
            pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
            AP (float): coefficient for decreasing the value of Power Attack linearly from AP to 0
            epxilon (float): should be 0.0001, 0.001
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.pp = self.validator.check_float("pp", pp, (0, 1.0))
        self.AP = self.validator.check_float("AP", AP, (0, 100))
        self.epxilon = self.validator.check_float("epxilon", epxilon, (0, 0.1))
        self.nfe_per_epoch = 2 * self.pop_size
        self.sort_flag = True
        self.s_size = int(self.pop_size / self.pp)

    def after_initialization(self):
        self.s_pop = self.create_population(self.s_size)
        _, self.g_best = self.get_global_best_solution(self.pop)  # pop = sailfish
        _, self.s_gbest = self.get_global_best_solution(self.s_pop)  # s_pop = sardines

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Calculate lamda_i using Eq.(7)
        ## Update the position of sailfish using Eq.(6)
        nfe_epoch = 0
        pop_new = []
        PD = 1 - self.pop_size / (self.pop_size + self.s_size)
        for idx in range(0, self.pop_size):
            lamda_i = 2 * np.random.uniform() * PD - PD
            pos_new = self.s_gbest[self.ID_POS] - lamda_i * \
                (np.random.uniform() * (self.pop[idx][self.ID_POS] + self.s_gbest[self.ID_POS]) / 2 - self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
        nfe_epoch += self.pop_size

        ## Calculate AttackPower using Eq.(10)
        AP = self.AP * (1 - 2 * (epoch + 1) * self.epxilon)
        if AP < 0.5:
            alpha = int(self.s_size * np.abs(AP))
            beta = int(self.problem.n_dims * np.abs(AP))
            ### Random np.random.choice number of sardines which will be updated their position
            list1 = np.random.choice(range(0, self.s_size), alpha)
            for i in range(0, self.s_size):
                if i in list1:
                    #### Random np.random.choice number of dimensions in sardines updated, remove third loop by numpy vector computation
                    pos_new = deepcopy(self.s_pop[i][self.ID_POS])
                    list2 = np.random.choice(range(0, self.problem.n_dims), beta, replace=False)
                    pos_new[list2] = (np.random.uniform(0, 1, self.problem.n_dims) *
                                      (self.pop[self.ID_POS] - self.s_pop[i][self.ID_POS] + AP))[list2]
                    pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                    self.s_pop[i] = [pos_new, None]
        else:
            ### Update the position of all sardine using Eq.(9)
            for i in range(0, self.s_size):
                pos_new = np.random.uniform() * (self.g_best[self.ID_POS] - self.s_pop[i][self.ID_POS] + AP)
                self.s_pop[i][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
        ## Recalculate the fitness of all sardine
        self.s_pop = self.update_target_wrapper_population(self.s_pop)
        nfe_epoch += self.s_size

        ## Sort the population of sailfish and sardine (for reducing computational cost)
        self.pop, g_best = self.get_global_best_solution(self.pop)
        self.s_pop, s_gbest = self.get_global_best_solution(self.s_pop)
        for i in range(0, self.pop_size):
            for j in range(0, self.s_size):
                ### If there is a better position in sardine population.
                if self.compare_agent(self.s_pop[j], self.pop[i]):
                    self.pop[i] = deepcopy(self.s_pop[j])
                    del self.s_pop[j]
                break  #### This simple keyword helped reducing ton of comparing operation.
                #### Especially when sardine pop size >> sailfish pop size
        temp = self.s_size - len(self.s_pop)
        if temp == 1:
            self.s_pop = self.s_pop + [self.create_solution(self.problem.lb, self.problem.ub)]
        else:
            self.s_pop = self.s_pop + self.create_population(self.s_size - len(self.s_pop))
        _, self.s_gbest = self.get_global_best_solution(self.s_pop)
        self.nfe_per_epoch = nfe_epoch


class ImprovedSFO(Optimizer):
    """
    My improved version of: Sailfish Optimizer (I-SFO)

    Notes
    ~~~~~
    + Reforms Energy equation
    + Removes parameters AP (A) and epsilon
    + Applies the idea of Opposition-based Learning technique

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.SFO import ImprovedSFO
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
    >>> pp = 0.1
    >>> model = ImprovedSFO(problem_dict1, epoch, pop_size, pp)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Socha, K. and Dorigo, M., 2008. Ant colony optimization for continuous domains.
    European journal of operational research, 185(3), pp.1155-1173.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, pp=0.1, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, SailFish pop size
            pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.pp = self.validator.check_float("pp", pp, (0, 1.0))

        self.nfe_per_epoch = 2 * self.pop_size
        self.sort_flag = True
        self.s_size = int(self.pop_size / self.pp)

    def after_initialization(self):
        self.s_pop = self.create_population(self.s_size)
        _, self.g_best = self.get_global_best_solution(self.pop)
        _, self.s_gbest = self.get_global_best_solution(self.s_pop)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Calculate lamda_i using Eq.(7)
        ## Update the position of sailfish using Eq.(6)
        nfe_epoch = 0
        pop_new = []
        for idx in range(0, self.pop_size):
            PD = 1 - len(self.pop) / (len(self.pop) + len(self.s_pop))
            lamda_i = 2 * np.random.uniform() * PD - PD
            pos_new = self.s_gbest[self.ID_POS] - \
                lamda_i * (np.random.uniform() * (self.g_best[self.ID_POS] + self.s_gbest[self.ID_POS]) / 2 - self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
        nfe_epoch += self.pop_size

        ## ## Calculate AttackPower using my Eq.thieu
        #### This is our proposed, simple but effective, no need A and epxilon parameters
        AP = 1 - epoch * 1.0 / self.epoch
        if AP < 0.5:
            for i in range(0, len(self.s_pop)):
                temp = (self.g_best[self.ID_POS] + AP) / 2
                pos_new = self.problem.lb + self.problem.ub - temp + np.random.uniform() * (temp - self.s_pop[i][self.ID_POS])
                self.s_pop[i][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
        else:
            ### Update the position of all sardine using Eq.(9)
            for i in range(0, len(self.s_pop)):
                pos_new = np.random.uniform() * (self.g_best[self.ID_POS] - self.s_pop[i][self.ID_POS] + AP)
                self.s_pop[i][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
        ## Recalculate the fitness of all sardine
        self.s_pop = self.update_target_wrapper_population(self.s_pop)
        nfe_epoch += len(self.s_pop)

        ## Sort the population of sailfish and sardine (for reducing computational cost)
        self.pop = self.get_sorted_strim_population(self.pop, self.pop_size)
        self.s_pop = self.get_sorted_strim_population(self.s_pop, len(self.s_pop))
        for i in range(0, self.pop_size):
            for j in range(0, len(self.s_pop)):
                ### If there is a better position in sardine population.
                if self.compare_agent(self.s_pop[j], self.pop[i]):
                    self.pop[i] = deepcopy(self.s_pop[j])
                    del self.s_pop[j]
                break  #### This simple keyword helped reducing ton of comparing operation.
                #### Especially when sardine pop size >> sailfish pop size

        self.s_pop = self.s_pop + self.create_population(self.s_size - len(self.s_pop))
        _, self.s_gbest = self.get_global_best_solution(self.s_pop)
        self.nfe_per_epoch = nfe_epoch
