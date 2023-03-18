#!/usr/bin/env python
# Created by "Thieu" at 21:19, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseEFO(Optimizer):
    """
    The developed version: Electromagnetic Field Optimization (EFO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r_rate (float): [0.1, 0.6], default = 0.3, like mutation parameter in GA but for one variable
        + ps_rate (float): [0.5, 0.95], default = 0.85, like crossover parameter in GA
        + p_field (float): [0.05, 0.3], default = 0.1, portion of population, positive field
        + n_field (float): [0.3, 0.7], default = 0.45, portion of population, negative field

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.EFO import BaseEFO
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
    >>> r_rate = 0.3
    >>> ps_rate = 0.85
    >>> p_field = 0.1
    >>> n_field = 0.45
    >>> model = BaseEFO(epoch, pop_size, r_rate, ps_rate, p_field, n_field)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_rate (float): default = 0.3     Like mutation parameter in GA but for one variable
            ps_rate (float): default = 0.85    Like crossover parameter in GA
            p_field (float): default = 0.1     portion of population, positive field
            n_field (float): default = 0.45    portion of population, negative field
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.r_rate = self.validator.check_float("r_rate", r_rate, (0, 1.0))
        self.ps_rate = self.validator.check_float("ps_rate", ps_rate, (0, 1.0))
        self.p_field = self.validator.check_float("p_field", p_field, (0, 1.0))
        self.n_field = self.validator.check_float("n_field", n_field, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "r_rate", "ps_rate", "p_field", "n_field"])
        self.phi = (1 + np.sqrt(5)) / 2  # golden ratio
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            r_idx1 = np.random.randint(0, int(self.pop_size * self.p_field))  # top
            r_idx2 = np.random.randint(int(self.pop_size * (1 - self.n_field)), self.pop_size)  # bottom
            r_idx3 = np.random.randint(int((self.pop_size * self.p_field) + 1), int(self.pop_size * (1 - self.n_field)))  # middle
            if np.random.rand() < self.ps_rate:
                # new = g_best + phi* r1 * (top - middle) + r2 (top - bottom)
                # pos_new = g_best[self.ID_POS] + \
                #            phi * np.random.uniform() * (pop[r_idx1][self.ID_POS] - pop[r_idx3][self.ID_POS]) + \
                #            np.random.uniform() * (pop[r_idx1][self.ID_POS] - pop[r_idx2][self.ID_POS])
                # new = top + phi * r1 * (g_best - bottom) + r2 * (g_best - middle)
                pos_new = self.pop[r_idx1][self.ID_POS] + self.phi * np.random.rand() * (self.g_best[self.ID_POS] - self.pop[r_idx3][self.ID_POS]) \
                          + np.random.rand() * (self.g_best[self.ID_POS] - self.pop[r_idx2][self.ID_POS])
            else:
                pos_new = self.generate_position(self.problem.lb, self.problem.ub)

            # replacement of one electromagnet of generated particle with a random number
            # (only for some generated particles) to bring diversity to the population
            if np.random.rand() < self.r_rate:
                RI = np.random.randint(0, self.problem.n_dims)
                pos_new[np.random.randint(0, self.problem.n_dims)] = np.random.uniform(self.problem.lb[RI], self.problem.ub[RI])

            # checking whether the generated number is inside boundary or not
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)


class OriginalEFO(BaseEFO):
    """
    The original version of: Electromagnetic Field Optimization (EFO)

    Links:
        2. https://www.mathworks.com/matlabcentral/fileexchange/52744-electromagnetic-field-optimization-a-physics-inspired-metaheuristic-optimization-algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r_rate (float): [0.1, 0.6], default = 0.3, like mutation parameter in GA but for one variable
        + ps_rate (float): [0.5, 0.95], default = 0.85, like crossover parameter in GA
        + p_field (float): [0.05, 0.3], default = 0.1, portion of population, positive field
        + n_field (float): [0.3, 0.7], default = 0.45, portion of population, negative field

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.EFO import OriginalEFO
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
    >>> r_rate = 0.3
    >>> ps_rate = 0.85
    >>> p_field = 0.1
    >>> n_field = 0.45
    >>> model = OriginalEFO(epoch, pop_size, r_rate, ps_rate, p_field, n_field)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Abedinpourshotorban, H., Shamsuddin, S.M., Beheshti, Z. and Jawawi, D.N., 2016.
    Electromagnetic field optimization: a physics-inspired metaheuristic optimization algorithm.
    Swarm and Evolutionary Computation, 26, pp.8-22.
    """

    def __init__(self, epoch=10000, pop_size=100, r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_rate (float): default = 0.3     Like mutation parameter in GA but for one variable
            ps_rate (float): default = 0.85    Like crossover parameter in GA
            p_field (float): default = 0.1     portion of population, positive field
            n_field (float): default = 0.45    portion of population, negative field
        """
        super().__init__(epoch, pop_size, r_rate, ps_rate, p_field, n_field, **kwargs)
        self.support_parallel_modes = False

    def amend_position(self, position=None, lb=None, ub=None):
        """
        Depend on what kind of problem are we trying to solve, there will be an different amend_position
        function to rebound the position of agent into the valid range.

        Args:
            position: vector position (location) of the solution.
            lb: list of lower bound values
            ub: list of upper bound values

        Returns:
            Amended position (make the position is in bound)
        """
        return np.where(np.logical_and(lb <= position, position <= ub), position, np.random.uniform(lb, ub))

    def initialization(self):
        # %random vectors (this is to increase the calculation speed instead of determining the random values in each
        # iteration we allocate them in the beginning before algorithm start
        self.r_index1 = np.random.randint(0, int(self.pop_size * self.p_field), (self.problem.n_dims, self.epoch))
        # random particles from positive field
        self.r_index2 = np.random.randint(int(self.pop_size * (1 - self.n_field)), self.pop_size, (self.problem.n_dims, self.epoch))
        # random particles from negative field
        self.r_index3 = np.random.randint(int((self.pop_size * self.p_field) + 1), int(self.pop_size * (1 - self.n_field)), (self.problem.n_dims, self.epoch))
        # random particles from neutral field
        self.ps = np.random.uniform(0, 1, (self.problem.n_dims, self.epoch))
        # Probability of selecting electromagnets of generated particle from the positive field
        self.r_force = np.random.uniform(0, 1, self.epoch)
        # random force in each generation
        self.rp = np.random.uniform(0, 1, self.epoch)
        # Some random numbers for checking randomness probability in each generation
        self.randomization = np.random.uniform(0, 1, self.epoch)
        # Coefficient of randomization when generated electro magnet is out of boundary
        self.RI = 0
        # index of the electromagnet (variable) which is going to be initialized by random number

        if self.pop is None:
            self.pop = self.create_population(self.pop_size)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        r = self.r_force[epoch]
        x_new = np.zeros(self.problem.n_dims)  # temporary array to store generated particle
        for i in range(0, self.problem.n_dims):
            if self.ps[i, epoch] > self.ps_rate:
                x_new[i] = self.pop[self.r_index3[i, epoch]][self.ID_POS][i] + \
                           self.phi * r * (self.pop[self.r_index1[i, epoch]][self.ID_POS][i] - self.pop[self.r_index3[i, epoch]][self.ID_POS][i]) + \
                           r * (self.pop[self.r_index3[i, epoch]][self.ID_POS][i] - self.pop[self.r_index2[i, epoch]][self.ID_POS][i])
            else:
                x_new[i] = self.pop[self.r_index1[i, epoch]][self.ID_POS][i]

        # replacement of one electromagnet of generated particle with a random number (only for some generated particles) to bring diversity to the population
        if self.rp[epoch] < self.r_rate:
            x_new[self.RI] = self.problem.lb[self.RI] + (self.problem.ub[self.RI] - self.problem.lb[self.RI]) * self.randomization[epoch]
            RI = self.RI + 1
            if RI >= self.problem.n_dims:
                self.RI = 0

        # checking whether the generated number is inside boundary or not
        pos_new = self.amend_position(x_new, self.problem.lb, self.problem.ub)
        target = self.get_target_wrapper(pos_new)
        # Updating the population if the fitness of the generated particle is better than worst fitness in
        #     the population (because the population is sorted by fitness, the last particle is the worst)
        self.pop[-1] = [pos_new, target]
