# !/usr/bin/env python
# Created by "Thieu" at 22:08, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalSA(Optimizer):
    """
    The original version of: Simulated Annealing (SA)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + max_sub_iter (int): [5, 10, 15], Maximum Number of Sub-Iteration (within fixed temperature), default=5
        + t0 (int): Fixed parameter, Initial Temperature, default=1000
        + t1 (int): Fixed parameter, Final Temperature, default=1
        + move_count (int): [5, 20], Move Count per Individual Solution, default=5
        + mutation_rate (float): [0.01, 0.2], Mutation Rate, default=0.1
        + mutation_step_size (float): [0.05, 0.1, 0.15], Mutation Step Size, default=0.1
        + mutation_step_size_damp (float): [0.8, 0.99], Mutation Step Size Damp, default=0.99

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.SA import OriginalSA
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
    >>> max_sub_iter = 5
    >>> t0 = 1000
    >>> t1 = 1
    >>> move_count = 5
    >>> mutation_rate = 0.1
    >>> mutation_step_size = 0.1
    >>> mutation_step_size_damp = 0.99
    >>> model = OriginalSA(epoch, pop_size, max_sub_iter, t0, t1, move_count, mutation_rate, mutation_step_size, mutation_step_size_damp)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Van Laarhoven, P.J. and Aarts, E.H., 1987. Simulated annealing. In Simulated
    annealing: Theory and applications (pp. 7-15). Springer, Dordrecht.
    """

    def __init__(self, epoch=10000, pop_size=100, max_sub_iter=5, t0=1000, t1=1, move_count=5,
                 mutation_rate=0.1, mutation_step_size=0.1, mutation_step_size_damp=0.99, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            max_sub_iter (int): Maximum Number of Sub-Iteration (within fixed temperature), default=5
            t0 (int): Initial Temperature, default=1000
            t1 (int): Final Temperature, default=1
            move_count (int): Move Count per Individual Solution, default=5
            mutation_rate (float): Mutation Rate, default=0.1
            mutation_step_size (float): Mutation Step Size, default=0.1
            mutation_step_size_damp (float): Mutation Step Size Damp, default=0.99
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.max_sub_iter = self.validator.check_int("max_sub_iter", max_sub_iter, [1, 100000])
        self.t0 = self.validator.check_int("t0", t0, [500, 2000])
        self.t1 = self.validator.check_int("t1", t1, [1, 100])
        self.move_count = self.validator.check_int("move_count", move_count, [2, int(self.pop_size/2)])
        self.mutation_rate = self.validator.check_float("mutation_rate", mutation_rate, (0, 1.0))
        self.mutation_step_size = self.validator.check_float("mutation_step_size", mutation_step_size, (0, 1.0))
        self.mutation_step_size_damp = self.validator.check_float("mutation_step_size_damp", mutation_step_size_damp, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "max_sub_iter", "t0", "t1", "move_count",
                             "mutation_rate", "mutation_step_size", "mutation_step_size_damp"])

        self.nfe_per_epoch = self.pop_size * self.max_sub_iter * self.move_count
        self.sort_flag = True
        self.dyn_t, self.t_damp, self.dyn_sigma = None, None, None

    def mutate__(self, position, sigma):
        # Select Mutating Variables
        pos_new = position + sigma * np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = np.where(np.random.random(self.problem.n_dims) < self.mutation_rate, position, pos_new)
        if np.all(pos_new == position):  # Select at least one variable to mutate
            pos_new[np.random.randint(0, self.problem.n_dims)] = np.random.uniform()
        return self.amend_position(pos_new, self.problem.lb, self.problem.ub)

    def initialization(self):
        # Initial Temperature
        self.dyn_t = self.t0  # Initial Temperature
        self.t_damp = (self.t1 / self.t0) ** (1.0 / self.epoch)  # Calculate Temperature Damp Rate
        self.dyn_sigma = self.mutation_step_size  # Initial Value of Step Size
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Sub-Iterations
        for g in range(0, self.max_sub_iter):

            # Create new population
            pop_new = []
            for idx in range(0, self.pop_size):
                for j in range(0, self.move_count):
                    # Perform Mutation (Move)
                    pos_new = self.mutate__(self.pop[idx][self.ID_POS], self.dyn_sigma)
                    pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                    pop_new.append([pos_new, None])
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
            pop_new = self.update_target_wrapper_population(pop_new)

            # Columnize and Sort Newly Created Population
            pop_new = self.get_sorted_strim_population(pop_new, self.pop_size)

            # Randomized Selection
            for idx in range(0, self.pop_size):
                # Check if new solution is better than current
                if self.compare_agent(pop_new[idx], self.pop[idx]):
                    self.pop[idx] = deepcopy(pop_new[idx])
                else:
                    # Compute difference according to problem type
                    delta = np.abs(pop_new[idx][self.ID_TAR][self.ID_FIT] - self.pop[idx][self.ID_TAR][self.ID_FIT])
                    p = np.exp(-delta / self.dyn_t)  # Compute Acceptance Probability
                    if np.random.uniform() <= p:  # Accept / Reject
                        self.pop[idx] = deepcopy(pop_new[idx])
        # Update Temperature
        self.dyn_t = self.t_damp * self.dyn_t
        self.dyn_sigma = self.mutation_step_size_damp * self.dyn_sigma
