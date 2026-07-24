#!/usr/bin/env python
# Created by "Thieu" at 12:24, 18/07/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo


class OriginalBCO(Optimizer):
    """
    The original version of: Bacterial Colony Optimization (BCO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.
    c_min : float
        Minimum chemotaxis step size, in range (0.0, 1.0). Default is 0.01.
    c_max : float
        Maximum chemotaxis step size, in range (c_min, 10.0). Default is 0.2.
    n_chemotaxis : int
        Nonlinear parameter for chemotaxis step, in range (1, 5). Default is 1.
    max_swim_steps : int
        Maximum swimming steps, in range (2, 10). Default is 4.
    migration_prob : float
        Migration probability, in range (0.0, 1.0). Default is 0.1.


    .. caution::
       1. On average, this algorithm calls the fitness function max_swim_steps*2*pop_size times per epoch, making it extremely slow for large-scale problems.
       2. It has several drawbacks, particularly hardcoded epoch thresholds for reproduction, elimination, and migration operations.

    References
    ~~~~~~~~~~
    1. Niu, B., & Wang, H. (2012). Bacterial colony optimization.
       Discrete dynamics in nature and society, 2012(1), 698057. https://doi.org/10.1155/2012/698057

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BCO
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
    >>> model = BCO.OriginalBCO(epoch=1000, pop_size=50, c_min=0.01, c_max=0.2, n_chemotaxis=2, max_swim_steps=4, migration_prob=0.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Artificial Algae Algorithm", year=2012, difficulty="medium", kind="original")

    def __init__(self, epoch: int = 10000, pop_size: int = 100, c_min: float = 0.01, c_max: float = 0.2,
                 n_chemotaxis: int = 1, max_swim_steps: int = 4, migration_prob: float = 0.1, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c_min = self.validator.check_float("c_min", c_min, (0., 1.0))
        self.c_max = self.validator.check_float("c_max", c_max, (c_min, 10.0))
        self.n_chemotaxis = self.validator.check_int("n_chemotaxis", n_chemotaxis, (1, 5))
        self.max_swim_steps = self.validator.check_int("max_swim_steps", max_swim_steps, (2, 10))
        self.migration_prob = self.validator.check_float("migration_prob", migration_prob, (0., 1.0))
        self.set_parameters(["epoch", "pop_size", "c_min", "c_max", "n_chemotaxis", "max_swim_steps", "migration_prob"])
        self.sort_flag = False
        self.is_parallelizable = False
        self.pop_local = []

    def initialization(self) -> None:
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        # Initialize personal best for each bacterium
        self.pop_local = [agent.copy() for agent in self.pop]

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch: The current iteration
        """
        # Calculate adaptive chemotaxis step size
        step = self.c_min + (self.c_max - self.c_min) * (1 - epoch / self.epoch) ** self.n_chemotaxis

        # 1. Chemotaxis and Communication
        for idx in range(0, self.pop_size):
            f_i = self.generator.choice([0, 1])
            global_direction = self.g_best.solution - self.pop[idx].solution
            personal_direction = self.pop_local[idx].solution - self.pop[idx].solution
            step_direction = f_i * global_direction + (1 - f_i) * personal_direction
            turbulent = self.generator.uniform(-1, 1, self.problem.n_dims)

            # Tumbling phase
            pos_new = self.pop[idx].solution + step * (step_direction + turbulent)
            pos_new = self.correct_solution(pos_new)
            agent_new = self.generate_agent(pos_new)

            # Swimming phase
            m = 0
            while m < self.max_swim_steps and self.compare_target(agent_new.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent_new.copy()
                # Update personal best
                if self.compare_target(agent_new.target, self.pop_local[idx].target, self.problem.minmax):
                    self.pop_local[idx] = agent_new.copy()

                # Continue swimming without turbulence
                pos_new = self.pop[idx].solution + step * step_direction
                agent_new.solution = self.correct_solution(pos_new)
                agent_new.target = self.get_target(pos_new)
                m += 1

        # 2. Interactive Exchange
        for idx in range(0, self.pop_size):
            exchange_type = self.generator.choice(['individual', 'group'])

            if exchange_type == 'individual':
                if self.generator.random() < 0.5:
                    # Dynamic neighbor oriented
                    if idx == 0:
                        neighbor = 1
                    elif idx == self.pop_size - 1:
                        neighbor = self.pop_size - 2
                    else:
                        neighbor = idx + 1 if self.generator.random() < 0.5 else idx - 1
                else:
                    # Random oriented
                    neighbor = self.sample_indexes_exclude_one(self.generator, self.pop_size, exclude_idx=idx, n_samples=1, replace=False)
                # Exchange information if neighbor is better
                if self.compare_target(self.pop[neighbor].target, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx] = self.pop[neighbor].copy()
            else:
                # Group oriented exchange
                if self.compare_target(self.g_best.target, self.pop[idx].target, self.problem.minmax):
                    pos_new = self.pop[idx].solution + 0.1 * (self.g_best.solution - self.pop[idx].solution)
                    pos_new = self.correct_solution(pos_new)
                    agent_new = self.generate_agent(pos_new)
                    if self.compare_target(agent_new.target, self.pop[idx].target, self.problem.minmax):
                        self.pop[idx] = agent_new.copy()

        # 3. Reproduction and Elimination (Every 10 epochs)
        if epoch % 10 == 0:
            self.pop, _ = self.get_sorted_population(self.pop, self.problem.minmax)
            half_pop = self.pop_size // 2

            # The healthier half reproduces, replacing the unhealthier half
            for idx in range(half_pop, self.pop_size):
                self.pop[idx] = self.pop[idx - half_pop].copy()
                self.pop_local[idx] = self.pop_local[idx - half_pop].copy()

        # 4. Migration (Every 50 epochs)
        if epoch % 50 == 0:
            for idx in range(self.pop_size):
                if self.generator.random() < self.migration_prob:
                    pos_new = self.generator.uniform(self.problem.lb, self.problem.ub)
                    agent_new = self.generate_agent(pos_new)
                    self.pop[idx] = agent_new
                    self.pop_local[idx] = agent_new.copy()
