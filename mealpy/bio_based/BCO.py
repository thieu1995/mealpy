#!/usr/bin/env python
# Created by "Thieu" at 12:24, 18/07/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalBCO(Optimizer):
    """
    The original version of: Bacterial Colony Optimization (BCO)

    Links:
        1. https://ieeexplore.ieee.org/abstract/document/4475427

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_m (float): (0, 1) -> better [0.01, 0.2], Mutation probability
        + n_elites (int): (2, pop_size/2) -> better [2, 5], Number of elites will be keep for next generation

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
    >>> model = BCO.OriginalBCO(epoch=1000, pop_size=50, p_m=0.01, n_elites=2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Niu, B., & Wang, H. (2012). Bacterial colony optimization. Discrete dynamics in nature and society, 2012(1), 698057.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, c_min: float = 0.01, c_max: float = 0.2,
                 n_chemotaxis: int = 1, max_swim_steps: int = 4, energy_threshold: float = 0.5,
                 migration_prob: float = 0.1, **kwargs: object) -> None:
        """
        Initialize the algorithm components.

        Args:
            epoch: Maximum number of iterations, default = 10000
            pop_size: Number of population size, default = 100
            c_min: Minimum chemotaxis step size
            c_max: Maximum chemotaxis step size
            n_chemotaxis: Nonlinear parameter for chemotaxis step
            max_swim_steps: Maximum swimming steps
            energy_threshold: Energy threshold for reproduction/elimination
            migration_prob: Migration probability
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c_min = self.validator.check_float("c_min", c_min, (0., 1.0))
        self.c_max = self.validator.check_int("c_max", c_max, (c_min, 10.0))
        self.n_chemotaxis = self.validator.check_int("n_chemotaxis", n_chemotaxis, (1, 5))
        self.max_swim_steps = self.validator.check_int("max_swim_steps", max_swim_steps, (2, 10))
        self.energy_threshold = self.validator.check_float("energy_threshold", energy_threshold, (0, 1.0))
        self.migration_prob = self.validator.check_float("migration_prob", migration_prob, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "c_min", "c_max", "n_chemotaxis", "max_swim_steps", "energy_threshold", "migration_prob"])
        self.sort_flag = False

    def initialize_variables(self):
        self.energy = self.generator.uniform(0, 1, self.pop_size)

    def initialization(self) -> None:
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        self.pop_local = self.pop.copy()

    def get_energy(self, best, worst, fits):
        if best.target.fitness == worst.target.fitness:
            norm_fit = np.zeros(self.pop_size)
        else:
            norm_fit = (fits - worst.target.fitness) / (best.target.fitness - worst.target.fitness)
        # Energy inversely proportional to fitness (lower fitness = higher energy)
        return 1 - norm_fit

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch: The current iteration
        """
        # Normalize fitness to [0, 1]
        _, best, worst = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)
        fits = np.array([agent.target.fitness for agent in self.pop])
        energy = self.get_energy(best[0], worst[0], fits)

        # Calculate adaptive chemotaxis step size
        step = self.c_min + (self.c_max - self.c_min) * (1 - epoch / self.epoch) ** self.n_chemotaxis
        pop = []
        ## Perform chemotaxis and communication
        for idx in range(0, self.pop_size):

            # Random factor for personal vs global best
            f_i = self.generator.random()
            personal_direction = self.pop_local[idx].solution - self.pop[idx].solution
            global_direction = self.g_best.solution - self.pop[idx].solution

            # Tumbling (with random turbulence)
            turbulent = self.generator.normal(0, 0.1, self.problem.n_dims)
            pos_new = f_i * global_direction + (1 - f_i) * personal_direction + turbulent
            for jdx in range(0, self.max_swim_steps):
                # Swimming (no turbulence)
                pos_new = f_i * global_direction + (1 - f_i) * personal_direction
            pos_new = self.pop[idx].solution + step * pos_new
            pos_new = self.correct_solution(pos_new)
            agent_new = self.generate_empty_agent(pos_new)
            pop.append(agent_new)
            if self.mode not in self.AVAILABLE_MODES:
                agent_new.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent_new, minmax=self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)

        ## Perform interactive exchange between bacteria
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
                    neighbor = self.generator.choice(list(set(range(self.pop_size)) - {idx}))

                # Exchange information if neighbor is better
                if self.compare_target(self.pop[neighbor].target, self.pop[idx].target):
                    self.pop[idx] = self.pop[neighbor]
            else:
                # Group exchange
                if self.compare_target(self.pop[idx].target, self.g_best.target):
                    # Move towards global best
                    self.pop[idx].solution += 0.1 * (self.g_best.solution - self.pop[idx].solution)
        self.pop = self.update_target_for_population(self.pop)

        ## Reproduction and elimination
        for idx in range(0, self.pop_size):
            # Sort bacteria by energy level
            sorted_indices = np.argsort(self.energy)[::-1]  # Descending order

            # Reproduction: top 50% reproduce
            n_reproduce = self.pop_size // 2
            reproduction_candidates = sorted_indices[:n_reproduce]

            # Elimination: bottom 25% eliminated
            n_eliminate = self.pop_size // 4
            elimination_candidates = sorted_indices[-n_eliminate:]

            # Replace eliminated bacteria with offspring of good bacteria
            for i in range(min(n_eliminate, n_reproduce)):
                parent_idx = reproduction_candidates[i % n_reproduce]
                child_idx = elimination_candidates[i]

                # Create offspring with small mutation
                mutation = np.random.normal(0, 0.1, self.dim)
                self.positions[child_idx] = self.positions[parent_idx] + mutation

                # Boundary handling
                self.positions[child_idx] = np.clip(self.positions[child_idx],
                                                    self.bounds[0], self.bounds[1])

            def migration(self):
                """Perform migration for some bacteria"""
                # Calculate migration condition based on diversity and stagnation
                position_variance = np.var(self.positions, axis=0)
                diversity = np.mean(position_variance)

                # Migrate if diversity is too low or randomly
                if diversity < 0.01 or np.random.random() < self.migration_prob:
                    n_migrate = max(1, self.pop_size // 10)  # Migrate 10% of population
                    migrate_indices = np.random.choice(self.pop_size, n_migrate, replace=False)

                    for idx in migrate_indices:
                        # Random migration
                        self.positions[idx] = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

            # Probabilistic migration to the i-th position
            pos_new = self.pop[idx].solution.copy()
            for j in range(self.problem.n_dims):
                if self.generator.random() < self.mr[idx]:  # Should we immigrate?
                    # Pick a position from which to emigrate (roulette wheel selection)
                    random_number = self.generator.random() * np.sum(self.mu)
                    select = self.mu[0]
                    select_index = 0
                    while (random_number > select) and (select_index < self.pop_size - 1):
                        select_index += 1
                        select += self.mu[select_index]
                    # this is the migration step
                    pos_new[j] = self.pop[select_index].solution[j]
            noise = self.generator.uniform(self.problem.lb, self.problem.ub)
            condition = self.generator.random(self.problem.n_dims) < self.p_m
            pos_new = np.where(condition, noise, pos_new)
            pos_new = self.correct_solution(pos_new)
            agent_new = self.generate_empty_agent(pos_new)
            pop.append(agent_new)
            if self.mode not in self.AVAILABLE_MODES:
                agent_new.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent_new, minmax=self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)
        # replace the solutions with their new migrated and mutated versions then Merge Populations
        self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_elites, self.pop_size, self.problem.minmax)
