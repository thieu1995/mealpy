#!/usr/bin/env python
# Created by "Thieu" at 11:01, 15/08/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class DevSMO(Optimizer):
    """
    The developed version of: Spider Monkey Optimization (SMO)

    Notes:
        + The original paper is truly difficult to read and unclear. The operators are somewhat more understandable,
        but the pseudo-code they provide is inaccurate. In addition, the design of the two parameters —
        `local_leader_limit` and `global_leader_limit` — is essentially meaningless. After each iteration,
        the population can be split and separated continuously, making it very unlikely for the if conditions
        involving these two values to ever be triggered. As a result, the operators in the two phases
        local_leader_decision and global_leader_decision will rarely be applied.

        + In summary, this algorithm has many issues, and the original MATLAB source code is also unavailable.
        I cannot guarantee its correctness, so I will refer to it as DevSMO.

    Links:
        1. https://doi.org/10.1007/s12293-013-0128-0

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SMO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = SMO.DevSMO(epoch=1000, pop_size=50, max_groups = 5, perturbation_rate = 0.7)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Bansal, J. C., Sharma, H., Jadon, S. S., & Clerc, M. (2014).
    Spider monkey optimization algorithm for numerical optimization. Memetic computing, 6(1), 31-47.
    """

    def __init__(self, epoch=10000, pop_size=100, max_groups: int = 5, perturbation_rate: float = 0.7, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            max_groups (int): Maximum number of groups for spider monkeys, default = 5
            perturbation_rate (float): Perturbation rate for spider monkeys, default = 0.7
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.max_groups = self.validator.check_int("max_groups", max_groups, [2, 100])
        self.perturbation_rate = self.validator.check_float("perturbation_rate", perturbation_rate, [0., 1.0])
        self.set_parameters(["epoch", "pop_size", "max_groups", "perturbation_rate"])
        self.sort_flag = False
        self.is_parallelizable = False

    def split_fill_by_group(self, pop, n_groups):
        """
        Chia theo kiểu lấp đầy từng group: group 0 trước, rồi group 1, ...
        k: số group.
        """
        n = len(pop)
        gsize = -(-n // n_groups)  # ceil(n/k)
        # Cắt lát theo chỉ số group i
        return [pop[i * gsize:(i + 1) * gsize] for i in range(n_groups) if i * gsize < n]

    def merge_groups(self, groups):
        return [x for g in groups for x in g]

    def initialize_variables(self):
        # Set default parameters as per paper
        max_possible_groups = self.pop_size // 3
        self.num_groups = min(self.max_groups, max_possible_groups)
        self.group_size = -(-self.pop_size // self.num_groups)
        self.LLL = self.epoch // 10        # local_leader_limit
        self.GLL = self.epoch // 20      # global_leader_limit

        # Counters
        self.local_limit_counts = [0] * self.num_groups
        self.global_limit_count = 0

    def initialization(self) -> None:
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        # Split groups
        self.groups = self.split_fill_by_group(self.pop, self.num_groups)
        # Get local leaders
        self.local_leaders = [self.get_sorted_population(group, self.problem.minmax, return_index=False)[0] for group in self.groups]

    def local_leader_phase(self):
        """Local Leader Phase - all monkeys update based on local leader"""
        for group_idx, group in enumerate(self.groups):
            n_items = len(group)
            for idx in range(n_items):
                list_rand = self.generator.choice(list(set(range(n_items)) - {idx}), size=self.problem.n_dims, replace=True)
                vector = np.array([group[jdx].solution[kdx] for kdx, jdx in enumerate(list_rand)])
                pos_new = group[idx].solution + self.generator.uniform(0, 1, self.problem.n_dims) * \
                          (self.local_leaders[group_idx].solution - group[idx].solution) + \
                          self.generator.uniform(-1, 1, self.problem.n_dims) * (vector - group[idx].solution)
                pos_new = np.where(self.generator.uniform(0, 1, self.problem.n_dims) >= self.perturbation_rate, pos_new, group[idx].solution)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(agent.target, group[idx].target, self.problem.minmax):
                    self.groups[group_idx][idx] = agent
        self.pop = self.merge_groups(self.groups)

    def global_leader_phase(self):
        """Global Leader Phase - selected monkeys update based on global leader"""
        # Calculate selection probabilities
        list_fits = np.array([agent.target.fitness for group in self.groups for agent in group])
        # Calculate fitness using formula from paper
        fitness = np.where(list_fits >= 0, 1 / (1 + list_fits), 1 + np.abs(list_fits))

        max_fitness = np.max(fitness)
        prob = 0.9 * (fitness / max_fitness) + 0.1

        for group_idx, group in enumerate(self.groups):
            attempts = 0
            while attempts < self.group_size:
                for idx, agent in enumerate(group):
                    if attempts >= self.group_size:
                        break
                    prob_idx = idx + group_idx * self.group_size
                    if self.generator.uniform(0, 1) < prob[prob_idx]:
                        attempts += 1
                        # Randomly select dimension to update
                        k = self.generator.integers(0, self.problem.n_dims)
                        # Select random monkey
                        jdx = self.generator.choice(list(set(range(self.group_size)) - {idx}))
                        pos_new = agent.solution.copy()
                        # Update using equation (4)
                        pos_new[k] = pos_new[k] + self.generator.uniform(0, 1) * (self.g_best.solution[k] - pos_new[k]) +\
                                    self.generator.uniform(-1, 1) * (group[jdx].solution[k] - pos_new[k])
                        # Apply bounds
                        pos_new = self.correct_solution(pos_new)
                        agent = self.generate_agent(pos_new)
                        # Greedy selection
                        if self.compare_target(agent.target, self.g_best.target, self.problem.minmax):
                            self.groups[group_idx][idx] = agent

    def local_leader_decision_phase(self):
        """Local Leader Decision Phase - handle stagnated local leaders"""
        local_leaders_new = [self.get_sorted_population(group, self.problem.minmax, return_index=False)[0] for group in self.groups]
        for group_idx, group in enumerate(self.groups):
            # Update local limit count
            if self.compare_target(self.local_leaders[group_idx].target, local_leaders_new[group_idx].target, self.problem.minmax):
                self.local_limit_counts[group_idx] += 1
            else:
                self.local_limit_counts[group_idx] = 0
                self.local_leaders[group_idx] = local_leaders_new[group_idx]

            # If local leader is stagnated
            if self.local_limit_counts[group_idx] > self.LLL:
                self.local_limit_counts[group_idx] = 0

                for idx, agent in enumerate(group):
                    # Random initialization
                    pos_new_01 = self.generator.uniform(self.problem.lb, self.problem.ub, self.problem.n_dims)
                    # Update using equation (5)
                    pos_new_02 = agent.solution + self.generator.uniform(0, 1, self.problem.n_dims) * (self.g_best.solution - agent.solution) + \
                                  self.generator.uniform(0, 1, self.problem.n_dims) * (agent.solution - self.local_leaders[group_idx].solution)
                    pos_new = np.where(self.generator.uniform(0, 1, self.problem.n_dims) >= self.perturbation_rate, pos_new_01, pos_new_02)
                    # Apply bounds
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_agent(pos_new)
                    # Always accept new position in this phase
                    self.groups[group_idx][idx] = agent

    def global_leader_decision_phase(self):
        """Global Leader Decision Phase - handle fission-fusion"""
        if self.global_limit_count > self.GLL:
            self.global_limit_count = 0
            if self.generator.random() < 0.5:
                # Fission - divide groups
                self.pop = self.merge_groups(self.groups)
                self.rng.shuffle(self.pop)
                self.groups = self.split_fill_by_group(self.pop, self.num_groups)
            else:
                # Fusion - combine all groups
                self.pop = self.merge_groups(self.groups)
            # Update local leaders after fission/fusion
            self.local_limit_counts = [0] * self.num_groups
            self.local_leaders = [self.get_sorted_population(group, self.problem.minmax, return_index=False)[0] for group in self.groups]

    def update_leaders(self):
        self.pop = self.merge_groups(self.groups)
        # Update global leader
        g_best_current = self.get_sorted_population(self.pop, self.problem.minmax, return_index=False)[0]
        if self.compare_target(g_best_current.target, self.g_best.target, self.problem.minmax):
            self.g_best = g_best_current
            # Update global limit count
            self.global_limit_count = 0
        else:
            self.global_limit_count += 1

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """


        self.local_leader_phase()
        self.global_leader_phase()
        self.update_leaders()
        self.local_leader_decision_phase()
        self.global_leader_decision_phase()
