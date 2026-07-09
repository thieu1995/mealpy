#!/usr/bin/env python
# Created by "Bora" on 06/01/2026
# Github: https://github.com/Orbadgu
# ---------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalEEFO(Optimizer):
    """
    Electric Eel Foraging Optimization (EEFO)

    Links:
        1. https://doi.org/10.1016/j.eswa.2023.122200
        2. https://www.mathworks.com/matlabcentral/fileexchange/153461-electric-eel-foraging-optimization-eefo

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + epoch (int): Maximum number of iterations, default = 10000
        + pop_size (int): Population size, default = 100

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar
    >>> from mealpy.swarm_based import EEFO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=[-100.] * 30, ub=[100.] * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = EEFO.OriginalEEFO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Wang, L., Zhang, Z., Fan, H., Zhang, J., Mirjalili, S., Khodadadi, N. and Cao, Q., 2024.
    Electric eel foraging optimization: A new bio-inspired optimizer for engineering applications.
    Expert systems with applications, 238, p.122200.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): Maximum number of iterations, default = 10000
            pop_size (int): Population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Amends the solution by replacing any out-of-bound dimension
        with a random uniform value between its lower and upper bounds.

        Args:
            solution: The position array to check and amend.

        Returns:
            The valid solution with out-of-bound dimensions randomized.
        """
        # 1. Create a boolean mask where True indicates a bound violation
        out_of_bounds = (solution < self.problem.lb) | (solution > self.problem.ub)
        # 2. Generate random uniform values for all dimensions based on their lb and ub
        rand_values = self.generator.uniform(self.problem.lb, self.problem.ub, size=solution.shape)
        # 3. Replace out-of-bound elements with random values, keep valid ones intact
        return np.where(out_of_bounds, rand_values, solution)

    def evolve(self, epoch):
        """
        The main evolution process of algorithm.
        """
        it = epoch
        max_it = self.epoch
        dim = self.problem.n_dims
        lb = self.problem.lb
        ub = self.problem.ub

        # Population mean position (required for Eqs. 20, 24, 25)
        pop_pos_matrix = np.array([agent.solution for agent in self.pop])
        mean_pop_pos = np.mean(pop_pos_matrix, axis=0)
        # Eq. (30): Energy factor E0 calculation
        e0 = 4 * np.sin(1 - it / max_it)

        pop_new = []
        for idx in range(self.pop_size):
            agent = self.pop[idx]
            x = agent.solution

            # Eq. (30): Energy factor E calculation
            e_factor = e0 * np.log(1 / self.generator.random())

            # --- Direct Vector Calculation ---
            # Used for determining which dimensions to update in the interaction phase
            if dim == 1:
                direct_vector = np.ones(1)
            else:
                direct_vector = np.zeros(dim)
                rand_num = int(np.ceil((max_it - it) / max_it * self.generator.random() * (dim - 2) + 2))
                rand_dim = self.generator.choice(dim, size=rand_num, replace=False)
                direct_vector[rand_dim] = 1

            # --- PHASE 1: Exploration (Interaction) ---
            # Active when Energy Factor > 1
            if e_factor > 1:
                # Select a random partner 'j' distinct from current agent 'i'
                jdx = self.sample_exclude_one(self.generator, self.pop_size, idx, n_samples=1)
                agent_j = self.pop[jdx]

                # Eq. (7): Interaction based on fitness comparison
                if self.compare_target(agent_j.target, agent.target):
                    if self.generator.random() > 0.5:
                        pos_new = agent_j.solution + self.generator.standard_normal() * direct_vector * (mean_pop_pos - x)
                    else:
                        xr = self.generator.uniform(lb, ub, dim)
                        pos_new = agent_j.solution + 1 * self.generator.standard_normal() * direct_vector * (xr - x)
                else:
                    if self.generator.random() > 0.5:
                        pos_new = x + self.generator.standard_normal() * direct_vector * (mean_pop_pos - agent_j.solution)
                    else:
                        xr = self.generator.uniform(lb, ub, dim)
                        pos_new = x + self.generator.standard_normal() * direct_vector * (xr - agent_j.solution)

            # --- PHASE 2: Exploitation ---
            # Active when Energy Factor <= 1
            else:
                rand_prob = self.generator.random()

                # Mode A: Resting (Eq. 16)
                if rand_prob < 1/3:
                    # Eq. (15): Alpha calculation
                    alpha = 2 * (np.exp(1) - np.exp(it / max_it)) * np.sin(2 * np.pi * self.generator.random())

                    rn = self.generator.integers(0, self.pop_size)
                    rd = self.generator.integers(0, dim)
                    agent_rn = self.pop[rn]

                    # Eq. (12) & (13): Z vector calculation
                    z_scalar = (agent_rn.solution[rd] - lb[rd]) / (ub[rd] - lb[rd])
                    z_vec = lb + z_scalar * (ub - lb)

                    # Eq. (14): Ri calculation (Interaction with global best)
                    r_i = z_vec + alpha * np.abs(z_vec - self.g_best.solution)

                    # Eq. (16): Position update
                    pos_new = r_i + self.generator.standard_normal() * (r_i - np.round(self.generator.random()) * x)

                # Mode B: Migrating (Eq. 24)
                elif rand_prob > 2/3:
                    rn = self.generator.integers(0, self.pop_size)
                    rd = self.generator.integers(0, dim)
                    agent_rn = self.pop[rn]

                    z_scalar = (agent_rn.solution[rd] - lb[rd]) / (ub[rd] - lb[rd])
                    z_vec = lb + z_scalar * (ub - lb)

                    alpha = 2 * (np.exp(1) - np.exp(it / max_it)) * np.sin(2 * np.pi * self.generator.random())
                    r_i = z_vec + alpha * np.abs(z_vec - self.g_best.solution)

                    # Eq. (21): Beta calculation
                    beta = 2 * (np.exp(1) - np.exp(it / max_it)) * np.sin(2 * np.pi * self.generator.random())

                    # Eq. (25): Hr calculation (Hunting area)
                    hr = self.g_best.solution + beta * np.abs(mean_pop_pos - self.g_best.solution)

                    # Eq. (26): Levy flight
                    levy = self.get_levy_flight_step(beta=1.5, multiplier=0.01, size=dim, case=-1)
                    # Eq. (24): Position update
                    pos_new = -self.generator.random() * r_i + self.generator.random() * hr - levy * (hr - x)

                # Mode C: Hunting (Eq. 22)
                else:
                    # Eq. (21): Beta calculation
                    beta = 2 * (np.exp(1) - np.exp(it / max_it)) * np.sin(2 * np.pi * self.generator.random())

                    # Eq. (20): Hprey calculation
                    h_prey = self.g_best.solution + beta * np.abs(mean_pop_pos - self.g_best.solution)

                    r4 = self.generator.random()
                    # Eq. (23): Eta calculation
                    eta = np.exp(r4 * (1 - it) / max_it) * np.cos(2 * np.pi * r4)

                    # Eq. (22): Position update
                    pos_new = h_prey + eta * (h_prey - np.round(self.generator.random()) * x)

            # Check bounds and evaluate
            pos_new = self.correct_solution(pos_new)
            agent_new = self.generate_empty_agent(pos_new)
            pop_new.append(agent_new)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[idx].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

        # --- Greedy Selection Mechanism ---
        # Update population only if the new position provides better fitness
        self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
