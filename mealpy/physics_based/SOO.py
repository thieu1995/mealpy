#!/usr/bin/env python
# Created by "Thieu" at 22:08, 28/08/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSOO(Optimizer):
    """
    The original version of: Stellar Oscillation Optimizer (SOO)

    Notes:
        + The MATLAB code in the link below by the author is completely different from the pseudocode in
        the original paper. I don’t understand how the author could write such an incorrect implementation
        and still obtain good results. There are only two possibilities: either the author fabricated the
        results in the paper, or the paper itself is fundamentally flawed.

        + For example, you can see equation number 8 — it involves taking the average of two new positions.
        However, in the code, it is incorrectly implemented as position 1 plus half of position 2.
        Even more concerning is that the pseudocode in the paper is completely different from the actual code.
        The MATLAB coding quality is really poor. In the pseudocode, it states that the fitness should be
        calculated and the global best as well as the top 3 best should be updated — yet this is entirely missing in the code.

        + Therefore, I do not recommend users to use this algorithm, as it lacks integrity between the
        results in the paper and the actual experimental implementation.

    Links:
        1. https://mathworks.com/matlabcentral/fileexchange/161921-stellar-oscillation-optimizer-meta-heuristic-optimimization

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SOO
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
    >>> model = SOO.OriginalSOO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rodan, A., Al-Tamimi, A. K., Al-Alnemer, L., & Mirjalili, S. (2025).
    Stellar oscillation optimizer: a nature-inspired metaheuristic optimization algorithm. Cluster Computing, 28(6), 362.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def initialize_variables(self):
        self.initial_period = 3

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Update period and angular frequency
        caf = 2 * np.pi / (self.initial_period + 0.001 * epoch)

        # Update scaling factor
        scaler = 2 * (1.0 -  epoch / self.epoch)

        # Update positions of star oscillators
        pop_new = []
        for idx in range(self.pop_size):
            r1 = self.generator.random(size=self.problem.n_dims)
            r2 = self.generator.random(size=self.problem.n_dims)
            r3 = self.generator.random(size=self.problem.n_dims)

            # Calculate oscillation positions
            osc1 = scaler * (caf * r1 - 1) *  (self.pop[idx].solution - np.abs(r1 * np.sin(r2) * np.abs(r3 * self.g_best.solution)))
            osc1_pos = self.g_best.solution - r1 * r3 * osc1
            osc2 = scaler * (caf * r1 - 1) * (self.pop[idx].solution - np.abs(r1 * np.cos(r2) * np.abs(r3 * self.g_best.solution)))
            osc2_pos = self.g_best.solution - r2 * r3 * osc2
            pos_new = r3 * (osc1_pos + osc2_pos) / 2
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        # Get top 3 stars
        _, best3, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)

        # Perform oscillatory movement update
        for idx in range(self.pop_size):
            # Average of top star positions
            avg3 = np.mean([agent.solution for agent in best3], axis=0)

            # Select 3 random indices different from current
            r1, r2, r3 = self.generator.choice(list(set(range(self.pop_size)) - {idx}), size=3, replace=False)

            # Generate new position based on oscillatory movement
            rf = self.generator.random()
            pos_new = avg3 + 0.5 * (np.sin(rf * np.pi) * (self.pop[r1].solution - self.pop[r2].solution) +
                                    np.cos((1 - rf) * np.pi) * (self.pop[r1].solution - self.pop[r3].solution))
            ## Probabilistic update
            pos_new = np.where(self.generator.random(size=self.problem.n_dims) <= 0.5, pos_new, self.pop[idx].solution)
            # Apply boundary constraints
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
