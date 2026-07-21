#!/usr/bin/env python
# -------------------------------------------------------------------------------
# Created By: Sadik on 01/03/2026
# Github: https://github.com/Sadik-Ahmet
# -------------------------------------------------------------------------------
# Updated By: Thieu on 11/07/2026
# Github: https://github.com/thieu1995
# -------------------------------------------------------------------------------

"""
Provides context and a disclaimer regarding the 'Narwhal Optimization Algorithm'.

.. danger::
   There are two distinct papers proposing algorithms under the same name.
   Both have been published in journals with low academic impact; therefore,
   the mathematical soundness and experimental results are highly questionable.
   Users are strongly advised to exercise extreme caution and perform rigorous
   validation before applying these to critical optimization tasks.

The two identified versions are:

1. 'Narwhal Optimizer: A Novel Nature-Inspired Metaheuristic Algorithm' (May 2024)
    - Acronym: NO
    - Note: Lacks significant or novel update operators.
    - DOI: https://doi.org/10.34028/iajit/21/3/6

2. 'Narwhal Optimizer: A Nature-Inspired Optimization Algorithm for Solving Complex Optimization Problems' (September 2025)
    - Acronym: NWOA
    - Note: Performance results reported in the paper may not be replicable or statistically valid.
    - DOI: https://doi.org/10.32604/cmc.2025.066797

Danger
------
   Neither implementation offers a robust contribution to the metaheuristic field.
   It is recommended to utilize established, peer-reviewed optimization frameworks instead.
"""

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalNWOA(Optimizer):
    """
    The original version of: Narwhal Optimization Algorithm (NWOA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.
    amplitude : float
        Wave amplitude, in range [-100.0, 100.0]. Default is 1.0.
    delta_decay : float
        Decay constant, in range (0.0, 1.0). Default is 0.01.
    lamda_decay : float
        Energy decay rate, in range (0.0, 1.0). Default is 0.001.

    References
    ----------
    1. Masadeh, R., Almomani, O., Zaqebah, A., Masadeh, S., Alshqurat, K., Sharieh, A., & Alsharman, N. (2025).
       Narwhal Optimizer: A Nature-Inspired Optimization Algorithm for Solving Complex Optimization Problems.
       Computers, Materials & Continua, 85(2). https://doi.org/10.32604/cmc.2025.066797

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, NWOA
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
    >>> model = NWOA.OriginalNWOA(epoch=1000, pop_size=50, amplitude=2.0, delta_decay=0.01, lamda_decay=0.001)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, amplitude: float = 1.0, delta_decay: float = 0.01,
                 lamda_decay: float = 0.001, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            amplitude (float): Wave amplitude, default = 1.0
            delta_decay (float): Decay constant, default = 0.01
            lamda_decay (float): Energy decay rate, default = 0.001
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.amplitude = self.validator.check_float("amplitude", amplitude, [-100., 100.])
        self.delta_decay = self.validator.check_float("delta_decay", delta_decay, (0, 1))
        self.lamda_decay = self.validator.check_float("lamda_decay", lamda_decay, (0, 1))
        self.set_parameters(["epoch", "pop_size", "amplitude", "delta_decay", "lamda_decay"])
        self.sort_flag = False

    def initialize_variables(self):
        """Initialize algorithm-specific variables"""
        self.prey_energy = 1.0  # Initial prey energy

    @staticmethod
    def cosine_similarity(agent_pos: np.ndarray, best_pos: np.ndarray) -> float:
        """
        Calculate cosine similarity distance (Eq. 3 from paper)
        
        Args:
            agent_pos: Current agent position
            best_pos: Best solution position
            
        Returns:
            float: Cosine similarity distance
        """
        dot_product = np.dot(agent_pos, best_pos)
        norm_agent = np.linalg.norm(agent_pos)
        norm_best = np.linalg.norm(best_pos)
        if norm_agent == 0 or norm_best == 0:
            return 1.0
        return 1 - dot_product / (norm_agent * norm_best)

    def wave_strength(self, agent_pos: np.ndarray, t: int) -> float:
        """
        Calculate wave strength using sonar wave propagation (Eq. 4 from paper)
        
        Args:
            agent_pos: Current agent position
            t: Current iteration
            
        Returns:
            float: Wave strength value
        """
        h_i = self.cosine_similarity(agent_pos, self.g_best.solution)
        strength = self.amplitude * abs(np.sin(2*np.pi * h_i - 2*np.pi * t)) * np.exp(-self.delta_decay * t)
        return strength

    def get_exploration_ratio(self, fitness_improvement: float) -> float:
        """
        Dynamic exploration ratio (Eq. 17 from paper)
        
        Args:
            fitness_improvement: Improvement in fitness from previous iteration

        Returns:
            float: Exploration ratio
        """
        if fitness_improvement < 0.01:
            return 0.7  # Slow improvement, prefer exploration
        elif self.prey_energy < 0.3:
            return 0.3  # Low energy, shift to exploitation
        else:
            return 0.7  # Default: prefer exploration

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Update parameters
        a = 2.0 - (2 * epoch / self.epoch)  # Exploration decay factor (Eq. 16)
        energy = self.prey_energy * np.exp(-self.lamda_decay * epoch)  # (Eq. 8 and 9 from paper)
        self.prey_energy = max(energy, 0)

        # Calculate fitness improvement
        if epoch <= 2:
            fitness_improvement = 1.0
        else:
            fitness_improvement = abs(self.history.list_global_best_fit[-2] - self.history.list_global_best_fit[-1])
        exploration_ratio = self.get_exploration_ratio(fitness_improvement)
        
        pop_new = []
        for idx in range(0, self.pop_size):
            r1 = self.generator.random()
            # Exploration or Exploitation phase
            wave_str = self.wave_strength(self.pop[idx].solution, epoch)

            if r1 < exploration_ratio:
                # Exploration phase (Eq. 5, 6, 7)
                A_exploration = 2 * a * self.generator.random() - a
                pos_new = self.pop[idx].solution + A_exploration * (self.g_best.solution - self.pop[idx].solution) + wave_str * self.generator.random()
            else:
                # Exploitation phase (Eq. 10-15)
                r2 = self.generator.random()
                AA = a * r1 - a
                CC = 2 * r2
                # Calculate suction force
                distance = np.linalg.norm(self.g_best.solution - self.pop[idx].solution)
                suction_force = self.prey_energy * self.prey_energy / (1 + distance)
                pos_new = self.pop[idx].solution - AA * (self.g_best.solution - self.pop[idx].solution) + CC * suction_force * wave_str * self.generator.random()
            
            # Boundary handling
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            
            # Update fitness in single mode
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        
        # Update fitness in parallel modes
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class OriginalNO(Optimizer):
    """
    The original version of: Narwhal Optimization (NO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.
    alpha : float
        Signal intensity control factor, in range [-100.0, 100.0]. Default is 2.0.
    sigma0 : float
        Initial standard deviation for signal propagation, in range [-100.0, 100.0]. Default is 2.0.

    References
    ----------
    1. Medjahed, Seyyid Ahmed, and Fatima Boukhatem. "Narwhal Optimizer: A Novel Nature-Inspired
       Metaheuristic Algorithm". Int. Arab J. Inf. Technol. 21.3 (2024): 418-426. https://doi.org/10.34028/iajit/21/3/6

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, NWOA
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
    >>> model = NWOA.OriginalNO(epoch=1000, pop_size=50, alpha=2.0, sigma0=2.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, alpha=2.0, sigma0=2.0, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): Signal intensity control factor. Default=2.0
            sigma0 (float): Initial standard deviation for signal propagation. Default=2.0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.alpha = self.validator.check_float("alpha", alpha, [-100., 100.])
        self.sigma0 = self.validator.check_float("sigma0", sigma0, [-100., 100.])
        self.set_parameters(["epoch", "pop_size", "alpha", "sigma0"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Update standard deviation sigma linearly over iterations
        sigma_t = self.sigma0 * (1.0 - epoch / self.epoch)

        # Loop through each search agent (narwhal)
        pop_new = []
        for idx in range(self.pop_size):
            # Calculate Euclidean distance between the current narwhal and the prey
            dist = np.linalg.norm(self.pop[idx].solution - self.g_best.solution)
            # Calculate Signal Emission
            SE = 0.1 / (1.0 + self.alpha * dist)

            # Calculate Signal Propagation
            if sigma_t > self.EPSILON:
                PR = np.exp(-(dist ** 2) / (2 * sigma_t ** 2))
            else:
                PR = 0.0
            SP = SE * PR
            # Calculate the step delta
            r1 = self.generator.random()
            beta = r1 - (1.0 / (sigma_t + 1.0))
            delta = beta * np.abs(SP * self.g_best.solution - self.pop[idx].solution)
            # Update the position of the current narwhal
            pos_new = self.pop[idx].solution + delta

            # Apply boundary constraints
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

            # Update fitness in single mode
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)

        # Update fitness in parallel modes
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
