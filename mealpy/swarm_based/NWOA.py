#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------.
# Created By: Sadik (adapted from original NWOA implementation)
# Created Date: 2026-01-03
# Version: 1.0.0
# ---------------------------------------------------------------------------
# "Narwhal Optimization Algorithm (NWOA)"
# ---------------------------------------------------------------------------

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalNWOA(Optimizer):
    """
    The original version of: Narwhal Optimization Algorithm (NWOA)

    Links:
        1. https://doi.org/10.32604/cmc.2025.066797

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + A (float): Wave amplitude, default = 1.0
        + k (float): Wave number, default = 2π
        + omega (float): Angular frequency, default = 2π
        + delta (float): Decay constant, default = 0.01
        + lambda_decay (float): Energy decay rate, default = 0.001

    Examples
    ~~~~~~~~
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
    >>> model = NWOA.OriginalNWOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Masadeh, R., Almomani, O., Zaqebah, A., Masadeh, S., Alshqurat, K., Sharieh, A., & Alsharman, N. (2025).
        Narwhal Optimizer: A Nature-Inspired Optimization Algorithm for Solving Complex Optimization Problems.
        Computers, Materials & Continua, 85(2). https://doi.org/10.32604/cmc.2025.066797
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, 
                 A: float = 1.0, k: float = 2*np.pi, omega: float = 2*np.pi,
                 delta: float = 0.01, lambda_decay: float = 0.001, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            A (float): Wave amplitude, default = 1.0
            k (float): Wave number, default = 2π
            omega (float): Angular frequency, default = 2π
            delta (float): Decay constant, default = 0.01
            lambda_decay (float): Energy decay rate, default = 0.001
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.A = self.validator.check_float("A", A, (0, 10))
        self.k = self.validator.check_float("k", k, (0, 20))
        self.omega = self.validator.check_float("omega", omega, (0, 20))
        self.delta = self.validator.check_float("delta", delta, (0, 1))
        self.lambda_decay = self.validator.check_float("lambda_decay", lambda_decay, (0, 1))
        self.set_parameters(["epoch", "pop_size", "A", "k", "omega", "delta", "lambda_decay"])
        self.sort_flag = False

    def initialize_variables(self):
        """Initialize algorithm-specific variables"""
        self.prey_energy = 1.0  # Initial prey energy

    def cosine_similarity(self, agent_pos: np.ndarray, best_pos: np.ndarray) -> float:
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
        
        similarity = dot_product / (norm_agent * norm_best)
        return 1 - similarity

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
        strength = self.A * abs(np.sin(self.k * h_i - self.omega * t)) * np.exp(-self.delta * t)
        return strength

    def update_prey_energy(self, t: int) -> float:
        """
        Update prey energy with exponential decay (Eq. 8 and 9 from paper)
        
        Args:
            t: Current iteration
            
        Returns:
            float: Updated prey energy
        """
        energy = self.prey_energy * np.exp(-self.lambda_decay * t)
        return max(energy, 0)

    def get_exploration_ratio(self, t: int, fitness_improvement: float) -> float:
        """
        Dynamic exploration ratio (Eq. 17 from paper)
        
        Args:
            t: Current iteration
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
        a = 2 - (2 * epoch / self.epoch)  # Exploration decay factor (Eq. 16)
        self.prey_energy = self.update_prey_energy(epoch)
        
        # Calculate fitness improvement
        if epoch > 0:
            fitness_improvement = abs(self.history.list_global_best_fit[-2] - self.history.list_global_best_fit[-1]) if len(self.history.list_global_best_fit) > 1 else 1.0
        else:
            fitness_improvement = 1.0
        
        exploration_ratio = self.get_exploration_ratio(epoch, fitness_improvement)
        
        pop_new = []
        for idx in range(0, self.pop_size):
            r1 = self.generator.random()
            
            # Exploration or Exploitation phase
            if r1 < exploration_ratio:
                # Exploration phase (Eq. 5, 6, 7)
                A_exploration = 2 * a * self.generator.random() - a
                wave_str = self.wave_strength(self.pop[idx].solution, epoch)
                
                pos_new = (self.pop[idx].solution + 
                          A_exploration * (self.g_best.solution - self.pop[idx].solution) + 
                          wave_str * self.generator.random(size=self.problem.n_dims))
            else:
                # Exploitation phase (Eq. 10-15)
                r2 = self.generator.random()
                A_exploitation = a * r1 - a
                C_exploitation = 2 * r2
                
                # Calculate suction force
                distance = np.linalg.norm(self.g_best.solution - self.pop[idx].solution)
                suction_strength = self.prey_energy / (1 + distance)
                suction_force = suction_strength * self.prey_energy
                
                wave_str = self.wave_strength(self.pop[idx].solution, epoch)
                
                pos_new = (self.pop[idx].solution - 
                          A_exploitation * (self.g_best.solution - self.pop[idx].solution) + 
                          C_exploitation * suction_force * wave_str * self.generator.random(size=self.problem.n_dims))
            
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
