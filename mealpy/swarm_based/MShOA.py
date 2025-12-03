#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class MShOA(Optimizer):
    """
    The original version of: Mantis Shrimp Optimization Algorithm (MShOA)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + polarization_rate (float): [0.1, 0.9], Controls switching between navigation and strike, default = 0.5
        + strike_factor (float): [1.0, 3.0], Controls the speed of convergence, default = 2.0
        + defense_factor (float): [0.05, 0.2], Controls the diversity around the best solution, default = 0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, MShOA
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
    >>> model = MShOA(epoch=1000, pop_size=50, polarization_rate=0.5, strike_factor=2.0, defense_factor=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Thieu, N. and Pan, J.S., 2025. Mantis Shrimp Optimization Algorithm: A Novel Nature-Inspired Metaheuristic Algorithm.
    Mathematics, 13(1), p.[page]. MDPI.
    
    .. code-block:: bibtex
    
        @article{mshoa2025,
            title={Mantis Shrimp Optimization Algorithm: A Novel Nature-Inspired Metaheuristic Algorithm},
            author={Thieu, Nguyen and Pan, Jeng-Shyang},
            journal={Mathematics},
            volume={13},
            number={1},
            year={2025},
            publisher={MDPI}
        }
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, polarization_rate: float = 0.5,
                 strike_factor: float = 2.0, defense_factor: float = 0.1, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            polarization_rate: Controls switching between navigation and strike, default = 0.5
            strike_factor: Controls the speed of convergence, default = 2.0
            defense_factor: Controls the diversity around the best solution, default = 0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.polarization_rate = self.validator.check_float("polarization_rate", polarization_rate, [0.0, 1.0])
        self.strike_factor = self.validator.check_float("strike_factor", strike_factor, [0.0, 5.0])
        self.defense_factor = self.validator.check_float("defense_factor", defense_factor, [0.0, 1.0])
        self.set_parameters(["epoch", "pop_size", "polarization_rate", "strike_factor", "defense_factor"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Extract all positions into a matrix (pop_size, n_dims)
        pop_pos = np.array([agent.solution for agent in self.pop])
        
        # Generate all random numbers in bulk for vectorization
        r1 = self.generator.random(self.pop_size)  # Shape: (pop_size,)
        r2 = self.generator.random(self.pop_size)  # Shape: (pop_size,)
        r3 = self.generator.random(self.pop_size)  # Shape: (pop_size,)
        
        # Generate random indices for navigation phase
        random_indices = self.generator.integers(0, self.pop_size, self.pop_size)  # Shape: (pop_size,)
        
        # Create boolean masks for vectorized conditional logic
        mask_navigation = r1 < self.polarization_rate  # Shape: (pop_size,)
        mask_defense = r3 < 0.1  # Shape: (pop_size,)
        
        # Expand masks to match dimensions: (pop_size, 1) -> (pop_size, n_dims)
        mask_nav_expanded = mask_navigation[:, np.newaxis]  # Shape: (pop_size, 1)
        mask_def_expanded = mask_defense[:, np.newaxis]  # Shape: (pop_size, 1)
        
        # Expand r2 to match dimensions: (pop_size,) -> (pop_size, n_dims)
        r2_expanded = r2[:, np.newaxis]  # Shape: (pop_size, 1)
        
        # Phase 1: Navigation (Exploration) - Equation 1
        # pos_new = pop[idx].solution + r2 * (pop[random_idx].solution - pop[idx].solution)
        random_pop_pos = pop_pos[random_indices]  # Shape: (pop_size, n_dims)
        navigation_pos = pop_pos + r2_expanded * (random_pop_pos - pop_pos)  # Shape: (pop_size, n_dims)
        
        # Phase 2: Raptorial Strike (Exploitation) - Equation 2
        # pos_new = g_best.solution + strike_factor * (g_best.solution - pop[idx].solution)
        g_best_pos = self.g_best.solution  # Shape: (n_dims,)
        strike_pos = g_best_pos + self.strike_factor * (g_best_pos - pop_pos)  # Shape: (pop_size, n_dims)
        
        # Combine Phase 1 and Phase 2 using np.where
        # If mask_navigation is True, use navigation_pos; else use strike_pos
        pos_new = np.where(mask_nav_expanded, navigation_pos, strike_pos)  # Shape: (pop_size, n_dims)
        
        # Phase 3: Defense Mechanism (Stagnation Avoidance)
        # pos_new = g_best.solution + defense_factor * random_uniform(lb, ub)
        defense_random = self.generator.uniform(self.problem.lb, self.problem.ub, size=(self.pop_size, self.problem.n_dims))  # Shape: (pop_size, n_dims)
        defense_pos = g_best_pos + self.defense_factor * defense_random  # Shape: (pop_size, n_dims)
        
        # Apply defense mechanism where mask_defense is True
        pos_new = np.where(mask_def_expanded, defense_pos, pos_new)  # Shape: (pop_size, n_dims)
        
        # Create new agents and properly evaluate targets
        pop_new = []
        for idx in range(self.pop_size):
            # Boundary check: correct solution
            pos_corrected = self.correct_solution(pos_new[idx])
            
            # Create agent with corrected position
            agent = self.generate_empty_agent(pos_corrected)
            
            # CRITICAL: Explicitly set target to ensure no None targets
            if self.mode not in self.AVAILABLE_MODES:
                # Single mode: evaluate target immediately
                agent.target = self.get_target(pos_corrected)
                # Update population immediately in single mode
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
            else:
                # Parallel/swarm mode: add to list for batch evaluation
                pop_new.append(agent)
        
        # For parallel/swarm modes: update all targets at once, then perform greedy selection
        if self.mode in self.AVAILABLE_MODES:
            # Ensure all agents have targets before greedy selection
            pop_new = self.update_target_for_population(pop_new)
            # Verify all agents have targets (safety check)
            for agent in pop_new:
                if agent.target is None:
                    agent.target = self.get_target(agent.solution)
            # Perform greedy selection
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
