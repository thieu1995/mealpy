#!/usr/bin/env python
#       Created by "Işıl Ada Yiğit"                ----------%
#       Email: adayigit80@gmail.com                          %
#       Github: https://github.com/isil-ada                  %
# -----------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer

class OriginalKLA(Optimizer):
    """
    The robust version of: Kirchhoff's Law Algorithm (KLA)

    Links:
        1. https://www.researchgate.net/publication/394028477_Kirchhoff's_law_algorithm_KLA_a_novel_physics-inspired_non-parametric_metaheuristic_algorithm_for_optimization_problems
        2. https://www.mathworks.com/matlabcentral/fileexchange/181589-kirchhoff-s-law-algorithm-kla

    Hyper-parameters should be fine-tuned in approximate ranges to obtain
    faster convergence toward the global optimum:
        + epoch (int): maximum number of iterations, default = 10000
        + pop_size (int): population size, default = 100

    Recent Improvements (Custom Implementation):
        1. Step Limiter (Fuse): Prevents moving more than 10% of the search space 
           in a single iteration using 'np.clip'. (Solves F1/F2 explosion issues).
        2. Adaptive Decay: Damping factor reduces step size over time.
        3. Hybrid Topology: 50% Random Neighbor, 50% Global Best (Leader) tracking.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar
    >>> from your_module import OriginalKLA
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
    >>> model = OriginalKLA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, "
    >>>       f"Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kirchhoff's law algorithm (KLA): a novel physics-inspired non-parametric 
        metaheuristic algorithm for optimization problems. ResearchGate.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): population size, default = 100
            **kwargs: object
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of the algorithm.
        Inherited from Optimizer class.

        Args:
            epoch (int): The current iteration.
        """
        # Zero division prevention
        EPSILON = 1e-15 
        
        # --- Damping Factor (w) ---
        # Starts at 2.0 (Exploration), Ends at 0.0 (Exploitation)
        w = 2.0 * (1 - (epoch / self.epoch))

        pop_new = []
        
        # Domain Range calculation
        # Used for the Current Limiter (Fuse) mechanism
        # In Mealpy, self.problem.ub and lb can be vectors, so we take the difference.
        domain_range = self.problem.ub - self.problem.lb

        for idx in range(0, self.pop_size):
            agent = self.pop[idx]
            
            # --- 1. Topology Selection ---
            available_indices = list(set(range(0, self.pop_size)) - {idx})
            
            # Strategy: 50% chance to involve the Leader (g_best)
            if self.generator.random() < 0.5:
                # Two random agents, one leader (Superconductor Mode)
                agent_a = self.pop[self.generator.choice(available_indices)]
                agent_b = self.pop[self.generator.choice(available_indices)]
                agent_c = self.g_best 
            else:
                # All three agents are random (Standard KLA)
                neighbor_indices = self.generator.choice(available_indices, 3, replace=False)
                agent_a = self.pop[neighbor_indices[0]]
                agent_b = self.pop[neighbor_indices[1]]
                agent_c = self.pop[neighbor_indices[2]]
            
            # --- 2. Current Calculation (KCL) ---
            f_i = agent.target.fitness
            
            # Inner Function: Current Calculator
            def calculate_current(target_agent):
                f_target = target_agent.target.fitness
                
                # Ratio Clipping: Do not allow ratio to exceed 50.0 (Prevent numerical explosion)
                ratio = f_i / (f_target + EPSILON)
                if ratio > 50.0: ratio = 50.0
                
                r_rand = self.generator.random()
                
                # I = V * Conductance
                # V = (X_target - X_current)
                current = (target_agent.solution - agent.solution) * (ratio ** (2 * r_rand))
                return current

            current_a = calculate_current(agent_a)
            current_b = calculate_current(agent_b)
            current_c = calculate_current(agent_c)
            
            # --- 3. Superposition (Total Current) ---
            total_current = current_a + current_b + current_c
            
            # --- 4. Position Update and FUSE (Step Limiter) ---
            
            # Raw step size
            step = self.generator.random() * total_current * (w + 0.1)
            
            # !!! CRITICAL IMPROVEMENT !!!
            # Limit the step: Do not move further than 10% of the map in one round.
            # This prevents the huge deviations observed in F1 and F2.
            max_step_size = 0.1 * domain_range 
            step = np.clip(step, -max_step_size, max_step_size)
            
            pos_new = agent.solution + step
            
            # --- 5. Boundary Check and Recording ---
            pos_new = self.correct_solution(pos_new)
            agent_new = self.generate_agent(pos_new)
            pop_new.append(agent_new)
            
        # --- 6. Elitism ---
        self.pop = self.get_sorted_and_trimmed_population(
            self.pop + pop_new, 
            self.pop_size, 
            self.problem.minmax
        )