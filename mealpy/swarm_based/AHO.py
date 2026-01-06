#!/usr/bin/env python
import numpy as np
from mealpy.optimizer import Optimizer


class OriginalAHO(Optimizer):
    """
    The original version of: Archerfish Hunting Optimizer (AHO)
    
    Links:
        1. https://doi.org/10.1007/s13369-021-06208-z
        
    Notes:
        1. The algorithm is based on shooting and jumping behaviors of archerfish
        2. Uses ballistic trajectory equations for exploration and exploitation
        3. Lévy flight is used to escape local optima
        
    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + theta (float): [0.1, 1.5] - Swapping angle between exploration and exploitation (default: pi/12)
        + omega (float): [0.01, 10.0] - Attractiveness rate (default: 0.01)
    
    Examples:
        >>> import numpy as np
        >>> from mealpy import FloatVar, AHO
        >>>
        >>> def objective_function(solution):
        >>>     return np.sum(solution**2)
        >>>
        >>> problem = {
        >>>     "obj_func": objective_function,
        >>>     "bounds": FloatVar(lb=[-10., ]*10, ub=[10., ]*10),
        >>>     "minmax": "min",
        >>> }
        >>>
        >>> model = AHO.OriginalAHO(epoch=100, pop_size=50, theta=np.pi/12, omega=0.01)
        >>> g_best = model.solve(problem)
        >>> print(f"Best solution: {g_best.solution}, Best fitness: {g_best.target.fitness}")
    
    References:
        [1] Zitouni, F., Harous, S., Belkeram, A., & Hammou, L. E. B. (2022). 
        The Archerfish Hunting Optimizer: A Novel Metaheuristic Algorithm for Global Optimization. 
        Arabian Journal for Science and Engineering, 47(2), 2513-2553.
    """
    
    def __init__(self, epoch: int = 10000, pop_size: int = 100, theta: float = None, omega: float = 0.01, **kwargs):
        """
        Args:
            epoch (int): Maximum number of iterations, default = 10000
            pop_size (int): Number of population size, default = 100
            theta (float): Swapping angle between exploration and exploitation, default = pi/12
            omega (float): Attractiveness rate, default = 0.01
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.theta = self.validator.check_float("theta", theta if theta is not None else np.pi/12, [0.01, np.pi])
        self.omega = self.validator.check_float("omega", omega, [0.001, 100.0])
        self.set_parameters(["epoch", "pop_size", "theta", "omega"])
        self.sort_flag = False
    
    def initialize_variables(self):
        """Initialize algorithm-specific variables"""
        self.no_improvement_count = np.zeros(self.pop_size, dtype=int)
        # Threshold for triggering Lévy flight (d × N as per paper)
        self.levy_threshold = self.problem.n_dims * self.pop_size
    
    def levy_flight(self, solution):
        """
        Generate new position using Lévy flight (Equations 7 and 8 from paper)
        
        Args:
            solution (np.ndarray): Current solution position
            
        Returns:
            np.ndarray: New position after Lévy flight
        """
        beta = 1.5
        
        # Calculate sigma (Equation 8)
        numerator = np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        denominator = np.math.gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)
        sigma = np.power(numerator / denominator, 1 / beta)
        
        # Generate u and v from normal distributions
        u = self.generator.normal(0, sigma, self.problem.n_dims)
        v = self.generator.normal(0, 1, self.problem.n_dims)
        
        # Lévy step
        step = u / np.power(np.abs(v), 1 / beta)
        
        # Generate alpha uniformly
        alpha = self.generator.uniform(0, 1)
        
        # New position (Equation 7)
        new_pos = solution + alpha * step
        
        return new_pos
    
    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class
        
        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        
        for idx in range(self.pop_size):
            # Generate perceiving angle theta_0 (Equation 6)
            b = self.generator.integers(0, 2)  # Bernoulli distribution (0 or 1)
            alpha_random = self.generator.uniform(0, 1)
            theta_0 = np.power(-1, b) * alpha_random * np.pi
            
            # Determine exploration or exploitation based on theta_0
            abs_theta_0 = np.abs(theta_0)
            
            # Shooting behavior (Exploration) - Equation 2 and 3
            if (abs_theta_0 > 0 and abs_theta_0 < self.theta) or \
               (abs_theta_0 > np.pi - self.theta and abs_theta_0 < np.pi):
                
                # Select random archerfish k
                k = self.generator.integers(0, self.pop_size)
                
                # Compute prey location X_prey (Equation 3)
                prey_pos = self.pop[k].solution.copy()
                
                # Add refraction effects (epsilon) and shooting distance
                random_dim = self.generator.integers(0, self.problem.n_dims)
                epsilon = self.generator.uniform(-0.5, 0.5, self.problem.n_dims)
                
                prey_pos[random_dim] += self.omega * np.sin(2 * theta_0)
                prey_pos = prey_pos + epsilon
                
                # Clip to bounds
                prey_pos = self.correct_solution(prey_pos)
                prey_fit = self.get_target(prey_pos)
                
                # Update current archerfish position if prey is better (Equation 2)
                distance = np.linalg.norm(prey_pos - self.pop[idx].solution)
                attractiveness = np.exp(-distance**2)
                
                new_pos = self.pop[idx].solution + attractiveness * (prey_pos - self.pop[idx].solution)
                new_pos = self.correct_solution(new_pos)
                
                # Check if position improved
                if self.compare_target(prey_fit, self.pop[idx].target, self.problem.minmax):
                    agent = self.generate_agent(new_pos)
                    self.no_improvement_count[idx] = 0
                else:
                    # Check for Lévy flight
                    self.no_improvement_count[idx] += 1
                    if self.no_improvement_count[idx] >= self.levy_threshold:
                        levy_pos = self.levy_flight(self.pop[idx].solution)
                        levy_pos = self.correct_solution(levy_pos)
                        agent = self.generate_agent(levy_pos)
                        self.no_improvement_count[idx] = 0
                    else:
                        agent = self.pop[idx].copy()
            
            # Jumping behavior (Exploitation) - Equation 4 and 5
            else:
                # Compute prey location X_prey (Equation 5)
                prey_pos = self.pop[idx].solution.copy()
                
                # Add two random dimensions with shooting distance
                dims = self.generator.choice(self.problem.n_dims, size=min(2, self.problem.n_dims), replace=False)
                epsilon = self.generator.uniform(-0.5, 0.5, self.problem.n_dims)
                
                prey_pos[dims[0]] += self.omega * np.sin(2 * theta_0)
                if len(dims) > 1:
                    prey_pos[dims[1]] += self.omega * np.sin(theta_0)**2
                prey_pos = prey_pos + epsilon
                
                # Clip to bounds
                prey_pos = self.correct_solution(prey_pos)
                
                # Update position (Equation 4)
                distance = np.linalg.norm(prey_pos - self.pop[idx].solution)
                attractiveness = np.exp(-distance**2)
                
                new_pos = self.pop[idx].solution + attractiveness * (prey_pos - self.pop[idx].solution)
                new_pos = self.correct_solution(new_pos)
                agent = self.generate_agent(new_pos)
                
                # Check if position improved
                if not self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                    self.no_improvement_count[idx] += 1
                    if self.no_improvement_count[idx] >= self.levy_threshold:
                        levy_pos = self.levy_flight(self.pop[idx].solution)
                        levy_pos = self.correct_solution(levy_pos)
                        agent = self.generate_agent(levy_pos)
                        self.no_improvement_count[idx] = 0
                else:
                    self.no_improvement_count[idx] = 0
            
            pop_new.append(agent)
        
        self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)