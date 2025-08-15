#!/usr/bin/env python
# Created by "Thieu" at 19:18, 15/08/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSquirrelSA(Optimizer):
    """
    The original version of: Squirrel Search Algorithm (SquirrelSA)

    Notes:
        + https://doi.org/10.1016/j.swevo.2018.02.013

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SquirrelSA
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
    >>> model = SquirrelSA.OriginalSquirrelSA(epoch=1000, pop_size=50, n_food_sources=4,
    >>>         predator_prob=0.1, gliding_constant=1.9, scaling_factor=18, beta=1.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Jain, M., Singh, V., & Rani, A. (2019). A novel nature-inspired algorithm for optimization: Squirrel search algorithm.
    Swarm and evolutionary computation, 44, 148-175.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_food_sources=4, predator_prob=0.1,
                 gliding_constant=1.9, scaling_factor=18, beta=1.5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_food_sources (int): number of food sources (1 hickory + 3 acorn trees), default = 4
            predator_prob (float): predator presence probability (P_dp), default = 0.1
            gliding_constant (float): gliding constant (G_c) for exploration/exploitation balance, default = 1.9
            scaling_factor (int): scaling factor for gliding distance, default = 18
            beta (float): beta parameter for Levy flight, default = 1.5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_food_sources = self.validator.check_int("n_food_sources", n_food_sources, [1, 10])
        self.predator_prob = self.validator.check_float("predator_prob", predator_prob, [0.0, 1.0])
        self.gliding_constant = self.validator.check_float("gliding_constant", gliding_constant, [0.0, 10.0])
        self.scaling_factor = self.validator.check_float("scaling_factor", scaling_factor, [1, 100])
        self.beta = self.validator.check_float("beta", beta, [0.0, 10.0])
        self.set_parameters(["epoch", "pop_size", "n_food_sources", "predator_prob", "gliding_constant", "scaling_factor", "beta"])
        self.sort_flag = True

    def initialize_variables(self):
        # Aerodynamic parameters from the paper
        self.rho = 1.204  # Air density (kg/m³)
        self.velocity = 5.25  # Gliding velocity (m/s)
        self.surface_area = 154e-4  # Surface area (m²)
        self.height_loss = 8  # Height loss during gliding (m)
        # Assign roles: 1 hickory, 3 acorn, rest normal trees
        self.n_acorn_trees = self.n_food_sources - 1  # Number of acorn trees

    def calculate_gliding_distance(self):
        """Calculate gliding distance based on aerodynamics"""
        # Random lift coefficient variations (0.675 to 1.5)
        C_L = self.generator.uniform(0.675, 1.5)
        C_D = 0.60  # Fixed drag coefficient

        # Calculate lift and drag forces
        lift = 0.5 * self.rho * (self.velocity ** 2) * self.surface_area * C_L
        drag = 0.5 * self.rho * (self.velocity ** 2) * self.surface_area * C_D

        # Calculate glide angle
        glide_angle = np.arctan(drag / lift)

        # Calculate gliding distance
        d_g = self.height_loss / np.tan(glide_angle)

        # Scale down the gliding distance
        return d_g / self.scaling_factor

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Assign roles: 1 hickory, 3 acorn, rest normal trees
        pop_new = self.pop.copy()
        # Case 1: Acorn squirrels move toward hickory tree
        for idx in range(1, self.n_acorn_trees+1):
            d_g = self.calculate_gliding_distance()
            if self.generator.random() >= self.predator_prob:
                # No predator: move toward hickory
                pos_new = self.pop[idx].solution + d_g * self.gliding_constant * (self.g_best.solution - self.pop[idx].solution)
            else:
                # Predator present: random location
                pos_new = self.generator.uniform(self.problem.lb, self.problem.ub, self.problem.n_dims)
            # Apply boundary constraints
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            agent.target = self.pop[idx].target
            pop_new[idx] = agent

        # Case 2: Normal squirrels move toward acorn trees
        indices_random = np.array(list(range(self.pop_size - self.n_food_sources)))
        self.generator.shuffle(indices_random)
        indices_random = indices_random + self.n_food_sources # True indices of normal squirrels
        n_cut = self.generator.integers(1, self.pop_size - self.n_food_sources - 1)
        for idx in indices_random[n_cut:]:
            # Select random acorn tree
            jdx = self.generator.integers(0, self.n_acorn_trees) + 1
            d_g = self.calculate_gliding_distance()
            if self.generator.random() >= self.predator_prob:
                # No predator: move toward acorn
                pos_new = self.pop[idx].solution + d_g * self.gliding_constant * (self.pop[jdx].solution - self.pop[idx].solution)
            else:
                # Predator present: random location
                pos_new = self.generator.uniform(self.problem.lb, self.problem.ub, self.problem.n_dims)
            # Apply boundary constraints
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            agent.target = self.pop[idx].target
            pop_new[idx] = agent

        # Case 3: Normal squirrels move toward hickory tree
        for idx in indices_random[:n_cut]:
            d_g = self.calculate_gliding_distance()
            if self.generator.random() >= self.predator_prob:
                # No predator: move toward hickory
                pos_new = self.pop[idx].solution + d_g * self.gliding_constant * (self.pop[0].solution - self.pop[idx].solution)
            else:
                # Predator present: random location
                pos_new = self.generator.uniform(self.problem.lb, self.problem.ub, self.problem.n_dims)
            # Apply boundary constraints
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            agent.target = self.pop[idx].target
            pop_new[idx] = agent

        # Seasonal monitoring condition
        S_c = np.mean([ np.sqrt(np.sum((self.pop[idx].solution - self.pop[0].solution)**2)) for idx in range(1, self.n_food_sources) ])
        # Calculate minimum seasonal constant
        S_min = (10e-6 / 365) * (epoch / (self.epoch / 2.5))

        if S_c < S_min:
            # Winter season is over: randomly relocate some squirrels
            n_relocate = max(1, len(self.pop_size - self.n_food_sources) // 4)
            relocate_indices = self.generator.choice(indices_random, n_relocate, replace=False)
            for idx in relocate_indices:
                levy = self.get_levy_flight_step(beta = self.beta, multiplier = 0.01, size = self.problem.n_dims, case = -1)
                pos_new = self.problem.lb + levy * (self.problem.ub - self.problem.lb)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                agent.target = self.pop[idx].target
                pop_new[idx] = agent
        if self.mode in self.AVAILABLE_MODES:
            # Update target for the population
            pop_new = self.update_target_for_population(pop_new)
        else:
            for idx, agent in enumerate(pop_new):
                pop_new[idx].target = self.get_target(agent.solution)
        self.pop = pop_new
