#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:52, 17/03/2020                                                               %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%
# STRICT PORT from MATLAB Source Code (Braik et al., 2022)                                              %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalWSO(Optimizer):
    """
    The original version of: White Shark Optimizer (WSO)
    
    Links:
        1. https://doi.org/10.1016/j.knosys.2022.109210
        2. https://github.com/malikbraik/White-Shark-Optimizer

    Notes:
        1. Strictly follows MATLAB source code logic.
        2. Frequency (f) is calculated as a constant (~0.899) matching the MATLAB source file 
           (Line 68 in WSO.m uses division '/', not multiplication '* rand').
        3. Global Best update uses Local Best memory, strictly following the sequential if-logic of the original code.
        4. Boundary handling mathematically matches MATLAB implementation (ub*a + lb*b).
    
    Args:
        epoch (int): Maximum number of iterations, default = 10000
        pop_size (int): Number of population size (white sharks), default = 100
        f_min (float): Minimum frequency for wave motion, default = 0.07
        f_max (float): Maximum frequency for wave motion, default = 0.75
        tau (float): Acceleration factor for velocity update, default = 4.11
        a0 (float): Movement strength coefficient 0, default = 6.250
        a1 (float): Movement strength coefficient 1, default = 100.0
        a2 (float): Movement strength coefficient 2, default = 0.0005
    
    Examples:
        >>> import numpy as np
        >>> from mealpy import FloatVar, WSO
        >>>
        >>> def objective_function(solution):
        >>>     return np.sum(solution**2)
        >>>
        >>> problem_dict = {
        >>>     "bounds": FloatVar(lb=(-10.,)*30, ub=(10.,)*30, name="delta"),
        >>>     "minmax": "min",
        >>>     "obj_func": objective_function
        >>> }
        >>>
        >>> model = WSO.OriginalWSO(epoch=100, pop_size=50)
        >>> g_best = model.solve(problem_dict)
        >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, 
                 f_min: float = 0.07, f_max: float = 0.75, tau: float = 4.11,
                 a0: float = 6.250, a1: float = 100.0, a2: float = 0.0005,
                 **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.f_min = self.validator.check_float("f_min", f_min, (0, 1.0))
        self.f_max = self.validator.check_float("f_max", f_max, (0, 1.0))
        self.tau = self.validator.check_float("tau", tau, (0, 10.0))
        self.a0 = self.validator.check_float("a0", a0, (0, 100.0))
        self.a1 = self.validator.check_float("a1", a1, (0, 1000.0))
        self.a2 = self.validator.check_float("a2", a2, (0, 1.0))
        
        self.set_parameters(["epoch", "pop_size", "f_min", "f_max", "tau", "a0", "a1", "a2"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        """Initialize algorithm-specific variables"""
        self.mu = 2.0 / abs(2.0 - self.tau - np.sqrt(self.tau**2 - 4.0 * self.tau))

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = np.zeros(self.problem.n_dims)
        local_solution = solution.copy()
        return Agent(solution=solution, velocity=velocity, local_solution=local_solution)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        # Initialize Personal Best
        agent.local_solution = agent.solution.copy()
        agent.local_target = agent.target.copy()
        return agent

    def evolve(self, epoch):
        """
        The main evolution step
        """
        mv = 1.0 / (self.a0 + np.exp((self.epoch / 2.0 - epoch) / self.a1))
        s_s = abs(1.0 - np.exp(-self.a2 * epoch / self.epoch))
        
        nu = np.floor(self.pop_size * self.generator.random(self.pop_size)).astype(int)
        
        # 1. Update Velocity
        for i in range(self.pop_size):
            rmin, rmax = 1.0, 3.0
            rr = rmin + self.generator.random() * (rmax - rmin)
            wr = abs((2.0 * self.generator.random() - (1.0 * self.generator.random() + self.generator.random())) / rr)
            
            # v[i] update
            self.pop[i].velocity = self.mu * self.pop[i].velocity + wr * (self.pop[nu[i]].local_solution - self.pop[i].solution)
        
        # 2. Update Position
        for i in range(self.pop_size):
            # STRICT MATLAB PORT: f = fmin + (fmax-fmin)/(fmax+fmin) -> Constant (~0.899)
            f = self.f_min + (self.f_max - self.f_min) / (self.f_max + self.f_min)
            
            # Boundary check logic (Using Booleans for safety)
            a = self.pop[i].solution > self.problem.ub # Boolean array (Upper Bound Violation)
            b = self.pop[i].solution < self.problem.lb # Boolean array (Lower Bound Violation)
            wo = np.logical_xor(a, b) # Boolean array (Any Violation)
            
            if self.generator.random() < mv:
                # MATLAB Logic: WSO_Positions(i,:) = WSO_Positions(i,:).*(~wo) + (ub.*a + lb.*b);
                # Correct implementation:
                # bound_val is calculated exactly as (ub * a + lb * b). 
                # Note: a and b act as masks (0 or 1).
                bound_val = self.problem.ub * a.astype(float) + self.problem.lb * b.astype(float)
                
                # Apply replacement where violation occurred (wo is True)
                self.pop[i].solution = np.where(wo, bound_val, self.pop[i].solution)
            else:
                self.pop[i].solution = self.pop[i].solution + self.pop[i].velocity / f
        
        # 3. Schooling (Sequential Chain Effect)
        for i in range(self.pop_size):
            for j in range(self.problem.n_dims):
                if self.generator.random() < s_s:
                    Dist = abs(self.generator.random() * (self.g_best.solution[j] - 1.0 * self.pop[i].solution[j]))
                    
                    if i == 0:
                        self.pop[i].solution[j] = self.g_best.solution[j] + self.generator.random() * Dist * np.sign(self.generator.random() - 0.5)
                    else:
                        WSO_Pos_ij = self.g_best.solution[j] + self.generator.random() * Dist * np.sign(self.generator.random() - 0.5)
                        self.pop[i].solution[j] = (WSO_Pos_ij + self.pop[i-1].solution[j]) / 2.0 * self.generator.random()
        
        # 4. Evaluate and Update Best
        for i in range(self.pop_size):
            # STRICT MATLAB PORT: Only evaluate if WITHIN bounds. Do not clip.
            if np.all((self.pop[i].solution >= self.problem.lb) & (self.pop[i].solution <= self.problem.ub)):
                
                # Evaluate fitness
                fit_new = self.get_target(self.pop[i].solution)
                self.pop[i].target = fit_new
                
                # Update Local Best
                if self.compare_target(fit_new, self.pop[i].local_target, self.problem.minmax):
                    self.pop[i].local_solution = self.pop[i].solution.copy()
                    self.pop[i].local_target = fit_new.copy()
                
                # Update Global Best (Independent check against Local Best Memory)
                if self.compare_target(self.pop[i].local_target, self.g_best.target, self.problem.minmax):
                    self.g_best.solution = self.pop[i].local_solution.copy()
                    self.g_best.target = self.pop[i].local_target.copy()