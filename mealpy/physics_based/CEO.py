#!/usr/bin/env python
# Created by "Baran Akyıldız" on 01/01/2026
# Github: https://github.com/akyilidizbaran
# ----------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo


class OriginalCEO(Optimizer):
    """
    The original version of: Cosmic Evolution Optimization (CEO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 1000.
    pop_size : int
        Number of population size, in range [10, 10000]. Default is 50.
    w1 : float
        Expansion weight, in range (0.0, 1.0). Default is 0.1.
    p_base : float
        Base collision probability, in range (0.0, 1.0). Default is 0.2.
    alpha : float
        Alignment parameter, in range (0.0, 1.0). Default is 0.7.

    References
    ~~~~~~~~~~
    1. Wang, Rui, Zhengxuan Jiang, and Guowen Ding. "Cosmic Evolution Optimization: A Novel Metaheuristic
       Algorithm for Numerical Optimization and Engineering Design." Mathematics 13.15 (2025): 2499.
       https://doi.org/10.3390/math13152499

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, CEO
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
    >>> model = CEO.OriginalCEO(epoch=1000, pop_size=50, w1=0.1, p_base=0.2, alpha=0.7)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Cosmic Evolution Optimization", year=2025, difficulty="hard", kind="original")

    def __init__(self, epoch: int = 1000, pop_size: int = 50,
                 w1: float = 0.1, p_base: float = 0.2, alpha: float = 0.7, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 1000
            pop_size (int): number of population size, default = 50
            w1 (float): Expansion weight, default=0.1
            p_base (float): Base collision probability, default=0.2
            alpha (float): Alignment parameter, default=0.7
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.w1 = self.validator.check_float("w1", w1, (0., 1.0))
        self.p_base = self.validator.check_float("p_base", p_base, (0., 1.0))
        self.alpha = self.validator.check_float("alpha", alpha, (0., 1.0))
        # Internal parameter C (number of systems) fixed to 3 as per paper specs
        self.C = 3
        
        self.set_parameters(["epoch", "pop_size", "w1", "p_base", "alpha"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Eq(3) Expansion speed: Vep(t)
        vep = self.w1 * (1 - 0.001 * (epoch / self.epoch)) * np.exp(-4 * epoch / self.epoch)
        # Eq(13) alpha(t)
        alpha_t = self.alpha * (1 + 0.0005 * (epoch / self.epoch))
        # Eq(15) Pglobal collision probability
        p_global = self.p_base * (1 - 0.0005 * (epoch / self.epoch)) * (1 - epoch / self.epoch)
        # Eq(17) Resonance probability Preson
        p_reson = 0.1 * np.exp(-3 * epoch / self.epoch)

        # Multi-stellar Setup (Top C solutions are centers)
        centers = [agent.solution for agent in self.pop[:self.C]]
        center_fits = [agent.target.fitness for agent in self.pop[:self.C]]
        
        # Eq(5) Calculate Radius Rc
        kc = max(2, int(round(self.pop_size / self.C)))
        
        radii = []
        for idx in range(self.C):
            center_pos = centers[idx]
            dists = np.linalg.norm([agent.solution - center_pos for agent in self.pop], axis=1)
            sorted_dists = np.sort(dists)
            nearest_dists = sorted_dists[:kc]
            rc = np.mean(nearest_dists)
            radii.append(rc)
        r_bar = np.mean(radii)
        
        # Use abs(f_best) for Eq(7)
        abs_f_best = np.abs(self.g_best.target.fitness)

        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.pop[idx]
            xi = agent.solution.copy()
            
            # --- Collision Strategy (Eq 16) ---
            if self.generator.random() < p_global:
                pos_new = xi + self.generator.normal(0, 1, self.problem.n_dims) * r_bar
            else:
                # --- Trajectory Update (Eq 14) ---
                # Eq(4) Expansion step part: Vep(t) * randn * (ub - lb)
                expansion = vep * self.generator.normal(0, 1, self.problem.n_dims) * (self.problem.ub - self.problem.lb)
                
                # Eq(6-9) Gravitational Force F(t)
                total_force = np.zeros(self.problem.n_dims)
                w_total = self.EPSILON
                
                for c in range(self.C):
                    o_c = centers[c]
                    f_c = center_fits[c]
                    
                    dist = np.linalg.norm(o_c - xi)
                    # Eq(6) u_c and F_c
                    u_c = (o_c - xi) / (dist + self.EPSILON)
                    f_c_force = u_c * np.exp(-dist / (radii[c] + self.EPSILON))
                    
                    # Eq(7) Weight coefficient w_c
                    diff = (agent.target.fitness - f_c) if self.problem.minmax == "min" else (f_c - agent.target.fitness)
                    w_c = np.exp(- diff / (abs_f_best + self.EPSILON))
                    
                    total_force += w_c * f_c_force
                    w_total += w_c
                    
                # Eq(9) Combined force
                F_t = total_force / w_total
                
                # Eq(12) Alignment A(t)
                s_i = self.g_best.solution - xi
                A_t = alpha_t * s_i * self.generator.normal(0, 1, self.problem.n_dims)
                
                # Eq(14) Full update
                pos_new = xi + expansion + F_t + A_t
                
            # Check bounds and evaluate
            pos_new = self.correct_solution(pos_new)
            agent_new = self.generate_empty_agent(pos_new)
            pop_new.append(agent_new)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[idx].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

        # Elite retention / Greedy Selection
        for idx in range(self.pop_size):
            if self.compare_target(pop_new[idx].target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = pop_new[idx]
            else:
                # Resonance Strategy Eq(17-18)
                if self.generator.random() < p_reson:
                    delta_x = 0.01 * self.generator.normal(0, 1, self.problem.n_dims) * (self.problem.ub - self.problem.lb)
                    pos_reson = pop_new[idx].solution + delta_x
                    pos_reson = self.correct_solution(pos_reson)
                    agent_new = self.generate_agent(pos_reson)
                    if self.compare_target(agent_new.target, self.pop[idx].target, self.problem.minmax):
                        self.pop[idx] = agent_new
