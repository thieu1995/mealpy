#!/usr/bin/env python
# Created by "Thieu" at 17:45, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalOrcaOA(Optimizer):
    """
    The original version of: Orca Optimization Algorithm (OrcaOA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, default = 10000.
    pop_size : int
        Number of population size, default = 100.
    p_percent : float
        The percentage of worst orcas to remove and regenerate per iteration, default = 0.1.
    R0 : float
        The initial radius of the ice floe, default=2.0.

    References
    ~~~~~~~~~~
    1. Golilarz, N. A., Gao, H., Addeh, A., & Pirasteh, S. (2020, December).
       ORCA optimization algorithm: A new meta-heuristic tool for complex optimization problems.
       In 2020 17th International Computer Conference on Wavelet Active Media Technology and
       Information Processing (ICCWAMTIP) (pp. 198-204). IEEE. https://doi.org/10.1109/ICCWAMTIP51612.2020.9317473

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ORCA
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
    >>> model = ORCA.OriginalOrcaOA(epoch=1000, pop_size=50, p_percent=0.15, R0=5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, p_percent: float=0.1, R0: float=2.0, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p_percent = self.validator.check_float("p_percent", p_percent, [0, 1.0])
        self.R0 = self.validator.check_float("R0", R0, [-1000, 1000])
        self.set_parameters(["epoch", "pop_size", "p_percent", "R0"])
        self.R = self.R0
        self.sort_flag = True
        self.n_removes = int(self.p_percent * pop_size)

    def evolve(self, epoch: int):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        fit_list = np.array([agent.target.fitness for agent in self.pop])
        # Calculate energy (e_i = F_i - F_s)
        e = fit_list - self.g_best.target.fitness
        e_min = np.min(e)
        e_max = np.max(e)

        # Compute the normalized energy of each orca
        if e_max == e_min:
            epsilon = np.zeros(self.pop_size)
        else:
            epsilon = (e - e_min) / (e_max - e_min)
        # Calculate d_i
        d_list = epsilon * self.R

        pop_new = []
        # Move the orcas toward the Seal and ice floe
        for idx in range(1, self.pop_size):  # Keep the best solution intact (elitism)
            if idx < self.pop_size - self.n_removes:
                # Generate random direction (unit vector)
                direction = self.generator.standard_normal(self.problem.n_dims)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                # Random position between R-d_i and R
                r_dist = self.generator.uniform(self.R - d_list[idx], self.R)
                # Update position
                pos_new = self.pop[0].solution + r_dist * direction
            else:
                # Remove the worst orcas (P%) and generate them randomly for the next iteration
                pos_new = self.generator.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.correct_solution(pos_new)
            pop_new.append(self.generate_empty_agent(pos_new))
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        # Evaluate fitness in parallel
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        self.pop = pop_new

        # Reduce the radius of the ice floe (simulated linear decay)
        self.R = self.R0 * (1. - epoch / self.epoch)


class OriginalOrcaPA(Optimizer):
    """
    The original version of: Orca Predation Algorithm (OrcaPA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.
    p1 : float
        Probability to select driving vs encircling, in range [0.0, 1.0]. Default is 0.5.
    p2 : float
        Probability for position adjustment, in range [0.0, 1.0]. Default is 0.1.
    q : float
        Probability parameter for driving methods, in range [0.0, 1.0]. Default is 0.9.


    .. caution::
       1. This algorithm uses approximately 2x more Number of Function Evaluations (NFEs) than other algorithms.
       That is, it calls the fitness function 2.x times per epoch, where x depends on the probability parameter "p_2".
       Therefore, users should be cautious when applying it to large-scale problems, as it will be very slow.

       2. This algorithm borrows ideas from the Whale Optimization Algorithm (WOA) and Grey Wolf Optimization (GWO),
       with slight modifications to the equations. Conceptually, however, the underlying ideas remain the same.

       3. This algorithm uses the same animal motif as the original Orca Optimization Algorithm, despite
       differences in the equations.

    References
    ~~~~~~~~~~
    1. Jiang, Y., Wu, Q., Zhu, S., & Zhang, L. (2022).
       Orca predation algorithm: A novel bio-inspired algorithm for global optimization problems.
       Expert Systems with Applications, 188, 116026. https://doi.org/10.1016/j.eswa.2021.116026

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ORCA
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
    >>> model = ORCA.OriginalOrcaPA(epoch=1000, pop_size=50, p1=0.5, p2=0.3, q=0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, p1: float=0.5, p2: float=0.1, q: float=0.9, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p1 = self.validator.check_float("p1", p1, [0, 1.0])
        self.p2 = self.validator.check_float("p2", p2, [0, 1.0])
        self.q = self.validator.check_float("q", q, [0, 1.0])
        self.set_parameters(["epoch", "pop_size", "p1", "p2", "q"])
        self.sort_flag = False

    def evolve(self, epoch: int):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Identify the four best orcas for the attacking phase.
        _, best, _ = self.get_special_agents(self.pop, n_best=4, n_worst=1, minmax=self.problem.minmax)
        M = np.mean([agent.solution for agent in self.pop], axis=0)  # Average position of the orca group.

        # Step 3: Position updating of the orca group at the chasing phase.
        pop_chase = []
        for idx in range(self.pop_size):
            if self.generator.random() > self.p1:
                # Driving of prey
                if self.generator.random() > self.q:
                    # First chasing method
                    a, b, d = self.generator.random(3)
                    F = 2.0
                    c = 1.0 - b
                    v_chase = a * (d * self.g_best.solution - F * (b * M + c * self.pop[idx].solution))
                else:
                    # Second chasing method
                    v_chase = self.generator.uniform(0, 2) * self.g_best.solution - self.pop[idx].solution
                new_pos = self.pop[idx].solution + v_chase
            else:
                # Encircling of prey
                j1, j2, j3 = self.sample_indexes_exclude_one(self.generator, self.pop_size, exclude_idx=idx, n_samples=3, replace=False)
                u = 2 * (self.generator.random() - 0.5) * (1 -  epoch / self.epoch)
                new_pos = self.pop[j1].solution + u * (self.pop[j2].solution - self.pop[j3].solution)
            pos_new = self.correct_solution(new_pos)
            pop_chase.append(self.generate_empty_agent(pos_new))
            if self.mode not in self.AVAILABLE_MODES:
                pop_chase[-1].target = self.get_target(pos_new)
        # Evaluate fitness in parallel
        if self.mode in self.AVAILABLE_MODES:
            pop_chase = self.update_target_for_population(pop_chase)
        pop_chase = self.greedy_selection_population(self.pop, pop_chase, self.problem.minmax)

        # Step 4: Position updating of orca group at the attacking phase.
        flag_attack = [False, ] * self.pop_size
        pop_new = []
        for idx in range(self.pop_size):
            j1, j2, j3 = self.sample_indexes_exclude_one(self.generator, self.pop_size, exclude_idx=idx, n_samples=3, replace=False)
            v_attack_1 = (best[0].solution + best[1].solution + best[2].solution + best[3].solution) / 4.0 - pop_chase[idx].solution
            v_attack_2 = (pop_chase[j1].solution + pop_chase[j2].solution + pop_chase[j3].solution) / 3.0 - self.pop[idx].solution

            g1 = self.generator.uniform(0, 2)
            g2 = self.generator.uniform(-2.5, 2.5)
            x_attack = pop_chase[idx].solution + g1 * v_attack_1 + g2 * v_attack_2
            pos_new = self.correct_solution(x_attack)
            pop_new.append(self.generate_empty_agent(pos_new))
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
                if self.compare_target(pop_new[-1].target, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx] = pop_new[-1].copy()
                else:
                    flag_attack[idx] = True
        # Evaluate fitness in parallel
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            for idx in range(self.pop_size):
                if self.compare_target(pop_new[-1].target, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx] = pop_new[-1].copy()
                else:
                    flag_attack[idx] = True

        # Position adjustment phase during an attack
        for idx in range(0, self.pop_size):
            if flag_attack[idx]:
                if self.generator.random() < self.p2:
                    # Assign the minimum boundary value (lb) for some dimensions
                    cond = self.generator.random(self.problem.n_dims) < self.p2
                    pos_new = np.where(cond, self.problem.lb, pop_chase[idx].solution)
                    self.pop[idx] = self.generate_agent(pos_new)
                else:
                    self.pop[idx] = pop_chase[idx].copy()
