#!/usr/bin/env python
# Created for Mealpy framework
# Chef-Based Optimization Algorithm (CBOA)
# Reference: Ayşe Beşkirli et al. (2025)
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalCBOA(Optimizer):
    """
    The original version of: Chef-Based Optimization Algorithm (CBOA)

    Chef-Based Optimization Algorithm is a population-based metaheuristic
    inspired by hierarchical cooperation and task distribution among chefs
    in professional kitchens. The population is divided into Chef Instructors
    and Cooking Students.

    Links:
        1. https://doi.org/10.1038/s41598-020-79893-w (Basis reference)
        2. https://doi.org/10.54287/gujsa.1667182 (Improved CBOA study)

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar
    >>> from mealpy.human_based import OriginalCBOA
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
    >>> # Parameters updated based on Beskirli (2025) experimental setup:
    >>> # epoch = 250, pop_size = 50
    >>> model = OriginalCBOA(epoch=250, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

    References
    ----------
    Beskirli, A. (2025). Improved Chef-Based Optimization Algorithm with Chaos-Based
    Fitness Distance Balance for Frequency-Constrained Truss Structures.
    GU J Sci, Part A, 12(2), 392-416. DOI: 10.54287/gujsa.1667182
    """

    def __init__(self, epoch: int = 250, pop_size: int = 50, **kwargs) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 250 (as per Beskirli, 2025)
            pop_size (int): Population size, default = 50
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])

        self.set_parameters(["epoch", "pop_size"])
        self.is_parallelizable = False
        # CBOA requires sorting to distinguish Chefs and Students
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # 1. Sort Population: Chef (Best) -> Students (Rest)
        # Mealpy handles sorting if self.sort_flag is True.
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
        
        # Global Best Position
        g_best_pos = self.g_best.solution

        # Determine number of Chef Instructors (Eq. 4)
        # N_c decreases as iterations progress (Exploration -> Exploitation)
        n_c = int(np.max([1, np.round(0.2 * self.pop_size * (1 - (epoch + 1) / self.epoch))]))

        pop_new = []

        # =================================================================
        # PHASE 1: CHEF INSTRUCTORS UPDATE (0 to n_c)
        # =================================================================
        for i in range(0, n_c):
            agent = pop_sorted[i]
            pos = agent.solution
            
            # --- Strategy 1: Learn from Global Best ---
            I = 1 + self.generator.random()
            rand_vec = self.generator.random(self.problem.n_dims)
            
            # Equation 6
            pos_new = pos + rand_vec * (g_best_pos - I * pos)
            pos_new = self.correct_solution(pos_new)
            
            # Greedy Selection 1
            agent_s1 = self.generate_agent(pos_new)
            if self.compare_target(agent_s1.target, agent.target, self.problem.minmax):
                agent = agent_s1

            # --- Strategy 2: Self-Improvement (Local Search) ---
            # Define local bounds based on current iteration (Equation 10)
            current_iter = epoch + 1
            
            if isinstance(self.problem.lb, (list, np.ndarray)):
                lo_local = np.array(self.problem.lb) / current_iter
                hi_local = np.array(self.problem.ub) / current_iter
            else:
                lo_local = self.problem.lb / current_iter
                hi_local = self.problem.ub / current_iter

            rand_vec_s2 = self.generator.random(self.problem.n_dims)
            pos_s2 = agent.solution + lo_local + rand_vec_s2 * (hi_local - lo_local)
            pos_s2 = self.correct_solution(pos_s2)

            # Greedy Selection 2
            agent_s2 = self.generate_agent(pos_s2)
            if self.compare_target(agent_s2.target, agent.target, self.problem.minmax):
                agent = agent_s2
            
            pop_new.append(agent)

        # =================================================================
        # PHASE 2: STUDENTS UPDATE (n_c to pop_size)
        # =================================================================
        for i in range(n_c, self.pop_size):
            agent = pop_sorted[i]
            pos = agent.solution

            # Pick a random chef index k (0 to n_c)
            k = self.generator.integers(0, n_c)
            chef_pos = pop_sorted[k].solution
            I = 1 + self.generator.random()

            # --- Strategy 1: Imitate Chef (Equation 12) ---
            rand_vec = self.generator.random(self.problem.n_dims)
            pos_new = pos + rand_vec * (chef_pos - I * pos)
            pos_new = self.correct_solution(pos_new)
            
            # Greedy Selection 1
            agent_s1 = self.generate_agent(pos_new)
            if self.compare_target(agent_s1.target, agent.target, self.problem.minmax):
                agent = agent_s1

            # --- Strategy 2: Copy Specific Dimension from Chef (Equation 14) ---
            pos_s2 = agent.solution.copy()
            rand_dim = self.generator.integers(0, self.problem.n_dims)
            pos_s2[rand_dim] = chef_pos[rand_dim]
            pos_s2 = self.correct_solution(pos_s2)

            # Greedy Selection 2
            agent_s2 = self.generate_agent(pos_s2)
            if self.compare_target(agent_s2.target, agent.target, self.problem.minmax):
                agent = agent_s2

            # --- Strategy 3: Random Perturbation / Trial and Error (Equation 16) ---
            pos_s3 = agent.solution.copy()
            rand_dim_s3 = self.generator.integers(0, self.problem.n_dims)
            current_iter = epoch + 1

            if isinstance(self.problem.lb, (list, np.ndarray)):
                lb_val = self.problem.lb[rand_dim_s3]
                ub_val = self.problem.ub[rand_dim_s3]
            else:
                lb_val, ub_val = self.problem.lb, self.problem.ub

            lo_val = lb_val / current_iter
            hi_val = ub_val / current_iter

            pos_s3[rand_dim_s3] = pos_s3[rand_dim_s3] + lo_val + self.generator.random() * (hi_val - lo_val)
            pos_s3 = self.correct_solution(pos_s3)

            # Greedy Selection 3
            agent_s3 = self.generate_agent(pos_s3)
            if self.compare_target(agent_s3.target, agent.target, self.problem.minmax):
                agent = agent_s3
            
            pop_new.append(agent)

        # Update population
        self.pop = pop_new
