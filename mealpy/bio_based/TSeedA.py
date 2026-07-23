#!/usr/bin/env python
# Created by "Ahmet Tunahan Yalcin" on 28/02/2026
# Email: ytunahan7878@gmail.com
# Github: https://github.com/tunayalc
# ----------------------------------------------%

from mealpy.optimizer import Optimizer


class OriginalTSeedA(Optimizer):
    """
    The original version: Tree-Seed Algorithm (TSeedA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations. Default is 10000.
    pop_size : int
        Population size (number of trees). Default is 100.
    st : float
        Search tendency parameter, in range (0.0, 1.0). Default is 0.1.

    Danger
    ------
    1. Lack of Mathematical Novelty: The search equations (Eq. 3 and Eq. 4)
       are functionally equivalent to basic difference-based mutation operators
       found in classical Differential Evolution (DE) and Particle Swarm Optimization (PSO).
    2. Over-Simplistic Selection: The exploration-exploitation balance is managed
       solely by a simple 'if-else' decision branch controlled by a single parameter
       (Search Tendency, ST) , which lacks the dynamic adaptation mechanisms of modern metaheuristics.
    3. Low Selection Pressure: Replacing parent trees directly with marginally
       better seeds can lead to premature convergence, high stagnation rates,
       and poor performance on high-dimensional multimodal landscapes.
    4. For solving high-performance or real-world industrial continuous optimization problems,
       users are strongly encouraged to choose more robust, mathematically sound, and modern algorithms

    References
    ----------
    1. Kiran, M. S. (2015). TSA: Tree-seed algorithm for continuous optimization.
       Expert Systems with Applications, 42(19), 6686-6698. https://doi.org/10.1016/j.eswa.2015.04.055

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, TSeedA
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
    >>> model = TSeedA.OriginalTSeedA(epoch=1000, pop_size=50, st=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, st: float = 0.1, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.st = self.validator.check_float("st", st, [0, 1.0])
        self.set_parameters(["epoch", "pop_size", "st"])
        self.sort_flag = False

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm.
        """
        # Decide seed production limits (10% to 25% of population size)
        min_seeds = max(1, int(0.1 * self.pop_size))
        max_seeds = max(2, int(0.25 * self.pop_size))

        for idx in range(self.pop_size):
            # Decide the number of seeds produced for this tree
            n_seeds = self.generator.integers(min_seeds, max_seeds + 1)
            
            pop_new = []
            rdx_list = self.sample_indexes_exclude_one(self.generator, self.pop_size, idx, n_samples=n_seeds, replace=True)
            for jdx in range(n_seeds):
                # Select a random tree 'r' different from 'i'
                rdx = rdx_list[jdx]

                # Create a new seed from the current tree
                seed = self.pop[idx].solution.copy()
                alpha = self.generator.uniform(-1, 1, self.problem.n_dims)  # Scaling factor

                # Update dimensions based on Search Tendency (ST)
                rand_vals = self.generator.random(self.problem.n_dims)
                mask_eq3 = rand_vals < self.st
                mask_eq4 = ~mask_eq3

                # Update using Eq. 3
                seed[mask_eq3] = self.pop[idx].solution[mask_eq3] + alpha[mask_eq3] * (self.g_best.solution[mask_eq3] - self.pop[idx].solution[mask_eq3])

                # Update using Eq. 4
                seed[mask_eq4] = self.pop[idx].solution[mask_eq4] + alpha[mask_eq4] * (self.pop[idx].solution[mask_eq4] - self.pop[rdx].solution[mask_eq4])

                # Boundary enforcement and create new agent
                pos_new = self.correct_solution(seed)
                # agent = self.generate_empty_agent(pos_new)
                pop_new.append(self.generate_empty_agent(pos_new))
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1].target = self.get_target(seed)
            # Update parallel
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_for_population(pop_new)

            # Update the current tree by the best seed
            best, _ = self.get_best_agent(pop_new, self.problem.minmax)
            if self.compare_target(best.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = best
