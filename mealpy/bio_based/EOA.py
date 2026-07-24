#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo, ScientificConcern


class OriginalEOA(Optimizer):
    """
    The original version of: Earthworm Optimisation Algorithm (EOA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.
    p_c : float
        Crossover probability, in range (0.0, 1.0). Default is 0.9.
    p_m : float
        Initial mutation probability, in range (0.0, 1.0). Default is 0.01.
    n_best : int
        How many of the best earthworm to keep from one generation to the next, in range [2, int(pop_size / 2)]. Default is 2.
    alpha : float
        Similarity factor, in range (0.0, 1.0). Default is 0.98.
    beta : float
        The initial proportional factor, in range (0.0, 1.0). Default is 0.9.
    gama : float
        A constant that is similar to cooling factor of a cooling schedule in the simulated annealing, in range (0.0, 1.0). Default is 0.9.


    .. attention::
       1. The author's MATLAB source code differs from the equations and parameters presented in the paper.
       2. This algorithm updates the population twice per epoch, resulting in double the number of function
          evaluations (NFEs) per epoch compared to standard algorithms.
       3. Users should be cautious when using algorithms published in low-quality journals like this.

    Links
    -----
    1. https://doi.org/10.1504/IJBIC.2018.093328
    2. https://www.mathworks.com/matlabcentral/fileexchange/53479-earthworm-optimization-algorithm-ewa

    References
    ~~~~~~~~~~
    1. Wang, G.G., Deb, S. and Coelho, L.D.S., 2018.
       Earthworm optimisation algorithm: a bio-inspired metaheuristic algorithm for global optimisation problems.
       International journal of bio-inspired computation, 12(1), pp.1-22.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, EOA
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
    >>> model = EOA.OriginalEOA(epoch=1000, pop_size=50, p_c = 0.9, p_m = 0.01, n_best = 2, alpha = 0.98, beta = 0.9, gama = 0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Earthworm Optimisation Algorithm", year=2018, difficulty="hard", kind="original",
                       scientific_status="questionable",
                       concerns=(
                           ScientificConcern.LACK_OF_NOVELTY,
                           ScientificConcern.QUESTIONABLE_MATH,
                           ScientificConcern.POOR_REPRODUCIBILITY,
                           ScientificConcern.FABRICATED_RESULTS
                       ))

    def __init__(self, epoch: int = 10000, pop_size: int = 100, p_c: float = 0.9, p_m: float = 0.01, n_best: int = 2,
                 alpha: float = 0.98, beta: float = 0.9, gama: float = 0.9, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_c (float): default = 0.9, crossover probability
            p_m (float): default = 0.01 initial mutation probability
            n_best (int): default = 2, how many of the best earthworm to keep from one generation to the next
            alpha (float): default = 0.98, similarity factor
            beta (float): default = 0.9, the initial proportional factor
            gama (float): default = 0.9, a constant that is similar to cooling factor of a cooling schedule in the simulated annealing.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p_c = self.validator.check_float("p_c", p_c, (0, 1.0))
        self.p_m = self.validator.check_float("p_m", p_m, (0, 1.0))
        self.n_best = self.validator.check_int("n_best", n_best, [2, int(self.pop_size / 2)])
        self.alpha = self.validator.check_float("alpha", alpha, (0, 1.0))
        self.beta = self.validator.check_float("beta", beta, (0, 1.0))
        self.gama = self.validator.check_float("gama", gama, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "p_c", "p_m", "n_best", "alpha", "beta", "gama"])
        self.sort_flag = True

    def initialize_variables(self):
        self.dyn_beta = self.beta

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        x_keep = np.array([self.pop[idx].solution for idx in range(0, self.n_best)])
        fit_list = np.array([agent.target.fitness for agent in self.pop])
        _, best_fit, worst_fit = self.get_special_fitness(self.pop, self.problem.minmax)
        if self.problem.minmax == "min":
            fits = worst_fit - fit_list + self.EPSILON
        else:
            fits = fit_list - worst_fit + self.EPSILON
        prob_selection = fits / np.sum(fits)
        self.dyn_beta = self.gama * self.beta

        # =========================================================
        # Begin Earthworm Reproduction Process
        pop_new = []
        for idx in range(self.pop_size):
            # Reproduction 1
            x0 = self.problem.lb + self.problem.ub - self.alpha * self.pop[idx].solution
            # Reproduction 2
            r1 = self.generator.integers(0, self.pop_size)
            if idx >= self.n_best:
                mate1 = self.generator.integers(0, self.pop_size)
                mate2 = self.generator.choice(self.pop_size, p=prob_selection)
                # Uniform Crossover (Vectorized numVar loop)
                rand_cross = self.generator.random(self.problem.n_dims)
                rand_select = self.generator.random(self.problem.n_dims)
                mask_p1 = ((rand_cross < self.p_c) & (rand_select > 0.5)) | ((rand_cross >= self.p_c) & (rand_select <= 0.5))
                x1 = np.where(mask_p1, self.pop[mate1].solution, self.pop[mate2].solution)
            else:
                x1 = self.pop[r1].solution
            # Combine Reproduction 1 and 2
            pos_new = self.dyn_beta * x0 + (1.0 - self.dyn_beta) * x1
            pos_new = self.correct_solution(pos_new)
            pop_new.append(self.generate_empty_agent(pos_new))
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        ## Update fitness in parallel
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

        # ---------------------------------------------------------
        # Cauchy Mutation (CM)
        # Update and sort
        self.pop, _ = self.get_sorted_population(pop_new, self.problem.minmax)
        self.pop, best, _ = self.get_special_agents(pop_new, n_best=1, n_worst=1, minmax=self.problem.minmax)
        pop_mean = np.mean([agent.solution for agent in self.pop], axis=0)
        pop_new = [self.pop[idx].copy() for idx in range(self.n_best)]
        for idx in range(self.n_best, self.pop_size):
            if idx < self.pop_size - self.n_best:
                mutation_mask = self.generator.random(self.problem.n_dims) < self.p_m
                cauchy_w = best[0].solution.copy()
                cauchy_w[mutation_mask] = pop_mean[mutation_mask]
                pos_new = (cauchy_w + best[0].solution) / 2.0
            else:
                pos_new = x_keep[idx - (self.pop_size - self.n_best)]
            pos_new = self.correct_solution(pos_new)
            pop_new.append(self.generate_empty_agent(pos_new))
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        ## Update fitness in parallel
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        self.pop = pop_new
