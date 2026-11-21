#!/usr/bin/env python
# Created by "Ahmet Emin Karkinli" on 21/11/2026
# Github: https://github.com/Aekarkinli
# ------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo


class OriginalMDE(Optimizer):
    """
    The original version of: Multi-population based Differential Evolution (MDE)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations (MaxCycle in the paper), in range [1, 100000]. Default is 1000.
    pop_size : int
        Size of the full population A, in range [9, 10000]. Default is 90.
        In the paper the full population holds ``3*N`` vectors and each iteration
        evolves a randomly selected sub-population of ``N`` vectors. Here
        ``N = pop_size // 3``, so ``pop_size=90`` corresponds to ``N=30`` in the
        original MATLAB code. Each iteration costs ``N`` function evaluations.

    Links
    -----
    1. Paper: https://doi.org/10.1007/s00521-022-07969-w
    2. Matlab code: https://github.com/Aekarkinli/Multi-Population-based-Differential-Evolution-Algorithm

    Note
    ----
    MDE has no hyper-parameters other than ``epoch`` and ``pop_size``: the scale
    factor, the crossover map and the noise term are generated randomly inside
    the algorithm (parameter-free).

    This implementation follows the authors' MATLAB code (``algo_MDE.m``),
    which produced the results reported in the paper:

    - The full population A (``pop_size`` agents) is kept in ``self.pop``. In each
      iteration ``N`` distinct agents are drawn at random (sub-population B),
      a trial vector is generated for every member of B and the members of B
      are replaced greedily. B is then written back into A.
    - Mutation: ``T_i = B_r2 + scale_i * (dx - dy)`` with ``r1 != r2 != i``,
      ``dx_j`` randomly taken from ``B_r1`` or the best agent of A, ``dy_j`` from
      ``B_i`` or ``B_r2``, and ``scale = |a - rand^b| * randn^d`` with
      ``a ~ U{0, 1}``, ``b ~ U{1..10}``, ``d ~ U{1..5}``.
    - Crossover: a parameter-free binary map
      ``|randi(0..1, N, c) - rand(N, D)^randi(1..5, N, c)| < 0.5`` with
      ``c`` equal to 1 or D, randomly chosen in each iteration.
    - A small noise term ``B * 10^phi * (rand(N, 1) - 0.5)``, ``phi ~ U{-12..-9}``,
      computed from the updated sub-population, is added to the trial vectors
      of the next iteration (Eq. 13).
    - Out-of-bound variables are re-sampled uniformly inside [lb, ub] (Eq. 15
      of the paper).

    References
    ----------
    1. Karkinli, A. E. (2023). Detection of object boundary from point cloud by using
       multi-population based differential evolution algorithm. Neural Computing and
       Applications, 35, 5193-5206. https://doi.org/10.1007/s00521-022-07969-w

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, MDE
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
    >>> model = MDE.OriginalMDE(epoch=1000, pop_size=90)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Multi-population based Differential Evolution", year=2023, difficulty="easy", kind="original")

    def __init__(self, epoch: int = 1000, pop_size: int = 90, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 1000
            pop_size (int): size of the full population (3 * sub-population size), default = 90
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [9, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def initialize_variables(self):
        self.n_sub = self.pop_size // 3
        self.noise = np.zeros((self.n_sub, self.problem.n_dims))

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Eq. 15: every out-of-bound variable is re-sampled uniformly inside [lb, ub].
        """
        solution = np.array(solution, dtype=float)
        invalid = (solution < self.problem.lb) | (solution > self.problem.ub)
        if np.any(invalid):
            solution[invalid] = self.generator.uniform(self.problem.lb[invalid], self.problem.ub[invalid])
        return solution

    def get_scale__(self):
        """scale = |a - rand^b| * randn^d, a ~ U{0, 1}, b ~ U{1..10}, d ~ U{1..5}"""
        return abs(self.generator.integers(0, 2) - self.generator.random() ** self.generator.integers(1, 11)) * \
            self.generator.standard_normal() ** self.generator.integers(1, 6)

    def generate_trial__(self, pos_sub, pos_best):
        """
        Generate the trial vectors of the sub-population (mutation + parameter-free crossover + noise).

        Args:
            pos_sub (np.ndarray): positions of the sub-population B, shape (N, D)
            pos_best (np.ndarray): position of the best agent in the full population A

        Returns:
            np.ndarray: trial positions, shape (N, D)
        """
        n_sub, n_dims = pos_sub.shape
        pos_mutant = np.empty_like(pos_sub)
        for idx in range(n_sub):
            scale = self.get_scale__()
            while True:     # r1 != r2, r1 != idx, r2 != idx
                r1, r2 = self.generator.choice(n_sub, 2, replace=False)
                if r1 != idx and r2 != idx:
                    break
            dx = np.where(self.generator.random(n_dims) < 0.5, pos_sub[r1], pos_best)
            dy = np.where(self.generator.random(n_dims) < 0.5, pos_sub[idx], pos_sub[r2])
            pos_mutant[idx] = pos_sub[r2] + scale * (dx - dy)
        # Parameter-free crossover map, c is 1 or D
        cc = 1 if self.generator.random() ** self.generator.integers(1, 6) < 0.5 else n_dims
        cmap = np.abs(self.generator.integers(0, 2, (n_sub, cc)) -
                      self.generator.random((n_sub, n_dims)) ** self.generator.integers(1, 6, (n_sub, cc))) < 0.5
        return pos_sub + cmap * (pos_mutant + self.noise - pos_sub)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        best_agent, _ = self.get_best_agent(self.pop, self.problem.minmax)
        # Select the sub-population B (N distinct agents of A)
        idx_sub = self.generator.permutation(self.pop_size)[:self.n_sub]
        pop_sub = [self.pop[idx] for idx in idx_sub]
        pos_sub = np.array([agent.solution for agent in pop_sub])
        pos_trial = self.generate_trial__(pos_sub, best_agent.solution)
        pop_new = []
        for idx in range(self.n_sub):
            pos_new = self.correct_solution(pos_trial[idx])
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        # Greedy selection inside B, then write B back into A
        pop_sub = self.greedy_selection_population(pop_sub, pop_new, self.problem.minmax)
        for jdx, idx in enumerate(idx_sub):
            self.pop[idx] = pop_sub[jdx]
        # Noise term for the next iteration (Eq. 13), computed from the updated B
        pos_sub = np.array([agent.solution for agent in pop_sub])
        self.noise = pos_sub * 10.0 ** self.generator.integers(-12, -8, (self.n_sub, self.problem.n_dims)) * \
            (self.generator.random((self.n_sub, 1)) - 0.5)
