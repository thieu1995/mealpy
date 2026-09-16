#!/usr/bin/env python
# Created by "Thieu" at 05:47, 16/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo, ScientificConcern


class OriginalFNO(Optimizer):
    """
    The original version of: Farthest Better or Nearest Worse Optimizer (FNO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000].
        Default is 10000.
    pop_size : int
        Number of search agents, in range [5, 10000].
        The paper does not explicitly report the FNO population size in the
        experimental settings. The authors' official MATLAB implementation
        contains `nPop = 50` as a reference setting. Default is 50.

    Warnings
    --------
    Unlike other algorithms, this one does not rely on any natural principles; it appears as though
    the authors simply contrived the equations solely to make the algorithm work. Furthermore, there are
    numerous questionable inconsistencies between the paper and their published code. The paper and
    the code differ significantly, with the specific discrepancies listed below. Users should carefully
    consider this when validating the algorithm. Many new algorithms claim to be superior to other
    state-of-the-art methods, but it is evident that their implementations are often incorrect.

    Note
    ----
    The original paper contains several inconsistencies relevant to reproducibility:

    - Eq. (2) incorrectly prints the same fitness inequality as Eq. (1).
      The accompanying text defines NW as the nearest *worse* solution, so
      this implementation uses worse-fitness candidates for NW.
    - Eq. (6) prints `Min(0, ...)`, although the text states that phi is
      binary. Algorithm behavior and the authors' official implementation
      use `max(0, ...)`, which is followed here.
    - Eq. (5) omits the absolute value around the DFS random vector, while
      Algorithm 1 and the official implementation use its absolute value.
    - The paper uses `r < alpha` to choose the NW-jumping phase, whereas the
      official implementation uses `r < 1 - FEs / MaxFEs`. This class follows
      the paper formulation, i.e. `r < alpha`.
    - The official implementation progressively truncates the set of better
      candidates before selecting FB. This mechanism is not described in
      Eq. (1), Algorithm 1, or the corresponding paper text and is therefore
      not included here.
    - The paper requires FEs / MaxFEs in Eq. (4). Therefore, an NFE-based
      termination criterion is required for faithful reproduction.
    - When no strictly better or worse solution exists, the paper does not
      define a fallback. Following the authors' implementation, the current
      solution itself is used.
    - The paper states that out-of-bound positions are clipped but does not
      define the repair operator. The authors' implementation randomly
      reinitializes each violated dimension within its bounds; this behavior
      is followed here.

    FNO evaluates one candidate per agent per iteration. Its dominant
    additional cost is the pairwise Euclidean distance matrix, requiring
    O(pop_size^2 * n_dims) operations per iteration.

    Links
    -----
    1. Matlab: https://github.com/AhmadTaheri2021/FNO-Optimizer/blob/main/Run_FNO.m
    2. Paper: https://doi.org/10.1007/s10462-025-11443-z

    References
    ----------
    1. Taheri, A., RahimiZadeh, K., Baumbach, J., Beheshti, A., Zolotareva, O., Al-Betar, M.A.,
       Mirjalili, S. and Gandomi, A.H., 2026.
       Farthest better or nearest worse optimizer: A novel metaheuristic algorithm.
       Artificial Intelligence Review, 59(2), p.79.

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, FNO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution ** 2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>> model = FNO.OriginalFNO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Farthest Better or Nearest Worse Optimizer",
                       year=2026, difficulty="medium", kind="original", scientific_status="questionable",
                       concerns=(ScientificConcern.INCORRECT_EQUATIONS, ScientificConcern.POOR_REPRODUCIBILITY,
                                 ScientificConcern.CODE_PSEUDOCODE_MISMATCH, ScientificConcern.AMBIGUOUS_METHODOLOGY,
                                 ScientificConcern.INSUFFICIENT_VALIDATION, ScientificConcern.QUESTIONABLE_MATH)
                       )

    def __init__(self, epoch: int = 10000, pop_size: int = 50, **kwargs: object) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 10000.
            pop_size (int): Number of search agents, default = 50.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Repair violated dimensions using the authors' reference behavior.

        Each dimension outside its bounds is independently reinitialized
        uniformly inside the corresponding feasible interval.
        """
        invalid = ((solution < self.problem.lb) | (solution > self.problem.ub))
        if np.any(invalid):
            solution[invalid] = self.generator.uniform(self.problem.lb[invalid], self.problem.ub[invalid])
        return solution

    def get_fb_nw_indices(self):
        """
        Determine the Farthest Better (FB) and Nearest Worse (NW)
        solution for every agent according to Eqs. (1)-(2).

        Returns
        -------
        fb_indices : np.ndarray
            Index of the farthest strictly better solution for each agent.
        nw_indices : np.ndarray
            Index of the nearest strictly worse solution for each agent.
        """
        positions = np.asarray([agent.solution for agent in self.pop])
        fitness = np.asarray([agent.target.fitness for agent in self.pop])

        # Pairwise Euclidean distance matrix.
        differences = (positions[:, None, :] - positions[None, :, :])
        distances = np.linalg.norm(differences, axis=2)
        fb_indices = np.arange(self.pop_size)
        nw_indices = np.arange(self.pop_size)
        for idx in range(self.pop_size):
            # Eq. (1): strictly better candidates.
            better_indices = np.flatnonzero(fitness < fitness[idx])
            # Eq. (2): strictly worse candidates.
            # The printed equation contains '<', but the accompanying
            # definition explicitly states f(X_j) > f(X_i) for NW.
            worse_indices = np.flatnonzero(fitness > fitness[idx])
            if better_indices.size > 0:
                local_idx = np.argmax(distances[idx, better_indices])
                fb_indices[idx] = (better_indices[local_idx])
            if worse_indices.size > 0:
                local_idx = np.argmin(distances[idx, worse_indices])
                nw_indices[idx] = (worse_indices[local_idx])
        return fb_indices, nw_indices

    def evolve(self, epoch):
        """
        The main operations of the Farthest Better or Nearest Worse
        Optimizer.

        Args:
            epoch (int): The current iteration.
        """
        # --------------------------------------------------------------
        # Eqs. (1)-(2): determine FB and NW using the population at the
        # beginning of the current generation.
        # --------------------------------------------------------------
        fb_indices, nw_indices = self.get_fb_nw_indices()
        pop_new = []
        for idx in range(self.pop_size):
            current = self.pop[idx].solution
            fb_pos = self.pop[fb_indices[idx]].solution
            nw_pos = self.pop[nw_indices[idx]].solution

            # ----------------------------------------------------------
            # Eq. (4): dynamic exploration/exploitation coefficient.
            # ----------------------------------------------------------
            alpha = np.sqrt(max(0.0, 1.0 - (self.nfe_counter - 1.) / (self.epoch * self.pop_size)))
            # ----------------------------------------------------------
            # Eq. (5): Dynamic Focus Strategy.
            # Algorithm 1 and the official implementation use the absolute value of U(mu-alpha, mu+alpha).
            # ----------------------------------------------------------
            mu = self.generator.random()
            dfs = np.abs(self.generator.uniform(mu - alpha, mu + alpha, self.problem.n_dims))
            # ----------------------------------------------------------
            # Eq. (10)
            # U([1, 2]) in Algorithm 1 is a discrete draw from {1, 2}.
            # ----------------------------------------------------------
            w = (2.0 + self.generator.uniform(-1.0, 1.0)) * self.generator.integers(1, 3)

            # ----------------------------------------------------------
            # Eq. (6)
            # The published equation prints Min(0, ...), but the text
            # defines phi as binary and the official implementation uses max(0, ...).
            # ----------------------------------------------------------
            phi = np.maximum(0.0, np.sign(fb_pos - current) * np.sign(nw_pos - current))
            # ----------------------------------------------------------
            # Eq. (7): temporary position.
            # ----------------------------------------------------------
            r_vec = self.generator.random(self.problem.n_dims)
            pos_temp = phi * nw_pos + (1.0 - phi) * current + np.sign(fb_pos - current) * r_vec * np.abs(current - nw_pos)

            # ----------------------------------------------------------
            # Eqs. (8)-(11)
            # Paper:
            #   r < alpha     -> NW jumping
            #   otherwise     -> FB + DFS exploration
            # ----------------------------------------------------------
            r = self.generator.random()
            if r < alpha:
                # Eq. (8)
                pos_new = pos_temp
            else:
                # Eq. (9)
                step = (dfs * (fb_pos - current))
                # Eq. (11)
                pos_new = (pos_temp + w * step)

            # Algorithm 1: bound correction before evaluation.
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

        # Algorithm 1, lines 23-27: greedy replacement.
        self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
