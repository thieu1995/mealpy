#!/usr/bin/env python
# Created by "Thieu" at 10:54, 15/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo


class OriginalGSO(Optimizer):
    """
    The original version of: Glider Snake Optimizer (GSO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000].
        The main experiments in the original paper use 100 iterations. Default is 100.
    pop_size : int
        Number of search agents, in range [5, 10000].
        The main experiments in the original paper use 10 agents. Default is 10.
    rp : float
        Replacement probability used for weak agents in Algorithm 1.
        The paper introduces `RP` but does not clearly specify a default numerical value. Default is 0.5.

    Note
    ----
    The original paper contains several reproducibility ambiguities:

    - `RP` is required by Algorithm 1, but no definitive default value is
      reported. It must therefore be supplied explicitly.
    - Eq. (5) uses `Ind_r1` and `Sol_count` without formally defining them.
      This implementation interprets them as the one-based index of the first
      randomly selected agent and the population size, respectively.
    - Eq. (5) adds a scalar fitness-ratio term to a position vector. This
      implementation follows the equation literally, so the scalar is
      broadcast over all dimensions.
    - The paper does not define the predecessor of the first agent in the
      chain. Its predecessor contribution is therefore set to zero.
    - The paper states that the agent with the "highest" fitness becomes the
      leader, although all reported benchmark experiments are minimization
      problems. Mealpy's `minmax` setting is used to determine the best agent.
    - The exact boundary-handling strategy is not specified. Mealpy's
      standard solution correction is applied.
    - Eq. (5) can contain divisions by zero when objective values are zero.
      Numerical safeguards are applied only to prevent undefined arithmetic.

    GSO evaluates one new solution per agent per iteration. Thus, excluding
    initialization, it requires approximately `pop_size` objective-function
    evaluations per iteration.

    References
    ----------
    1. El-Kenawy, E.S.M., Khodadadi, N., Mirjalili, S., Zaki, A.M., Ibrahim,
       A., Alhussan, A.A., Khafaga, D.S. and Eid, M.M., 2026.
       Glider snake optimizer (GSO): a nature-inspired metaheuristic algorithm for global and
       engineering optimization problems. Artificial Intelligence Review, 59(3), p.91.
       https://doi.org/10.1007/s10462-026-11504-x

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, GSO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution ** 2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = GSO.OriginalGSO(epoch=100, pop_size=10, rp=0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Glider Snake Optimizer", year=2026, difficulty="easy", kind="original")

    def __init__(self, epoch: int = 100, pop_size: int = 10, rp: float = 0.5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 100.
            pop_size (int): Number of search agents, default = 10.
            rp (float): Weak-agent replacement probability.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.rp = self.validator.check_float("rp", rp, [0.0, 1.0])
        self.set_parameters(["epoch", "pop_size", "rp"])
        self.sort_flag = False

    def _safe_ratio(self, numerator, denominator):
        """
        Safely evaluate a scalar ratio used in Eq. (5).

        The safeguard is applied only when the denominator is numerically zero.
        """
        if np.abs(denominator) > self.EPSILON:
            return numerator / denominator
        if denominator < 0.0:
            denominator = -self.EPSILON
        else:
            denominator = self.EPSILON
        return numerator / denominator

    def _generate_replacement(self, idx, pop, leader, a, ):
        """
        Generate a weak-agent replacement according to Eq. (5).
        """
        # Algorithm 1 selects three random search agents.
        random_indices = self.generator.choice(self.pop_size, size=3, replace=False)
        idx_r1, idx_r2, idx_r3 = random_indices
        agent_r1 = pop[idx_r1]
        agent_r2 = pop[idx_r2]
        agent_r3 = pop[idx_r3]
        f_leader = leader.target.fitness
        f_current = pop[idx].target.fitness
        f_r2 = agent_r2.target.fitness
        f_r3 = agent_r3.target.fitness

        # Eq. (5) uses Ind_r1 / Sol_count.
        # The paper does not define Ind_r1 explicitly. Since the original
        # formulation uses one-based mathematical indexing, idx_r1 + 1 is used here.
        index_ratio = ((idx_r1 + 1) / self.pop_size)
        fitness_ratio = (self._safe_ratio(f_leader, f_current) + self._safe_ratio(f_r2, f_r3))

        # Eq. (5):
        # S_i(t+1) = Ind_r1 / Sol_count * S_r1(t) + A * (F_l / F_s + F_rs2 / F_rs3)
        # The second term is scalar in the published equation and is
        # therefore broadcast over the position dimensions.
        pos_new = (index_ratio * agent_r1.solution + a * fitness_ratio)
        return self.correct_solution(pos_new)

    def evolve(self, epoch):
        """
        The main operations of the Glider Snake Optimizer.

        Args:
            epoch (int): The current iteration.
        """
        # Eq. (4)
        a = 1.0 - (epoch - 1) / self.epoch
        # The paper sorts the population at the beginning of every iteration.
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)[0]
        # The current leader is the best solution after sorting.
        leader = pop_sorted[0]
        # Bottom 50% of the sorted population are weak agents.
        n_strong = int(np.ceil(self.pop_size / 2.0))

        pop_new = []
        for idx in range(self.pop_size):
            current = pop_sorted[idx]
            is_weak = idx >= n_strong
            # Algorithm 1:
            # if RP > rand and solution is weak
            #     replacement by Eq. (5)
            # else
            #     normal chain update by Eqs. (1)-(4)
            if is_weak and self.generator.random() < self.rp:
                pos_new = self._generate_replacement(idx, pop_sorted, leader, a)
            else:
                # Eq. (2): distance to current leader.
                dcl = (leader.solution - current.solution)
                # Eq. (3): distance to immediate predecessor.
                # The paper does not define a predecessor for the first
                # member of the chain. No predecessor contribution is used for that member.
                if idx == 0:
                    dpl = np.zeros(self.problem.n_dims)
                else:
                    dpl = (pop_sorted[idx - 1].solution - current.solution)
                # Eq. (1)
                pos_new = current.solution + a * (dcl + dpl)
                pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        # Algorithm 1 directly updates the population. No greedy
        # parent-offspring selection is stated in the paper.
        self.pop = pop_new
