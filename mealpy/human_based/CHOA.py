#!/usr/bin/env python
# Created by "Thieu" at 10:35, 16/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo, ScientificConcern


class OriginalCHOA(Optimizer):
    """
    The original version of: Cultural History Optimization Algorithm (CHOA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000].
        The paper uses 1000 iterations in the main benchmark experiments. Default is 1000.
    pop_size : int
        Number of ideas in the population, in range [10, 10000].
        The paper uses 50 ideas in the main benchmark experiments. Default is 50.
    n_clusters : int
        Number of idea ranks/clusters, in range [1, pop_size].
        Table 6 uses Nclus = 10 for the main benchmark experiments. Default is 10.
    p_comb : float
        Fraction of the population assigned to the combination operator.
        Table 6 uses Pcomb = 0.5. Default is 0.5.
    p_comp : float
        Fraction of the population assigned to the competition operator.
        Table 6 uses Pcomp = 0.5. Default is 0.5.
    c : float
        Modified uniform-crossover coefficient used in Eq. (1).
        The paper introduces this parameter but does not report a universal
        numerical value for the main benchmark experiments. Default is 0.1

    Note
    ----
    This implementation follows Eqs. (1)-(7), Algorithm 1, Fig. 1, and the
    parameter settings reported in the paper as closely as possible.
    Several implementation-critical details are not fully specified:

    - Eq. (1) is printed as
      `alpha = -gamma + rand * (1 + 2*gamma)`, while the nomenclature and
      GUI refer to the modified crossover coefficient as `c`. This
      implementation treats `gamma` in Eq. (1) as the parameter `c`.
    - The paper states that ideas are divided into `Nclus` ranked groups of
      constant size, but does not give an explicit clustering algorithm.
      Here, ideas are sorted by fitness and divided consecutively into
      approximately equal-sized rank groups.
    - The paper does not precisely define which members of a rank are
      considered "elite". Here, the best member of each rank is the elite
      representative; the remaining members are treated as non-elites.
    - The exact roulette-wheel probability formula is not provided. Rank
      weights are therefore used: high-ranked elite representatives receive
      larger weights for combination, while weaker non-elites receive larger
      weights for inverse-roulette competition.
    - The paper requires the two parents in the combination operator to come
      from different clusters. This restriction is enforced explicitly.
    - The competition operator selects two non-elites and compares them.
      Eq. (6) is applied to the worse competitor using the better competitor
      and the current global best.
    - Fig. 1 shows `Ncomb / 2` combination events and `Ncomp / 2`
      competition events. Since each combination produces two offspring and
      each competition updates one of two competitors, this is also
      consistent with the NFE expression in Table 6:
      `Npop + (Ncomb + Ncomp / 2) * Tm`.
    - The paper does not specify rounding rules when `Pcomb * Npop` or
      `Pcomp * Npop` is odd. This implementation rounds down to the nearest
      even number.
    - Acceptance is described as elitist truncation: the previous population
      and newly generated ideas are pooled, ranked, and truncated to
      `pop_size`.
    - The paper recommends applying Eq. (6) to all ideology factors; this
      implementation therefore updates the complete solution vector.
    - Boundary handling for Eqs. (2), (3), and (6) is clipping in the paper.
      Mealpy's standard solution correction is used equivalently.
    - The benchmark section reports `Nclus = 10`, `Pcomb = 0.5`, and
      `Pcomp = 0.5`, but no universal value of `c` is reported. Therefore,
      the default `c = 0.1` is a Mealpy default, not a claimed paper value.

    For the main benchmark configuration (`pop_size=50`, `p_comb=0.5`, `p_comp=0.5`), the nominal
    number of new evaluations per iteration is `Ncomb + Ncomp/2`, rather than `pop_size`.

    References
    ----------
    1. Sharifi, T., Mirsalim, M., Soleimanian Gharehchopogh, F. and Mirjalili, S., 2025.
       Cultural history optimization algorithm: a new human-inspired metaheuristic algorithm for
       engineering optimization problems. Neural Computing and Applications, 37(25), pp.21009-21068.
       https://doi.org/10.1007/s00521-025-11379-z

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, CHOA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution ** 2)
    >>>
    >>> problem = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>> model = CHOA.OriginalCHOA(epoch=1000, pop_size=50, n_clusters=10, p_comb=0.5, p_comp=0.5)
    >>> g_best = model.solve(problem)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Cultural History Optimization Algorithm", year=2025, difficulty="hard", kind="original",
                       concerns=(ScientificConcern.POOR_REPRODUCIBILITY,
                                 ScientificConcern.CODE_PSEUDOCODE_MISMATCH, ScientificConcern.AMBIGUOUS_METHODOLOGY,
                                 ScientificConcern.INSUFFICIENT_VALIDATION, ScientificConcern.QUESTIONABLE_MATH)
                       )

    def __init__(self, epoch: int = 1000, pop_size: int = 50, n_clusters: int = 10, p_comb: float = 0.5,
            p_comp: float = 0.5, c: float = 0.1, **kwargs: object, ) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 1000.
            pop_size (int): Population size, default = 50.
            n_clusters (int): Number of idea ranks/clusters, default = 10.
            p_comb (float): Combination percentage, default = 0.5.
            p_comp (float): Competition percentage, default = 0.5.
            c (float): Modified uniform-crossover coefficient, default = 0.1.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_clusters = self.validator.check_int("n_clusters", n_clusters, [1, self.pop_size])
        self.p_comb = self.validator.check_float("p_comb", p_comb, [0.0, 1.0])
        self.p_comp = self.validator.check_float("p_comp", p_comp, [0.0, 1.0])
        self.c = self.validator.check_float("c", c, [0.0, 1.0])
        self.set_parameters(["epoch", "pop_size", "n_clusters", "p_comb", "p_comp", "c"])
        self.sort_flag = True

    def initialize_variables(self):
        """
        Calculate the nominal population sizes assigned to the two operators.
        """
        # Fig. 1 explicitly performs Ncomb/2 and Ncomp/2 pair operations.
        # Therefore, use even operator population sizes.
        self.n_comb = int(self.p_comb * self.pop_size)
        self.n_comp = int(self.p_comp * self.pop_size)
        self.n_comb -= self.n_comb % 2
        self.n_comp -= self.n_comp % 2

    def get_rank_groups(self, pop):
        """
        Divide the fitness-ranked population into approximately equal groups.

        The paper specifies ranked groups with constant group size but does
        not provide an explicit clustering procedure. Consecutive splitting
        of the sorted population is used here.
        """
        pop_sorted, _ = self.get_sorted_population(pop, self.problem.minmax)
        indices = np.arange(self.pop_size)
        # np.array_split is used only because Npop may not be divisible
        # exactly by Nclus. The paper does not define this case.
        groups = np.array_split(indices, self.n_clusters)
        return pop_sorted, groups

    @staticmethod
    def get_rank_weights(n_items, reverse=False):
        """
        Create rank-based roulette-wheel weights.

        Better ranks receive larger probability in normal roulette
        selection. The order is reversed for inverse roulette selection.
        """
        weights = np.arange(n_items, 0, -1, dtype=float, )
        if reverse:
            weights = weights[::-1]
        return weights / np.sum(weights)

    def roulette_index(self, weights):
        """
        Draw one index according to the supplied roulette probabilities.
        """
        return int(self.generator.choice(len(weights), p=weights))

    def get_elite_and_non_elite_indices(self, groups):
        """
        Extract one elite representative from each ranked group.

        The first member in every fitness-ranked group is treated as that
        group's elite representative. Remaining members are non-elites.
        """
        elite_indices = []
        non_elite_indices = []
        cluster_of = {}
        for cluster_id, group in enumerate(groups):
            if len(group) == 0:
                continue
            elite_indices.append(int(group[0]))
            cluster_of[int(group[0])] = cluster_id
            for idx in group[1:]:
                idx = int(idx)
                non_elite_indices.append(idx)
                cluster_of[idx] = cluster_id
        return np.asarray(elite_indices, dtype=int), np.asarray(non_elite_indices, dtype=int), cluster_of

    def combination_operator(self, pop_sorted, elite_indices, cluster_of, ):
        """
        Generate ideas using the combination operator, Eqs. (1)-(5).
        """
        if self.n_comb < 2 or len(elite_indices) < 2:
            return []
        offspring = []
        # High-ranked elites are more likely to be selected.
        weights = self.get_rank_weights(len(elite_indices), reverse=False, )
        for _ in range(self.n_comb // 2):
            # ----------------------------------------------------------
            # Select the first elite using roulette-wheel selection.
            p1_slot = self.roulette_index(weights)
            p1_idx = elite_indices[p1_slot]
            # ----------------------------------------------------------
            # The second parent must belong to another cluster.
            valid_slots = np.asarray([k for k, idx in enumerate(elite_indices) if cluster_of[int(idx)] != cluster_of[int(p1_idx)]])
            if valid_slots.size == 0:
                continue
            valid_weights = weights[valid_slots]
            valid_weights = valid_weights / np.sum(valid_weights)
            p2_slot = int(self.generator.choice(valid_slots, p=valid_weights))
            p2_idx = elite_indices[p2_slot]
            x1 = pop_sorted[p1_idx].solution
            x2 = pop_sorted[p2_idx].solution
            # ----------------------------------------------------------
            # Eq. (1)
            # alpha = -c + rand * (1 + 2*c)
            # The paper prints gamma in Eq. (1), while the nomenclature
            # calls the crossover coefficient c.
            alpha = (-self.c + self.generator.random(self.problem.n_dims) * (1.0 + 2.0 * self.c))
            # Eqs. (2)-(3)
            pos_1 = (alpha * x1 + (1.0 - alpha) * x2)
            pos_2 = (alpha * x2 + (1.0 - alpha) * x1)
            # Eqs. (4)-(5)
            pos_1 = self.correct_solution(pos_1)
            pos_2 = self.correct_solution(pos_2)
            offspring.append(self.generate_empty_agent(pos_1))
            offspring.append(self.generate_empty_agent(pos_2))
        return offspring

    def competition_operator(self, pop_sorted, non_elite_indices, ):
        """
        Generate ideas using the competition operator, Eq. (6).
        """
        if self.n_comp < 2 or len(non_elite_indices) < 2:
            return []
        offspring = []
        # Inverse roulette: weaker non-elites receive larger probabilities.
        weights = self.get_rank_weights(len(non_elite_indices), reverse=True)
        for _ in range(self.n_comp // 2):
            # Select two distinct non-elites.
            first_slot = self.roulette_index(weights)
            valid_slots = np.delete(np.arange(len(non_elite_indices)), first_slot)
            valid_weights = weights[valid_slots]
            valid_weights = valid_weights / np.sum(valid_weights)
            second_slot = int(self.generator.choice(valid_slots, p=valid_weights))
            idx_1 = non_elite_indices[first_slot]
            idx_2 = non_elite_indices[second_slot]
            agent_1 = pop_sorted[idx_1]
            agent_2 = pop_sorted[idx_2]
            # ----------------------------------------------------------
            # Compare the two selected competitors.
            # x1 = better non-elite
            # x2 = worse non-elite to be modified.
            if self.compare_target(agent_1.target, agent_2.target, self.problem.minmax):
                better = agent_1
            else:
                better = agent_2
            # ----------------------------------------------------------
            # Eq. (6)
            # delta in [0, 1]
            # phi   in [0, 0.25]
            # Paper recommends updating all ideology factors.
            delta = self.generator.random(self.problem.n_dims)
            phi = (0.25 * self.generator.random(self.problem.n_dims))
            pos_new = (delta * better.solution + phi * self.g_best.solution)
            pos_new = self.correct_solution(pos_new)
            offspring.append(self.generate_empty_agent(pos_new))
        return offspring

    def evolve(self, epoch):
        """
        The main operations of CHOA.

        Args:
            epoch (int): The current iteration.
        """
        # --------------------------------------------------------------
        # Step 3: Ranking
        pop_sorted, groups = self.get_rank_groups(self.pop)
        elite_indices, non_elite_indices, cluster_of = self.get_elite_and_non_elite_indices(groups)
        # --------------------------------------------------------------
        # Step 4.1: Combination operator
        pop_comb = self.combination_operator(pop_sorted, elite_indices, cluster_of)
        # --------------------------------------------------------------
        # Step 4.2: Competition operator
        pop_comp = self.competition_operator(pop_sorted, non_elite_indices)
        # Evaluate all newly generated ideas.
        pop_new = pop_comb + pop_comp
        if self.mode not in self.AVAILABLE_MODES:
            for agent in pop_new:
                agent.target = self.get_target(agent.solution)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        # --------------------------------------------------------------
        # Step 4.3: Acceptance
        # The paper explicitly describes elitist truncation: append new ideas to the current population,
        # rank, and retain the best Npop ideas for the next generation.
        pop_all = self.pop + pop_new
        self.pop = self.get_sorted_population(pop_all, self.problem.minmax)[0][:self.pop_size]
