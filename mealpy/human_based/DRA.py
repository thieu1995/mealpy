#!/usr/bin/env python
# Created by "Thieu" at 14:26, 23/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo, ScientificConcern


class OriginalDRA(Optimizer):
    """
    The original version of: Divine Religions Algorithm (DRA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000].
        The paper uses 500 iterations in the benchmark experiments. Default is 500.
    pop_size : int
        Number of followers in the belief profile, in range [5, 10000].
        The paper uses 50 followers in the benchmark experiments. Default is 50.
    n_groups : int
        Number of religious communities/missionaries.
        The paper treats this quantity as the input `M` but does not provide
        a numerical default. The released MATLAB code uses 5 groups. Default is 5.
    bpsp : float
        Belief Profile Selection Probability (BPSP), in [0, 1].
        The paper uses BPSP = 0.5. Default is 0.5.
    rp : float
        Reward or Penalty Probability (RP), in [0, 1].
        The paper uses RP = 0.2. Default is 0.2.

    Link
    ----
    1. Paper: https://doi.org/10.1007/s10586-024-04954-x
    2. Matlab code: https://nl.mathworks.com/matlabcentral/fileexchange/180492-divine-religions-algorithm-dra

    Notes
    -----
    This implementation follows the mathematical formulation, descriptive
    text, flowchart, and Algorithm 1 in the paper. The released MATLAB code
    differs substantially from the published formulation:

    - The paper treats `M`, the number of communities, as an input without
      giving a numerical default. The MATLAB code fixes `M = 5`; this value
      is used here only as the default.
    - The paper describes beliefs as normalized to [0, 1], whereas its
      benchmark problems and the MATLAB code operate directly within the
      problem bounds. Mealpy therefore uses the actual problem bounds.
    - Eq. (4) defines fitness as the sum of belief attributes, although the
      experiments and MATLAB code evaluate the supplied objective function.
      This implementation uses the actual optimization objective.
    - Eqs. (8), (17), (19), and (22) use maximization-oriented terminology,
      while the paper also states that populations are sorted in ascending
      objective value and all benchmark experiments are minimization
      problems. Mealpy's `minmax` semantics are therefore used consistently.
    - Algorithm 1 places population/community initialization inside its
      repetition block, while the flowchart, descriptive text, and MATLAB
      code initialize them only once. Initialization is performed once here.
    - The paper does not specify how followers are initially assigned to
      communities. Following the released source, the best `n_groups`
      followers initialize the missionaries and the remaining followers are
      randomly assigned to communities.
    - Eq. (11) copies a transformed attribute from the most prominent
      follower. The MATLAB code instead copies an arbitrary attribute from
      an arbitrary population member to another arbitrary attribute and
      omits the `(rand - cos(rand))` factor.
    - Eq. (12) contains `t / (2 * MaxIter)`. The MATLAB expression
      `it / MaxIteration * 2` evaluates as `2*it/MaxIteration`, producing a
      different MP schedule that becomes negative after half the run.
      This implementation follows Eq. (12).
    - The prose states that exactly one of the miracle and proselytism
      operators is selected. Algorithm 1 is ambiguous on this point.
      This implementation follows the prose and Eqs. (13)-(14).
    - The MATLAB code applies the miracle/proselytism operator to every
      existing population member, whereas the paper formulates these
      operators for the newly attracted follower. The paper is followed.
    - The second branch of Eq. (13) contains
      `round(1**rand * c)`. The MATLAB code evaluates
      `round(1**rand) * c`, which is mathematically different and reduces
      that branch to an unchanged position.
    - The first branch of Eq. (14) contains `sin(pi * rand)`. The MATLAB
      source instead uses `sin(sin(3.14 * rand))`.
    - The second branch of Eq. (14) is replaced completely in the MATLAB
      code by `Leader * (rand - cos(rand))`.
    - Eq. (14) uses the mean belief value. Its second branch consequently
      produces a scalar although a belief vector is required. The scalar is
      broadcast to all dimensions here; the paper gives no alternative rule.
    - Eq. (16) and Algorithm 1 indicate that one randomly selected belief
      attribute is rewarded or penalized. The MATLAB source modifies the
      entire solution vector.
    - Eq. (15) is titled as the fitness calculation of a newly entered
      follower, but it does not define a fitness equation; it only describes
      adding penalized followers to a forbidden list. The paper gives no
      lifecycle or operational rules for removing or using this list, so no
      persistent forbidden-list mechanism is invented here.
    - The paper states that a new follower replaces the weakest follower or
      a duplicated profile, but it does not define duplicate detection.
      Only replacement of the weakest follower is implemented.
    - The MATLAB group structure is almost unused after initialization.
      Its replacement step operates only on the group stored in the stale
      loop variable `k` and swaps members without the comparison in Eq. (22).
      Here, missionaries are always updated as the best member of each group.
    - In several MATLAB stages, a belief vector is changed unconditionally
      while its stored cost is updated only if the new objective improves.
      This makes the stored solution and fitness inconsistent. This behavior
      is not reproduced.
    - The MATLAB reward/penalty block evaluates `BP(index)` but updates
      `BP(i).Cost`, where `i` is the stale index left by the previous loop.
    - The MATLAB merge decision reuses `NewFobj` from the reward/penalty
      candidate to decide whether to insert `NewFollower`, so the compared
      fitness and inserted solution can refer to different individuals.
    - The MATLAB population is sorted only during initialization, yet its
      reported solution is always `BP(1)`, which need not remain the best
      individual after subsequent updates.
    - The MATLAB variable `PP = 0.9` is never used and is not listed among
      the DRA hyperparameters in the paper; it is therefore omitted.

    Eq. (13) contains `cos(pi / 2)`, which mathematically collapses the
    corresponding miracle candidate to approximately zero. This
    implementation follows the published equation literally.

    Under the paper-oriented interpretation used here, one newly attracted
    follower and one reward/penalty candidate are evaluated per iteration,
    giving approximately `pop_size + 2 * epoch` objective evaluations.
    The released MATLAB code instead evaluates approximately
    `pop_size + 3` new solutions per iteration.

    References
    ----------
    1. Mozhdehi, A. T., Khodadadi, N., Aboutalebi, M., El-kenawy, E. M.,
       Hussien, A. G., Zhao, W., Nadimi-Shahraki, M. H., & Mirjalili, S.
       (2025).
       Divine Religions Algorithm: a novel social-inspired metaheuristic
       algorithm for engineering and continuous optimization problems.
       Cluster Computing, 28, 253.
       https://doi.org/10.1007/s10586-024-04954-x

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, DRA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution ** 2)
    >>>
    >>> problem = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>> model = DRA.OriginalDRA(epoch=500, pop_size=50, n_groups=5, bpsp=0.5, rp=0.2)
    >>> g_best = model.solve(problem)
    >>> print(g_best.target.fitness)
    """

    OPT_INFO = OptInfo(name="Divine Religions Algorithm", year=2025, difficulty="hard", kind="original",
                       scientific_status="questionable",
                       concerns=(
                           ScientificConcern.QUESTIONABLE_MATH, ScientificConcern.INCORRECT_EQUATIONS,
                           ScientificConcern.POOR_REPRODUCIBILITY, ScientificConcern.FABRICATED_RESULTS,
                           ScientificConcern.AMBIGUOUS_METHODOLOGY, ScientificConcern.CODE_PSEUDOCODE_MISMATCH,
                       ))

    def __init__(self, epoch: int = 500, pop_size: int = 50, n_groups: int = 5, bpsp: float = 0.5, rp: float = 0.2, **kwargs: object) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 500.
            pop_size (int): Population size, default = 50.
            n_groups (int): Number of communities, default = 5.
            bpsp (float): Belief Profile Selection Probability, default = 0.5.
            rp (float): Reward or Penalty Probability, default = 0.2.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_groups = self.validator.check_int("n_groups", n_groups, [1, self.pop_size - 1])
        self.bpsp = self.validator.check_float("bpsp", bpsp, [0.0, 1.0])
        self.rp = self.validator.check_float("rp", rp, [0.0, 1.0])
        self.set_parameters(["epoch", "pop_size", "n_groups", "bpsp", "rp"])
        self.sort_flag = False
        self.is_parallelizable = False
        self.group_ids = None
        self.missionary_indices = None

    def before_main_loop(self):
        """
        Create the religious communities.

        The paper does not define the initial assignment procedure.
        The released MATLAB implementation is therefore followed:
        the best `n_groups` members initialize the missionaries and
        the remaining followers are randomly distributed among them.
        """
        self.pop, _ = self.get_sorted_population(self.pop, self.problem.minmax)
        self.group_ids = np.empty(self.pop_size, dtype=int)
        # One initial missionary per community.
        self.group_ids[:self.n_groups] = np.arange(self.n_groups)
        # Randomly distribute the remaining followers.
        self.group_ids[self.n_groups:] = (self.generator.integers(0, self.n_groups, size=self.pop_size - self.n_groups))
        self.update_missionaries()

    def get_best_index(self, indices):
        """
        Return the best population index according to the problem direction.
        """
        indices = np.asarray(indices, dtype=int)
        fitness = np.asarray([self.pop[idx].target.fitness for idx in indices])
        if self.problem.minmax == "min":
            return int(indices[np.argmin(fitness)])
        return int(indices[np.argmax(fitness)])

    def get_worst_index(self, indices):
        """
        Return the worst population index according to the problem direction.
        """
        indices = np.asarray(indices, dtype=int)
        fitness = np.asarray([self.pop[idx].target.fitness for idx in indices])
        if self.problem.minmax == "min":
            return int(indices[np.argmax(fitness)])
        return int(indices[np.argmin(fitness)])

    def update_missionaries(self):
        """
        Identify the most prominent follower in every community.

        This corresponds to Eqs. (20)-(22). Since a missionary is a role,
        changing the missionary does not alter the numerical solution; the
        best current member of each community simply receives that role.
        """
        missionaries = np.empty(self.n_groups, dtype=int, )
        for group in range(self.n_groups):
            indices = np.flatnonzero(self.group_ids == group)
            missionaries[group] = self.get_best_index(indices)
        self.missionary_indices = missionaries

    def get_miracle_position(self, position):
        """
        Apply the miracle operator according to Eq. (13).
        """
        pos_new = position.copy()
        if self.generator.random() <= 0.5:
            r1 = self.generator.random()
            r2 = self.generator.random()
            # Eq. (13), first branch.
            pos_new = (pos_new * np.cos(np.pi / 2.0) * (r1 - np.cos(r2)))
        else:
            r1 = self.generator.random()
            r2 = self.generator.random()
            # Eq. (13), second branch.
            # The paper places the belief inside round(): round(1**rand * c).
            rounded = np.round(np.power(1.0, r2) * pos_new)
            pos_new = (pos_new + r1 * (pos_new - rounded))
        return pos_new

    def get_proselytism_position(self, position, mp):
        """
        Apply the proselytism operator according to Eq. (14).
        """
        pos_new = position.copy()
        mean_belief = np.mean(pos_new)
        if self.generator.random() > (1.0 - mp):
            r1 = self.generator.random()
            r2 = self.generator.random()
            # Eq. (14), first branch.
            term = (mean_belief * (1.0 - mp) + (1.0 - mean_belief) - (r1 - 4.0 * np.sin(np.pi * r2)))
            pos_new = (0.01 * pos_new + term)
        else:
            r1 = self.generator.random()
            r2 = self.generator.random()
            r3 = self.generator.random()
            randn = self.generator.normal()
            # Eq. (14), second branch.
            # The printed equation returns a scalar although the belief
            # profile is d-dimensional. The scalar is therefore broadcast to every belief attribute.
            value = (r1 - 2.0 * r2 - mean_belief) * (2.0 * r3 - (1.0 - mp) * randn)
            pos_new = np.full(self.problem.n_dims, value, dtype=float)
        return pos_new

    def apply_reward_penalty(self):
        """
        Apply the reward/penalty operator from Eq. (16).

        The paper specifies changing one randomly selected belief attribute.
        """
        # Followers are non-missionary members.
        all_indices = np.arange(self.pop_size)
        follower_indices = np.setdiff1d(all_indices, self.missionary_indices, assume_unique=False)
        if follower_indices.size == 0:
            return
        idx = int(self.generator.choice(follower_indices))
        dim = int(self.generator.integers(self.problem.n_dims))
        pos_new = self.pop[idx].solution.copy()
        randn = self.generator.normal()
        if self.generator.random() >= self.rp:
            # Eq. (16): Reward operator.
            pos_new[dim] *= 1.0 - randn
        else:
            # Eq. (16): Penalty operator.
            pos_new[dim] *= 1.0 + randn
        pos_new = self.correct_solution(pos_new)
        agent = self.generate_agent(pos_new)
        # The paper does not define a greedy acceptance rule for the
        # reward/penalty operation. The transformed follower is therefore accepted directly.
        self.pop[idx] = agent

    def evolve(self, epoch):
        """
        The main operations of DRA.

        Args:
            epoch (int): The current iteration.
        """
        # --------------------------------------------------------------
        # Identify the most prominent follower -- Eqs. (8)-(9)
        leader, leader_idx = self.get_best_agent(self.pop, self.problem.minmax)
        # The newly attracted follower is associated with the community
        # containing the current most prominent follower.
        target_group = int(self.group_ids[leader_idx])

        # --------------------------------------------------------------
        # Create a new follower.
        pos_new = self.problem.generate_solution(encoded=True)

        # --------------------------------------------------------------
        # Selection operator -- Eqs. (10)-(11)
        if self.generator.random() <= self.bpsp:
            dim = int(self.generator.integers(self.problem.n_dims))
            r1 = self.generator.random()
            r2 = self.generator.random()
            pos_new[dim] = (leader.solution[dim] * (r1 - np.cos(r2)))
        # --------------------------------------------------------------
        # Miracle probability -- Eq. (12)
        # Paper: MP = rand * (1 - t / (2 * MaxIter)) * rand
        mp = (self.generator.random() * (1.0 - epoch / (2.0 * self.epoch)) * self.generator.random())

        # --------------------------------------------------------------
        # Exactly one of the miracle/proselytism operators.
        if self.generator.random() <= mp:
            pos_new = self.get_miracle_position(pos_new)
        else:
            pos_new = self.get_proselytism_position(pos_new, mp)
        pos_new = self.correct_solution(pos_new)
        new_follower = self.generate_agent(pos_new)
        # Reward or penalty operator -- Eq. (16)
        self.apply_reward_penalty()

        # --------------------------------------------------------------
        # Update the follower's belief profile -- Eqs. (17)-(19)
        # The paper additionally mentions replacement of duplicated
        # profiles but never defines a duplicate-detection rule.
        group_indices = np.flatnonzero(self.group_ids == target_group)
        worst_idx = self.get_worst_index(group_indices)
        if self.compare_target(new_follower.target, self.pop[worst_idx].target, self.problem.minmax):
            self.pop[worst_idx] = new_follower
            self.group_ids[worst_idx] = target_group

        # --------------------------------------------------------------
        # Replacement operator -- Eqs. (20)-(22)
        self.update_missionaries()
