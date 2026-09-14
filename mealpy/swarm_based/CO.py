#!/usr/bin/env python
# Created by "Thieu" at 09:39, 14/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo


class OriginalCO(Optimizer):
    """
    The original version of: Cheetah Optimizer (CO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of cheetahs in the population, in range [2, 10000]. The original experiments
        commonly use a small population, including `pop_size = 6`. Default is 6.
    m : int
        Number of cheetahs participating in the hunting process at each iteration, in range [2, pop_size].
        The original experimental configuration uses `m = 2`. Default is 2.

    Note
    ----
    CO updates only `m` randomly selected cheetahs at each iteration rather than the whole population.
    Therefore, its number of objective-function evaluations per iteration is approximately `m`,
    excluding occasional evaluations associated with the leave-prey strategy.

    The maximum hunting time is defined by the original algorithm as `T = 60 * ceil(n_dims / 10)`
    and is therefore not exposed as a user-defined parameter.

    The search, sit-and-wait, and attack strategies follow Eqs. (1)-(3) of the original paper.
    The step length of the leader is based on the variable range, while the other selected
    cheetahs use their distance from another selected cheetah.

    The paper describes returning home when the hunting time exceeds `T` and the leader has not improved
    during the hunting period. This implementation treats the absence of any leader improvement within
    the current hunting period as this condition and does not introduce
    an additional numerical stagnation threshold.

    References
    ----------
    1. Akbari, M.A., Zare, M., Azizipanah-Abarghooee, R., Mirjalili, S., & Deriche, M. (2022).
       The cheetah optimizer: a nature-inspired metaheuristic algorithm for large-scale optimization problems.
       Scientific Reports, 12, 10953. https://doi.org/10.1038/s41598-022-14338-z

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, CO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function,
    >>> }
    >>> model = CO.OriginalCO(epoch=10000, pop_size=6, m=2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Cheetah Optimizer", year=2022, difficulty="medium", kind="original")

    def __init__(self, epoch: int = 10000, pop_size: int = 6, m: int = 2, **kwargs: object) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 10000.
            pop_size (int): Number of cheetahs, default = 6.
            m (int): Number of selected cheetahs per iteration, default = 2.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [2, 10000])
        self.m = self.validator.check_int("m", m, [2, self.pop_size])
        self.set_parameters(["epoch", "pop_size", "m"])
        self.sort_flag = False
        # Population and leader states are updated sequentially.
        self.is_parallelizable = False

    def before_main_loop(self):
        """
        Initialize the home, leader, prey, and hunting-time states.
        """
        # Initial population corresponds to the home positions.
        self.home = [agent.copy() for agent in self.pop]

        # Current hunting leader.
        leader, _ = self.get_best_agent(self.pop, self.problem.minmax)
        self.leader = leader.copy()

        # Best prey found during the complete optimization process.
        self.prey = leader.copy()

        # Current hunting time.
        self.hunting_time = 0

        # Eq./Algorithm 1: maximum hunting time.
        self.max_hunting_time = (60 * int(np.ceil(self.problem.n_dims / 10.0)))

        # Tracks whether the leader improved during the current hunt.
        self.leader_improved = False

    def update_leader(self, agent):
        """
        Update the current hunting leader.
        """
        if self.compare_target(agent.target, self.leader.target, self.problem.minmax):
            self.leader = agent.copy()
            self.leader_improved = True

    def update_prey(self):
        """
        Update the global prey solution.
        """
        if self.compare_target(self.leader.target, self.prey.target, self.problem.minmax, ):
            self.prey = self.leader.copy()

    def go_back_home(self, selected_indices):
        """
        Apply the leave-prey and go-back-home strategy.

        Selected members return to their initial home positions, while one
        randomly selected member is placed at the best prey position.
        """
        for idx in selected_indices:
            self.pop[idx] = self.home[idx].copy()
        prey_idx = self.generator.choice(selected_indices)
        self.pop[prey_idx] = self.prey.copy()
        leader, _ = self.get_best_agent(self.pop, self.problem.minmax, )
        self.leader = leader.copy()
        self.hunting_time = 0
        self.leader_improved = False

    def evolve(self, epoch):
        """
        The main operations of the Cheetah Optimizer.

        Args:
            epoch (int): The current iteration.
        """
        selected_indices = self.generator.choice(self.pop_size, size=self.m, replace=False)
        for k, idx in enumerate(selected_indices):
            current = self.pop[idx].solution.copy()

            # The neighboring member is the next selected cheetah;
            # the last member uses the previous selected cheetah.
            if k == len(selected_indices) - 1:
                neighbor_idx = selected_indices[k - 1]
            else:
                neighbor_idx = selected_indices[k + 1]
            neighbor = self.pop[neighbor_idx].solution
            pos_new = current.copy()

            # Random variables are generated independently for all arrangements (dimensions).
            r_hat = self.generator.normal(0.0, 1.0, self.problem.n_dims)
            r = self.generator.normal(0.0, 1.0, self.problem.n_dims)

            # Turning factor in Eq. (3).
            r_check = (np.abs(r) ** np.exp(r / 2.0) * np.sin(2.0 * np.pi * r))
            # Interaction factor beta in Eq. (3).
            beta = neighbor - current
            # Step length alpha in Eq. (1).
            if k == 0:
                # The first selected cheetah acts as the leader.
                alpha = (0.001 * self.hunting_time / self.max_hunting_time * (self.problem.ub - self.problem.lb))
            else:
                alpha = (0.001 * self.hunting_time / self.max_hunting_time * np.abs(current - neighbor))

            # H = exp(2 * (1 - t / T)) * (2*r1 - 1)
            r1 = self.generator.random(self.problem.n_dims)
            h = (np.exp(2.0 * (1.0 - self.hunting_time / self.max_hunting_time)) * (2.0 * r1 - 1.0))
            r2 = self.generator.random(self.problem.n_dims)
            r3 = self.generator.random(self.problem.n_dims)
            r4 = 3.0 * self.generator.random(self.problem.n_dims)

            # According to the textual description in the paper:
            # r2 >= r3 -> sit-and-wait
            # r2 <  r3 -> attack/search
            # Within the latter case:
            # H >= r4 -> attack
            # H <  r4 -> search

            sit_mask = r2 >= r3
            active_mask = ~sit_mask
            attack_mask = active_mask & (h >= r4)
            search_mask = active_mask & (h < r4)

            # Eq. (1): Search strategy
            # X_new = X + r_hat^(-1) * alpha
            safe_r_hat = np.where(np.abs(r_hat) < self.EPSILON, np.where(r_hat >= 0.0, self.EPSILON, -self.EPSILON), r_hat)
            pos_search = (current + alpha / safe_r_hat)

            # Eq. (2): Sit-and-wait strategy
            pos_wait = current
            # Eq. (3): Attack strategy
            pos_attack = (self.prey.solution + r_check * beta)
            pos_new = np.where(search_mask, pos_search, pos_new)
            pos_new = np.where(attack_mask, pos_attack, pos_new)
            pos_new = np.where(sit_mask, pos_wait, pos_new)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            # Greedy replacement of the current member.
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax, ):
                self.pop[idx] = agent
                self.update_leader(agent)

        # One hunting-time unit is completed.
        self.hunting_time += 1

        # Update global prey solution.
        self.update_prey()

        # Leave the prey and return home when the hunting period has
        # expired without an improvement of the leader.
        if self.hunting_time > self.max_hunting_time and not self.leader_improved:
            self.go_back_home(selected_indices)

        elif self.hunting_time > self.max_hunting_time:
            # A successful hunting period starts a new hunt.
            self.hunting_time = 0
            self.leader_improved = False
