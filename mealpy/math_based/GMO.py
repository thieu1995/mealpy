#!/usr/bin/env python
# Created by "Thieu" at 10:04, 14/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent
from mealpy.utils.opt_info import OptInfo


class OriginalGMO(Optimizer):
    """
    The original version of: Geometric Mean Optimizer (GMO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 1000.
    pop_size : int
        Number of search agents, in range [5, 10000]. Default is 50.

    Note
    ----
    GMO has no algorithm-specific control parameter to tune. The defaults `pop_size=50` and
    `epoch=1000` correspond to the main experimental configuration used in the original paper.

    Each agent maintains its personal best-so-far solution. The Dual-Fitness Index (DFI) is computed
    from the fuzzy membership values of the opposite personal-best agents and simultaneously
    reflects fitness and diversity. The number of elite personal-best agents decreases linearly from the
    population size to 2 over the optimization process.

    The paper recommends omitting the small denominator constant in the guide
    calculation when no prior information about the problem is available.
    Therefore, this implementation follows the default setting `epsilon = 0`
    used in the authors' reference code.

    The Matlab code initializes and limits velocity to 10% of the search-space range. When a position exceeds
    a bound, it is clipped to that bound and the corresponding velocity component reverses direction.
    GMO evaluates one new solution per agent per iteration. Thus, excluding initialization, it requires
    approximately `pop_size` objective-function evaluations per iteration.

    References
    ----------
    1. Rezaei, F., Safavi, H.R., Abd Elaziz, M., & Mirjalili, S. (2023).
       GMO: geometric mean optimizer for solving engineering problems. Soft Computing, 27, 10571-10606.
       https://doi.org/10.1007/s00500-023-08202-z

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, GMO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function,
    >>> }
    >>> model = GMO.OriginalGMO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Geometric Mean Optimizer", year=2023, difficulty="medium", kind="original")

    def __init__(self, epoch: int = 1000, pop_size: int = 50, **kwargs: object, ) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 1000.
            pop_size (int): Number of search agents, default = 50.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def initialize_variables(self):
        """
        Initialize velocity limits used in the authors' reference code.
        """
        self.velocity_bound = 0.1 * (self.problem.ub - self.problem.lb)

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        """
        Generate a GMO search agent with position and velocity.
        """
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(-self.velocity_bound, self.velocity_bound)
        return Agent(solution=solution, velocity=velocity, local_solution=solution.copy(), local_target=None)

    def before_main_loop(self):
        """
        Initialize personal-best states.
        """
        for agent in self.pop:
            agent.local_solution = agent.solution.copy()
            agent.local_target = agent.target.copy()

    def calculate_guides(self, epoch):
        """
        Calculate the unique guide of every search agent according
        to Eqs. (2)-(5).
        """
        # Personal-best objective values.
        fit_pbest = np.asarray([agent.local_target.fitness for agent in self.pop], dtype=float)

        # Mean and standard deviation used by Eq. (2).
        fit_mean = np.mean(fit_pbest)
        fit_std = np.std(fit_pbest, ddof=1)

        # Eq. (2): fuzzy membership values.
        # MATLAB std() uses sample standard deviation, hence ddof=1.
        denom = fit_std * np.sqrt(np.e)
        if denom == 0:
            denom += self.EPSILON
        mf = 1.0 / (1.0 + np.exp((-4.0 / denom) * (fit_pbest - fit_mean)))

        # Eq. (3): DFI_i is the product of the membership values
        # of all opposite personal-best agents.
        dfi = np.empty(self.pop_size)
        for idx in range(self.pop_size):
            mask = np.arange(self.pop_size) != idx
            dfi[idx] = np.prod(mf[mask])

        # Number of elite agents decreases linearly from N to 2.
        # This follows the authors' implementation: kbest = N - (N - 2) * t / t_max
        n_best = int(np.round(self.pop_size - (self.pop_size - 2) * epoch / self.epoch))
        n_best = max(2, n_best)
        # Sort DFI values in descending order.
        elite_indices = np.argsort(-dfi)[:n_best]
        elite_dfi = dfi[elite_indices]

        # Denominator of Eq. (5).
        denominator = np.sum(elite_dfi)
        if denominator == 0:
            denominator += self.EPSILON
        guides = np.zeros((self.pop_size, self.problem.n_dims))
        # Eq. (5): each search agent has its own unique guide.
        for idx in range(self.pop_size):
            for elite_pos, elite_idx in enumerate(elite_indices):
                if elite_idx != idx:
                    guides[idx] += (elite_dfi[elite_pos] / denominator * self.pop[elite_idx].local_solution)
        return guides

    def evolve(self, epoch):
        """
        The main operations of the Geometric Mean Optimizer.

        Args:
            epoch (int): The current iteration.
        """
        # First update the personal-best solutions from the current
        # population before computing DFI and guides.
        for idx in range(self.pop_size):
            if self.compare_target(self.pop[idx].target, self.pop[idx].local_target, self.problem.minmax):
                self.pop[idx].local_solution = (self.pop[idx].solution.copy())
                self.pop[idx].local_target = (self.pop[idx].target.copy())

        # Eqs. (2)-(5)
        guides = self.calculate_guides(epoch)
        # Standard deviation of personal-best positions used in Eq. (6).
        pbest_positions = np.asarray([agent.local_solution for agent in self.pop])
        std_dims = np.std(pbest_positions, axis=0, ddof=1)
        max_std = np.max(std_dims)

        # Eq. (9)
        w = 1.0 - epoch / self.epoch
        pop_new = []
        for idx in range(self.pop_size):
            # Eq. (6): Gaussian mutation of the unique guide.
            guide_mutated = (guides[idx] + w * self.generator.normal(0.0, 1.0, self.problem.n_dims, ) * (max_std - std_dims))

            # Eq. (7)
            u = (1.0 + (2.0 * self.generator.random(self.problem.n_dims) - 1.0) * w)
            velocity = (w * self.pop[idx].velocity + u * (guide_mutated - self.pop[idx].solution))
            # Velocity limiting used by the authors' implementation.
            velocity = np.clip(velocity, -self.velocity_bound, self.velocity_bound, )

            # Eq. (8)
            raw_position = (self.pop[idx].solution + velocity)
            # Detect dimensions exceeding the position boundaries.
            lower_mask = raw_position < self.problem.lb
            upper_mask = raw_position > self.problem.ub
            outside_mask = lower_mask | upper_mask

            # Reference implementation clips the position.
            pos_new = np.clip(raw_position, self.problem.lb, self.problem.ub)
            # The corresponding velocity component reverses direction.
            velocity = np.where(outside_mask, -velocity, velocity)
            agent = self.generate_empty_agent(pos_new)

            # Preserve the velocity obtained from Eq. (7).
            agent.velocity = velocity
            # Preserve the parent's personal-best state.
            agent.local_solution = self.pop[idx].local_solution.copy()
            agent.local_target = (self.pop[idx].local_target.copy())
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        # GMO does not use greedy survivor selection.
        self.pop = pop_new
