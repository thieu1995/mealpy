#!/usr/bin/env python
# Created by "Thieu" at 05:12, 14/09/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent
from mealpy.utils.opt_info import OptInfo


class OriginalCompSO(Optimizer):
    """
    The original version of: Competitive Swarm Optimizer (CompSO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of particles in the swarm, in range [4, 10000].
        The population size must be even because particles are randomly
        paired for pairwise competitions. Default is 100.
    phi : float
        Social factor controlling the influence of the swarm mean position
        in Eq. (6), in range [0.0, 10.0]. Default is 0.0.

    Note
    ----
    CompSO randomly partitions the swarm into pairs at each generation. In each pair, the better particle
    becomes the winner and is passed unchanged to the next generation, while the loser updates
    its velocity and position according to Eqs. (6) and (7).

    Only half of the particles are updated and re-evaluated in each generation,
    so CompSO requires approximately `pop_size / 2` new objective-function evaluations per iteration.

    The original paper does not explicitly specify the initial velocity distribution or the boundary-handling rule.
    This implementation initializes velocities to zero and uses Mealpy's standard solution correction after
    position updates. The global mean position of the whole swarm is used, corresponding to the
    default CompSO formulation studied in the paper.

    References
    ----------
    1. Cheng, R., & Jin, Y. (2015).
       A Competitive Swarm Optimizer for Large Scale Optimization.
       IEEE Transactions on Cybernetics, 45(2), 191-204. https://doi.org/10.1109/TCYB.2014.2322602

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, CompSO
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
    >>> model = CompSO.OriginalCompSO(epoch=1000, pop_size=100, phi=0.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Competitive Swarm Optimizer", year=2015, difficulty="easy", kind="original")

    def __init__(self, epoch: int = 10000, pop_size: int = 100, phi: float = 0.0, **kwargs: object) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 10000.
            pop_size (int): Number of particles in the swarm, default = 100.
            phi (float): Social factor in Eq. (6), default = 0.0.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [4, 10000])
        self.phi = self.validator.check_float("phi", phi, [0.0, 10.0])
        if self.pop_size % 2 != 0:
            raise ValueError("'pop_size' must be even because CompSO performs pairwise "
                             "competitions between all particles.")
        self.set_parameters(["epoch", "pop_size", "phi"])
        self.sort_flag = False

    def generate_empty_agent(self, solution: np.ndarray = None, ) -> Agent:
        """
        Generate an empty CompSO particle.

        The paper does not explicitly define an initialization equation
        for particle velocities, so zero initial velocity is used here.
        """
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = np.zeros(self.problem.n_dims)
        return Agent(solution=solution, velocity=velocity)

    def evolve(self, epoch):
        """
        The main operations of the Competitive Swarm Optimizer.

        Args:
            epoch (int): The current iteration.
        """
        # The global mean position X_bar(t) in Eq. (6).
        mean_position = np.mean(np.array([agent.solution for agent in self.pop]), axis=0)
        # Randomly assign all particles into pop_size / 2 pairs.
        indices = self.generator.permutation(self.pop_size)
        pop_new = []
        for idx in range(0, self.pop_size, 2):
            idx1 = indices[idx]
            idx2 = indices[idx + 1]
            agent1 = self.pop[idx1]
            agent2 = self.pop[idx2]
            # Determine winner and loser based on current fitness.
            if self.compare_target(agent1.target, agent2.target, self.problem.minmax):
                winner = agent1
                loser = agent2
            else:
                winner = agent2
                loser = agent1
            # Winner is passed directly to the next generation.
            pop_new.append(winner.copy())
            # Eq. (6)
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            r3 = self.generator.random(self.problem.n_dims)

            velocity = r1 * loser.velocity + r2 * (winner.solution - loser.solution) + self.phi * r3 * (mean_position - loser.solution)
            # Eq. (7)
            pos_new = loser.solution + velocity
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            agent.velocity = velocity
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        self.pop = pop_new
