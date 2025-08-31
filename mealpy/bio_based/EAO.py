#!/usr/bin/env python
# Created by "Thieu" at 23:50, 28/08/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalEAO(Optimizer):
    """
    The original version of: Enzyme Action Optimizer (EAO)

    Notes:
        + This algorithm used 3 fitness calculations for each update enzyme. Therefor, it is slower 3 times than other algorithms.

    Links:
        1. https://mathworks.com/matlabcentral/fileexchange/170296-enzyme-action-optimizer-a-novel-bio-inspired-optimization

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, EAO
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
    >>> model = EAO.OriginalEAO(epoch=1000, pop_size=50, p_m=0.01, n_elites=2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rodan, A., Al-Tamimi, A. K., Al-Alnemer, L., Mirjalili, S., & Tiňo, P. (2025).
    Enzyme action optimizer: a novel bio-inspired optimization algorithm. The Journal of Supercomputing, 81(5), 686.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, ec: float = 0.1, **kwargs: object) -> None:
        """
        Initialize the algorithm components.

        Args:
            epoch: Maximum number of iterations, default = 10000
            pop_size: Number of population size, default = 100
            ec: Enzyme Concentration, default=0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.ec = self.validator.check_float("ec", ec, [0., 100])
        self.set_parameters(["epoch", "pop_size", "ec"])
        self.sort_flag = False
        self.is_parallelizable = False

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch: The current iteration
        """
        # Adaptation Factor - tăng dần theo thời gian
        AF = np.sqrt(epoch/ self.epoch)

        # Handle each enzyme
        for idx in range(self.pop_size):
            # 1. Update FirstSubstratePosition
            r1 = self.generator.random(size=self.problem.n_dims)
            pos1 = (self.g_best.solution - self.pop[idx].solution) + r1 * np.sin(AF * self.pop[idx].solution)
            pos1 = self.correct_solution(pos1)
            agent1 = self.generate_agent(pos1)

            # 2. Select 2 randoms
            j1, j2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), size=2, replace=False)

            ## Candidate A: vector-valued random factors
            scA1 = self.ec + (1 - self.ec) * self.generator.random(size=self.problem.n_dims)
            exA = AF * (self.ec + (1 - self.ec) * self.generator.random(size=self.problem.n_dims))
            posA = self.pop[idx].solution + scA1 * (self.pop[j1].solution - self.pop[j2].solution) + exA * (self.g_best.solution - self.pop[idx].solution)
            posA = self.correct_solution(posA)
            agentA = self.generate_agent(posA)

            ## Candidate B: scalar random factors
            scB1 = self.ec + (1 - self.ec) * self.generator.random()
            exB = AF * (self.ec + (1 - self.ec) * self.generator.random())
            posB = self.pop[idx].solution + scB1 * (self.pop[j1].solution - self.pop[j2].solution) + exB * (self.g_best.solution - self.pop[idx].solution)
            posB = self.correct_solution(posB)
            agentB = self.generate_agent(posB)

            pop_new = [self.pop[idx], agent1, agentA, agentB]
            self.pop[idx] = self.get_best_agent(pop_new, minmax=self.problem.minmax)
