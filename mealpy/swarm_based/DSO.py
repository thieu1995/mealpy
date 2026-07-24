#!/usr/bin/env python
# Created by "Thieu" at 10:19, 23/07/2026 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo


class OriginalDSO(Optimizer):
    """
    The original version of: Dove Swarm Optimization (DSO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.
    lamda : float
        Satiety decay rate, in range [0.0, 1.0]. Default is 0.9.
    eta : float
        Scaled-step for position updates, in range [-100.0, 100.0]. Default is 0.2.

    Note
    ----
    1. This algorithm is of low quality as it lacks any novel or specialized operators, making it highly prone to getting trapped in local optima.
    2. Relying on fitness and distance within the operator is not an effective approach to improving algorithmic performance.

    References
    ~~~~~~~~~~
    1. Su, M. C., Chen, J. H., Utami, A. M., Lin, S. C., & Wei, H. H. (2022).
       Dove swarm optimization algorithm. IEEE Access, 10, 46690-46696.
       https://doi.org/10.1109/ACCESS.2022.3170112

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, DSO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = DSO.OriginalDSO(epoch=1000, pop_size=50, lamda=0.9, eta=0.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Dove Swarm Optimization", year=2022, difficulty="easy", kind="original")

    def __init__(self, epoch: int = 10000, pop_size: int = 100, lamda: float=0.9, eta: float=0.2, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.lamda = self.validator.check_float("lamda", lamda, [0, 1.0])
        self.eta = self.validator.check_float("eta", eta, [-100.0, 100.0])
        self.set_parameters(["epoch", "pop_size", "lamda", "eta"])
        self.sort_flag = False
        self.satiety = None

    def before_main_loop(self):
        self.satiety = np.zeros(self.pop_size)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Compute fitness for all doves
        fits = np.array([agent.target.fitness for agent in self.pop])
        _, best_fit, _ = self.get_special_fitness(self.pop, self.problem.minmax)

        # Update each dove's satiety degree
        if best_fit != 0:
            if self.problem.minmax == "min":
                self.satiety = self.lamda * self.satiety + np.exp(best_fit - fits)
            else:
                self.satiety = self.lamda * self.satiety + np.exp(fits - best_fit)
        else:
            self.satiety = self.lamda * self.satiety + 1.0

        # Select the most satisfied dove (highest degree of satiety)
        d_s = np.argmax(self.satiety)
        w_ds = self.pop[d_s].solution
        s_ds = self.satiety[d_s]

        # Compute Euclidean distances between each dove and the most satisfied dove
        pos_list = np.array([agent.solution for agent in self.pop])
        distances = np.linalg.norm(pos_list - w_ds, axis=1)
        max_distance = np.max(distances)

        # Avoid division by zero in convergence scenarios
        if max_distance != 0 or s_ds != 0:
            # Update each dove's position vector (Eq. 8 & 9)
            for idx in range(self.pop_size):
                # Eq. 9: Calculate social impact coefficient
                beta_j = ((s_ds - self.satiety[idx]) / s_ds) * (1.0 - (distances[idx] / max_distance))
                # Eq. 8: Update position
                pos_new = self.pop[idx].solution + self.eta * beta_j * (w_ds - self.pop[idx].solution)
                # Boundary constraints mapping
                pos_new = self.correct_solution(pos_new)
                self.pop[idx].solution = pos_new
                if self.mode not in self.AVAILABLE_MODES:
                    self.pop[idx].target = self.get_target(pos_new)
            # Update fitness in parallel
            if self.mode in self.AVAILABLE_MODES:
                self.pop = self.update_target_for_population(self.pop)
