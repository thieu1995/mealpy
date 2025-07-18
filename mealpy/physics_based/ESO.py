#!/usr/bin/env python
# Created by "Thieu" at 07:03, 16/07/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalESO(Optimizer):
    """
    The original version of: Electrical Storm Optimization (ESO)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ESO
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
    >>> model = ESO.OriginalESO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Soto Calvo, Manuel, and Han Soo Lee. 2025. "Electrical Storm Optimization (ESO) Algorithm: Theoretical Foundations, Analysis, and Application to Engineering Problems" Machine Learning and Knowledge Extraction 7, no. 1: 24. https://doi.org/10.3390/make7010024
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.is_parallelizable = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Calculate storm parameters
        # Calculate field resistance based on population spread
        pos_pop = np.array([agent.solution for agent in self.pop])
        mean_pos = np.mean(pos_pop, axis=0)
        std_pos = np.sqrt(np.mean(np.sum((pos_pop - mean_pos)**2, axis=1)))
        # std_pos = np.std(pos_pop, axis=0)
        peak_to_peak = np.max(np.max(pos_pop, axis=0) - np.min(pos_pop, axis=0))
        # Field resistance
        if peak_to_peak <= 0:
            resistance = 0
            ionized_pop = []
        else:
            resistance = std_pos / peak_to_peak
            # Identify ionized areas (promising regions)
            # Calculate percentile threshold
            percentile_threshold = (resistance / 2) * 100
            # Find solutions better than percentile
            fits = np.array([agent.target.fitness for agent in self.pop])
            fitness_percentile = np.percentile(fits, percentile_threshold)
            ionized_indices = np.where(fits <= fitness_percentile)[0]
            ionized_pop = [self.pop[idx] for idx in ionized_indices]

        # Calculate field conductivity using logistic function
        if resistance <= 0:
            fc = 1.0
        else:
            # Beta calculation (logistic function)
            try:
                exp_term = np.exp(resistance) / resistance
                log_term = np.log(1. - resistance) if resistance < 1 else 0
                beta = 1. / (1. + np.exp(-exp_term) * (resistance - abs(log_term)))
            except (OverflowError, ValueError):
                beta = 0.5
            try:
                fc = (np.exp(resistance) + np.exp(1 - resistance) * abs(np.log(resistance)) * beta)
            except (OverflowError, ValueError):
                fc = 1.0

        # Calculate field intensity using logistic function
        if resistance <= 0:
            fi = fc
        else:
            # Gamma calculation
            try:
                exp_term = np.exp(resistance) / resistance
                iter_ratio = epoch / self.epoch
                log_term = np.log(1 - iter_ratio) if iter_ratio < 1 else 0
                gama = 1 / (1 + np.exp(-exp_term * (resistance - abs(log_term))))
            except (OverflowError, ValueError):
                gama = 0.5
            fi = fc * gama

        # Calculate storm power
        if fc > 0:
            storm_power = (resistance * fi) / fc
        else:
            storm_power = 0

        # Update each lighting agent
        pop_new = []
        for idx in range(0, self.pop_size):
            # Initialize new lighting position
            if idx == 0 or len(ionized_pop) == 0:
                agent = self.generate_empty_agent()
            else:
                # Initialize near ionized areas
                alpha = ionized_pop[self.generator.integers(0, len(ionized_pop))]
                perturbation = self.generator.normal(loc=0, scale=storm_power, size=self.problem.n_dims)
                pos_new = alpha.solution + perturbation
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
            agent.target = self.get_target(agent.solution)

            ## Branching and propagation
            # Simulate branching and propagation of lightning
            in_ionized = False
            for alpha in ionized_pop:
                if np.linalg.norm(agent.solution - alpha.solution) < 0.1:
                    in_ionized = True
                    break

            if in_ionized:
                # Propagate within ionized area
                pos_new = agent.solution * storm_power
            else:
                # Propagate towards ionized areas
                if len(ionized_pop) > 0:
                    # Average position of ionized areas
                    avg_ionized = np.mean([agent.solution for agent in ionized_pop], axis=0)
                    # Random perturbation
                    pos_new = avg_ionized + storm_power * np.exp(fc) * self.generator.uniform(-fc, fc, self.problem.n_dims)
                else:
                    # Random search
                    pos_new = self.generator.uniform(self.problem.lb, self.problem.ub, self.problem.n_dims)
            pos_new = self.correct_solution(pos_new)
            agent_new = self.generate_agent(pos_new)

            # Select better position
            if self.compare_target(agent_new.target, agent.target):
                pop_new.append(agent_new)
            else:
                pop_new.append(agent)
        self.pop = pop_new
