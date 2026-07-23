#!/usr/bin/env python
# Created by "Thieu" at 19:49, 09/07/2026 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalAPO(Optimizer):
    """
    The original version of: Artificial Protozoa Optimizer (APO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations. Default is 10000.
    pop_size : int
        Population size. Default is 100.
    pf_max : float
        Proportion fraction maximum, in range (0.0, 1.0), better [0.1, 0.3].
    n_pairs : int
        Number of neighbor pairs, in range [1, floor(pop_size/2)], better [2, 5].

    Links
    -----
    1. https://doi.org/10.1016/j.knosys.2024.111737
    2. https://www.mathworks.com/matlabcentral/fileexchange/162656-artificial-protozoa-optimizer

    References
    ~~~~~~~~~~
    1. Wang, X., Snášel, V., Mirjalili, S., Pan, J. S., Kong, L., & Shehadeh, H. A. (2024).
       Artificial Protozoa Optimizer (APO): A novel bio-inspired metaheuristic algorithm
       for engineering optimization. Knowledge-based systems, 295, 111737.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, APO
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
    >>> model = APO.OriginalAPO(epoch=1000, pop_size=50, pf_max=0.1, n_pairs=2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, pf_max: float = 0.1, n_pairs: int = 2, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pf_max = self.validator.check_float("pf_max", pf_max, (0., 1.0))
        self.n_pairs = self.validator.check_int("n_pairs", n_pairs, [1, int(np.floor(self.pop_size / 2))])
        self.set_parameters(["epoch", "pop_size", "pf_max", "n_pairs"])
        self.sort_flag = True

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch: The current iteration
        """
        pf = self.pf_max * self.generator.random()
        # Random indices for dormancy or reproduction forms
        ri = self.generator.choice(self.pop_size, size=int(np.ceil(self.pop_size * pf)), replace=False)

        pop_new = []
        for idx in range(self.pop_size):
            if idx in ri:
                # Dormancy or Reproduction form
                pdr = 0.5 * (1 + np.cos((1 - (idx + 1) / self.pop_size) * np.pi))
                pos_rand = self.problem.lb + self.generator.random(self.problem.n_dims) * (self.problem.ub - self.problem.lb)
                if self.generator.random() < pdr:
                    # Dormancy form
                    pos_new = pos_rand
                else:
                    # Reproduction form
                    flag = 1 if self.generator.random() < 0.5 else -1
                    mr = np.zeros(self.problem.n_dims)
                    rand_indices = self.generator.choice(self.problem.n_dims, size=int(np.ceil(self.generator.random() * self.problem.n_dims)), replace=False)
                    mr[rand_indices] = 1
                    pos_new = self.pop[idx].solution + flag * self.generator.random() * pos_rand * mr
            else:
                # Foraging form
                ff = self.generator.random() * (1 + np.cos(epoch / self.epoch * np.pi))
                mf = np.zeros(self.problem.n_dims)
                rand_indices = self.generator.choice(self.problem.n_dims, size=int(np.ceil(self.problem.n_dims * (idx + 1) / self.pop_size)), replace=False)
                mf[rand_indices] = 1
                pah = 0.5 * (1 + np.cos(epoch / self.epoch * np.pi))
                if self.generator.random() < pah:
                    # Autotroph form
                    jdx = self.generator.integers(0, self.pop_size)
                    epn = np.zeros((self.n_pairs, self.problem.n_dims))
                    for kdx in range(self.n_pairs):
                        # Index selection adjusted for Python 0-based indexing
                        if idx == 0:
                            km = idx
                            kp = self.generator.integers(idx + 1, self.pop_size)
                        elif idx == self.pop_size - 1:
                            km = self.generator.integers(0, self.pop_size - 1)
                            kp = idx
                        else:
                            km = self.generator.integers(0, idx)
                            kp = self.generator.integers(idx + 1, self.pop_size)

                        wa = np.exp(-np.abs(self.pop[km].target.fitness / (self.pop[kp].target.fitness + self.EPSILON)))
                        epn[kdx] = wa * (self.pop[km].solution - self.pop[kp].solution)
                    pos_new = self.pop[idx].solution + ff * (self.pop[jdx].solution - self.pop[idx].solution + (1 / self.n_pairs) * np.sum(epn, axis=0)) * mf
                else:
                    # Heterotroph form
                    epn = np.zeros((self.n_pairs, self.problem.n_dims))
                    for k_idx in range(self.n_pairs):
                        k = k_idx + 1
                        imk = max(0, idx - k)
                        ipk = min(self.pop_size - 1, idx + k)
                        wh = np.exp(-np.abs(self.pop[imk].target.fitness / (self.pop[ipk].target.fitness + self.EPSILON)))
                        epn[k_idx] = wh * (self.pop[imk].solution - self.pop[ipk].solution)

                    flag = 1 if self.generator.random() < 0.5 else -1
                    x_near = (1 + flag * self.generator.random(self.problem.n_dims) * (1 - epoch / self.epoch)) * self.pop[idx].solution
                    pos_new = self.pop[idx].solution + ff * (x_near - self.pop[idx].solution + (1 / self.n_pairs) * np.sum(epn, axis=0)) * mf
            pos_new = self.correct_solution(pos_new)
            agent_new = self.generate_empty_agent(pos_new)
            pop_new.append(agent_new)
            if self.mode not in self.AVAILABLE_MODES:
                agent_new.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent_new, minmax=self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)
