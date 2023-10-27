#!/usr/bin/env python
# Created by "Thieu" at 18:09, 13/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalEVO(Optimizer):
    """
    The original version of: Energy Valley Optimizer (EVO)

    Links:
        1. https://www.nature.com/articles/s41598-022-27344-y
        2. https://www.mathworks.com/matlabcentral/fileexchange/123130-energy-valley-optimizer-a-novel-metaheuristic-algorithm

    Notes:
        1. The algorithm is straightforward and does not require any specialized knowledge or techniques.
        2. The algorithm may not perform optimally due to slow convergence and no good operations, which could be improved by implementing better strategies and operations.
        3. The problem is that it is stuck at a local optimal around 1/2 of the max generations because fitness distance is being used as a factor in the equations.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, EVO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = EVO.OriginalEVO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Azizi, M., Aickelin, U., A. Khorshidi, H., & Baghalzadeh Shishehgarkhaneh, M. (2023). Energy valley optimizer: a novel
    metaheuristic algorithm for global and engineering optimization. Scientific Reports, 13(1), 226.
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
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_list = np.array([agent.solution for agent in self.pop])
            fit_list = np.array([agent.target.fitness for agent in self.pop])
            dis = np.sqrt(np.sum((self.pop[idx].solution - pos_list)**2, axis=1))
            idx_dis_sort = np.argsort(dis)
            CnPtIdx = self.generator.choice(list(set(range(2, self.pop_size)) - {idx}))
            x_team = pos_list[idx_dis_sort[1:CnPtIdx], :]
            x_avg_team = np.mean(x_team, axis=0)
            x_avg_pop = np.mean(pos_list, axis=0)
            eb = np.mean(fit_list)
            sl = (fit_list[idx] - self.g_best.target.fitness) / (self.g_worst.target.fitness - self.g_best.target.fitness + self.EPSILON)

            pos_new1 = self.pop[idx].solution.copy()
            pos_new2 = self.pop[idx].solution.copy()
            if self.compare_fitness(eb, self.pop[idx].target.fitness, self.problem.minmax):
                if self.generator.random() > sl:
                    a1_idx = self.generator.integers(self.problem.n_dims)
                    a2_idx = self.generator.integers(0, self.problem.n_dims, size=a1_idx)
                    pos_new1[a2_idx] = self.g_best.solution[a2_idx]
                    g1_idx = self.generator.integers(self.problem.n_dims)
                    g2_idx = self.generator.integers(0, self.problem.n_dims, size=g1_idx)
                    pos_new2[g2_idx] = x_avg_team[g2_idx]
                else:
                    ir = self.generator.uniform(0, 1, 2)
                    jr = self.generator.uniform(0, 1, self.problem.n_dims)
                    pos_new1 += jr * (ir[0] * self.g_best.solution - ir[1] * x_avg_pop) / sl
                    ir = self.generator.uniform(0, 1, 2)
                    jr = self.generator.uniform(0, 1, self.problem.n_dims)
                    pos_new2 += jr * (ir[0] * self.g_best.solution - ir[1] * x_avg_team)
                pos_new1 = self.correct_solution(pos_new1)
                pos_new2 = self.correct_solution(pos_new2)
                agent1 = self.generate_empty_agent(pos_new1)
                agent2 = self.generate_empty_agent(pos_new2)
                pop_new.append(agent1)
                pop_new.append(agent2)
            else:
                pos_new = pos_new1 + self.generator.random() * sl * self.generator.uniform(self.problem.lb, self.problem.ub, self.problem.n_dims)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
        if self.mode not in self.AVAILABLE_MODES:
            for idx in range(0, len(pop_new)):
                pop_new[idx].target = self.get_target(pop_new[idx].solution)
        pop_new = self.update_target_for_population(pop_new)
        self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_new, self.pop_size, self.problem.minmax)
