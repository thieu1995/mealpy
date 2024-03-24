#!/usr/bin/env python
# Created by "Thieu" at 21:00, 14/03/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalFLA(Optimizer):
    """
    The original version of: Fick's Law Algorithm (FLA)

    Notes:
        1. The algorithm contains a high number of parameters, some of which may be unnecessary.
        2. Despite the complexity of the algorithms, they may not perform optimally and could potentially become trapped in local optima.
        3. Division by the fitness value may cause overflow issues to arise.
        4. https://www.mathworks.com/matlabcentral/fileexchange/121033-fick-s-law-algorithm-fla

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + C1 (float): factor C1, default=0.5
        + C2 (float): factor C2, default=2.0
        + C3 (float): factor C3, default=0.1
        + C4 (float): factor C4, default=0.2
        + C5 (float): factor C5, default=2.0
        + DD (float): factor D in the paper, default=0.01

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, FLA
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
    >>> model = FLA.OriginalFLA(epoch=1000, pop_size=50, C1 = 0.5, C2 = 2.0, C3 = 0.1, C4 = 0.2, C5 = 2.0, DD = 0.01)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hashim, F. A., Mostafa, R. R., Hussien, A. G., Mirjalili, S., & Sallam, K. M. (2023). Fick’s Law Algorithm: A physical
    law-based algorithm for numerical optimization. Knowledge-Based Systems, 260, 110146.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, C1: float = 0.5, C2: float = 2.0,
                 C3: float = 0.1, C4: float = 0.2, C5: float = 2.0, DD: float = 0.01, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            C1 (float): factor C1, default=0.5
            C2 (float): factor C2, default=2.0
            C3 (float): factor C3, default=0.1
            C4 (float): factor C4, default=0.2
            C5 (float): factor C5, default=2.0
            DD (float): factor D in the paper, default=0.01
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.C1 = self.validator.check_float("C1", C1, (-100., 100.))
        self.C2 = self.validator.check_float("C2", C2, (-100., 100.))
        self.C3 = self.validator.check_float("C3", C3, (-100., 100.))
        self.C4 = self.validator.check_float("C4", C4, (-100., 100.))
        self.C5 = self.validator.check_float("C5", C5, (-100., 100.))
        self.DD = self.validator.check_float("DD", DD, (-100., 100.))
        self.set_parameters(["epoch", "pop_size", "C1", "C2", "C3", "C4", "C5", "DD"])
        self.sort_flag = False

    def before_main_loop(self):
        self.xss = self.get_sorted_population(self.pop, self.problem.minmax)
        self.g_best = self.xss[0].copy()
        self.n1 = int(np.round(self.pop_size/2))
        self.n2 = self.pop_size - self.n1
        self.pop1 = self.pop[:self.n1].copy()
        self.pop2 = self.pop[self.n1:].copy()
        self.best1 = self.get_best_agent(self.pop1, self.problem.minmax)
        self.best2 = self.get_best_agent(self.pop2, self.problem.minmax)
        if self.compare_target(self.best1.target, self.best2.target, self.problem.minmax):
            self.fsss = self.best1.target.fitness
        else:
            self.fsss = self.best2.target.fitness

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pos_list = np.array([agent.solution for agent in self.pop])
        pos1_list = np.array([agent.solution for agent in self.pop1])
        pos2_list = np.array([agent.solution for agent in self.pop2])
        xm1 = np.mean(pos1_list, axis=0)
        xm2 = np.mean(pos2_list, axis=0)
        xm = np.mean(pos_list, axis=0)
        tf = np.sinh((epoch+1) / self.epoch)**self.C1
        pop_new = []
        if tf < 0.9:
            dof = np.exp(-(self.C2 * tf - self.generator.random()))**self.C2
            tdo = self.C5 * tf - self.generator.random()
            if tdo < self.generator.random():
                m1n, m2n = self.C3*self.n1, self.C4*self.n1
                nt12 = int(np.round((m2n - m1n)*self.generator.random() + m1n))
                for idx in range(0, nt12):
                    dfg = self.generator.integers(1, 3)
                    jj = -self.DD * (xm2 - xm1) / np.linalg.norm(self.best2.solution - self.pop1[idx].solution + self.EPSILON)
                    pos_new = self.best2.solution + dfg*dof*self.generator.random(self.problem.n_dims)*(jj*self.best2.solution - self.pop1[idx].solution)
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    pop_new.append(agent)
                for idx in range(nt12, self.n1):
                    tt = self.pop1[idx].solution + dof * (self.generator.random(self.problem.n_dims) * (self.problem.ub - self.problem.lb) + self.problem.lb)
                    pp = self.generator.random(self.problem.n_dims)
                    pos_new = np.where(pp < 0.8, self.best1.solution, np.where(pp >= 0.9, self.pop1[idx].solution, tt))
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    pop_new.append(agent)
                for idx in range(0, self.n2):
                    pos_new = self.best2.solution + dof * (self.generator.random(self.problem.n_dims) * (self.problem.ub - self.problem.lb) + self.problem.lb)
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    pop_new.append(agent)
            else:
                m1n, m2n = 0.1 * self.n2, 0.2 * self.n2
                nt12 = int(np.round((m2n - m1n) * self.generator.random() + m1n))
                for idx in range(0, nt12):
                    dfg = self.generator.integers(1, 3)
                    jj = -self.DD*(xm1-xm2) / np.linalg.norm(self.best1.solution - self.pop2[idx].solution + self.EPSILON)
                    pos_new = self.best1.solution + dfg * dof * self.generator.random(self.problem.n_dims) * (jj * self.best1.solution - self.pop2[idx].solution)
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    pop_new.append(agent)
                for idx in range(nt12, self.n2):
                    tt = self.pop2[idx].solution + dof * (self.generator.random(self.problem.n_dims) * (self.problem.ub - self.problem.lb) + self.problem.lb)
                    pp = self.generator.random(self.problem.n_dims)
                    pos_new = np.where(pp < 0.8, self.best2.solution, np.where(pp >= 0.9, self.pop2[idx].solution, tt))
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    pop_new.append(agent)
                for idx in range(0, self.n1):
                    pos_new = self.best1.solution + dof * (self.generator.random(self.problem.n_dims) * (self.problem.ub - self.problem.lb) + self.problem.lb)
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    pop_new.append(agent)
        else:       # Equilibrium operator (EO)
            if tf <= 1:
                for idx in range(0, self.n1):
                    dfg = self.generator.integers(1, 3)
                    tttt = np.linalg.norm(self.best1.solution - self.pop1[idx].solution)
                    if tttt == 0:
                        jj = 0
                    else:
                        jj = -self.DD*(self.best1.solution - xm1) / tttt
                    drf = np.exp(-jj / tf)
                    ms = np.exp(-self.best1.target.fitness / self.pop1[idx].target.fitness + self.EPSILON)
                    qeo = dfg * drf * self.generator.random(self.problem.n_dims)
                    pos_new = self.best1.solution + qeo*self.pop1[idx].solution + qeo *(ms * self.best1.solution - self.pop1[idx].solution)
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    pop_new.append(agent)
                for idx in range(0, self.n2):
                    dfg = self.generator.integers(1, 3)
                    tttt = np.linalg.norm(self.best2.solution - self.pop2[idx].solution)
                    if tttt == 0:
                        jj = 0
                    else:
                        jj = -self.DD * (self.best2.solution - xm2) / tttt
                    drf = np.exp(-jj / tf)
                    ms = np.exp(-self.best2.target.fitness / self.pop2[idx].target.fitness + self.EPSILON)
                    qeo = dfg * drf * self.generator.random(self.problem.n_dims)
                    pos_new = self.best2.solution + qeo * self.pop2[idx].solution + qeo * (ms * self.best2.solution - self.pop2[idx].solution)
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    pop_new.append(agent)
            else:   # Steady state operator (SSO)
                for idx in range(0, self.n1):
                    dfg = self.generator.integers(1, 3)
                    tttt = np.linalg.norm(self.g_best.solution - self.pop1[idx].solution)
                    if tttt == 0:
                        jj = 0
                    else:
                        jj = -self.DD * (xm - xm1) / tttt
                    drf = np.exp(-jj / tf)
                    ms = np.exp(-self.fsss / self.pop1[idx].target.fitness + self.EPSILON)
                    qg = dfg * drf * self.generator.random(self.problem.n_dims)
                    pos_new = self.g_best.solution + qg * self.pop1[idx].solution + qg * (ms * self.best1.solution - self.pop1[idx].solution)
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    pop_new.append(agent)
                for idx in range(0, self.n2):
                    dfg = self.generator.integers(1, 3)
                    tttt = np.linalg.norm(self.g_best.solution - self.pop2[idx].solution)
                    if tttt == 0:
                        jj = 0
                    else:
                        jj = -self.DD * (xm - xm2) / tttt
                    drf = np.exp(-jj / tf)
                    ms = np.exp(-self.fsss / self.pop2[idx].target.fitness + self.EPSILON)
                    qg = dfg * drf * self.generator.random(self.problem.n_dims)
                    pos_new = self.g_best.solution + qg * self.pop2[idx].solution + qg * (ms * self.g_best.solution - self.pop2[idx].solution)
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    pop_new.append(agent)
        if self.mode not in self.AVAILABLE_MODES:
            for idx in range(0, self.pop_size):
                pop_new[idx].target = self.get_target(pop_new[idx].solution)
        else:
            pop_new = self.update_target_for_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_target(pop_new[idx].target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = pop_new[idx]
        self.pop1 = self.pop[:self.n1].copy()
        self.pop2 = self.pop[self.n1:].copy()
        self.best1 = self.get_best_agent(self.pop1, self.problem.minmax)
        self.best2 = self.get_best_agent(self.pop2, self.problem.minmax)
        if self.compare_target(self.best1.target, self.best2.target, self.problem.minmax):
            self.fsss = self.best1.target.fitness
        else:
            self.fsss = self.best2.target.fitness
