#!/usr/bin/env python
# Created by "Thieu" at 10:14, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from functools import reduce
from mealpy.optimizer import Optimizer


class DevTLO(Optimizer):
    """
    The developed version: Teaching Learning-based Optimization (TLO)

    Links:
       1. https://doi.org/10.5267/j.ijiec.2012.03.007

    Notes:
        + Use numpy np.array to make operations faster
        + The global best solution is used

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, TLO
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
    >>> model = TLO.DevTLO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R. and Patel, V., 2012. An elitist teaching-learning-based optimization algorithm for solving
    complex constrained optimization problems. international journal of industrial engineering computations, 3(4), pp.535-560.
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

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            ## Teaching Phrase
            TF = self.generator.integers(1, 3)  # 1 or 2 (never 3)
            list_pos = np.array([agent.solution for agent in self.pop])
            DIFF_MEAN = self.generator.random(self.problem.n_dims) * (self.g_best.solution - TF * np.mean(list_pos, axis=0))
            pos_new = self.pop[idx].solution + DIFF_MEAN
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        pop_child = []
        for idx in range(0, self.pop_size):
            ## Learning Phrase
            pos_new = self.pop[idx].solution.copy().astype(float)
            id_partner = self.generator.choice(np.setxor1d(np.array(range(self.pop_size)), np.array([idx])))
            if self.compare_target(self.pop[idx].target, self.pop[id_partner].target, self.problem.minmax):
                pos_new += self.generator.random(self.problem.n_dims) * (self.pop[idx].solution - self.pop[id_partner].solution)
            else:
                pos_new += self.generator.random(self.problem.n_dims) * (self.pop[id_partner].solution - self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.pop = self.greedy_selection_population(self.pop, pop_child, self.problem.minmax)


class OriginalTLO(DevTLO):
    """
    The original version of: Teaching Learning-based Optimization (TLO)

    Notes:
        + Third loops are removed
        + This version is inspired from above link
        + https://github.com/andaviaco/tblo

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, TLO
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
    >>> model = TLO.OriginalTLO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R.V., Savsani, V.J. and Vakharia, D.P., 2011. Teaching–learning-based optimization: a novel method
    for constrained mechanical design optimization problems. Computer-aided design, 43(3), pp.303-315.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.is_parallelizable = False
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):
            ## Teaching Phrase
            TF = self.generator.integers(1, 3)  # 1 or 2 (never 3)
            #### Remove third loop here
            list_pos = np.array([agent.solution for agent in self.pop])
            pos_new = self.pop[idx].solution + self.generator.uniform(0, 1, self.problem.n_dims) * \
                      (self.g_best.solution - TF * np.mean(list_pos, axis=0))
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
            ## Learning Phrase
            id_partner = self.generator.choice(np.setxor1d(np.array(range(self.pop_size)), np.array([idx])))
            #### Remove third loop here
            if self.compare_target(self.pop[idx].target, self.pop[id_partner].target, self.problem.minmax):
                diff = self.pop[idx].solution - self.pop[id_partner].solution
            else:
                diff = self.pop[id_partner].solution - self.pop[idx].solution
            pos_new = self.pop[idx].solution + self.generator.uniform(0, 1, self.problem.n_dims) * diff
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent


class ImprovedTLO(DevTLO):
    """
    The original version of: Improved Teaching-Learning-based Optimization (ImprovedTLO)

    Links:
       1. https://doi.org/10.1016/j.scient.2012.12.005

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_teachers (int): [3, 10], number of teachers in class, default=5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, TLO
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
    >>> model = TLO.ImprovedTLO(epoch=1000, pop_size=50, n_teachers = 5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rao, R.V. and Patel, V., 2013. An improved teaching-learning-based optimization algorithm
    for solving unconstrained optimization problems. Scientia Iranica, 20(3), pp.710-720.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_teachers: int = 5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_teachers (int): number of teachers in class
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.n_teachers = self.validator.check_int("n_teachers", n_teachers, [2, int(np.sqrt(self.pop_size)-1)])
        self.set_parameters(["epoch", "pop_size", "n_teachers"])
        self.n_students = self.pop_size - self.n_teachers
        self.n_students_in_team = int(self.n_students / self.n_teachers)
        self.sort_flag = False

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        sorted_pop = self.get_sorted_population(self.pop, self.problem.minmax)
        self.g_best = sorted_pop[0].copy()
        self.teachers = sorted_pop[:self.n_teachers].copy()
        sorted_pop = sorted_pop[self.n_teachers:]
        idx_list = self.generator.permutation(range(0, self.n_students))
        self.teams = []
        for id_teacher in range(0, self.n_teachers):
            group = []
            for idx in range(0, self.n_students_in_team):
                start_index = id_teacher * self.n_students_in_team + idx
                group.append(sorted_pop[idx_list[start_index]])
            self.teams.append(group)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for id_teach, teacher in enumerate(self.teachers):
            team = self.teams[id_teach]
            list_pos = np.array([student.solution for student in self.teams[id_teach]])  # Step 7
            mean_team = np.mean(list_pos, axis=0)
            pop_new = []
            for id_stud, student in enumerate(team):
                if teacher.target.fitness == 0:
                    TF = 1
                else:
                    TF = student.target.fitness / teacher.target.fitness
                diff_mean = self.generator.random() * (teacher.solution - TF * mean_team)  # Step 8
                id2 = self.generator.choice(list(set(range(0, self.n_teachers)) - {id_teach}))
                if self.compare_target(teacher.target, team[id2].target, self.problem.minmax):
                    pos_new = (student.solution + diff_mean) + self.generator.random() * (team[id2].solution - student.solution)
                else:
                    pos_new = (student.solution + diff_mean) + self.generator.random() * (student.solution - team[id2].solution)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    pop_new[-1] = self.get_better_agent(agent, student, self.problem.minmax)
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_for_population(pop_new)
                pop_new = self.greedy_selection_population(team, pop_new, self.problem.minmax)
            self.teams[id_teach] = pop_new

        for id_teach, teacher in enumerate(self.teachers):
            ef = round(1 + self.generator.random())
            team = self.teams[id_teach]
            pop_new = []
            for id_stud, student in enumerate(team):
                id2 = self.generator.choice(list(set(range(0, self.n_students_in_team)) - {id_stud}))
                if self.compare_target(student.target, team[id2].target, self.problem.minmax):
                    pos_new = student.solution + self.generator.random() * (student.solution - team[id2].solution) + \
                              self.generator.random() * (teacher.solution - ef * team[id2].solution)
                else:
                    pos_new = student.solution + self.generator.random() * (team[id2].solution - student.solution) + \
                              self.generator.random() * (teacher.solution - ef * student.solution)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    pop_new[-1] = self.get_better_agent(agent, student, self.problem.minmax)
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_for_population(pop_new)
                pop_new = self.greedy_selection_population(team, pop_new, self.problem.minmax)
            self.teams[id_teach] = pop_new
        for id_teach, teacher in enumerate(self.teachers):
            team = self.teams[id_teach] + [teacher]
            team = self.get_sorted_population(team, self.problem.minmax)
            self.teachers[id_teach] = team[0].copy()
            self.teams[id_teach] = team[1:]
        self.pop = self.teachers + reduce(lambda x, y: x + y, self.teams)
