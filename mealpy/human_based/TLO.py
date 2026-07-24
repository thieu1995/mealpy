#!/usr/bin/env python
# Created by "Thieu" at 10:14, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from functools import reduce
from mealpy.optimizer import Optimizer
from mealpy.utils.opt_info import OptInfo


class ETLBO(Optimizer):
    """
    The original version of: Elitist Teaching Learning-based Optimization (ETLBO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.
    elite_size : int
        Number of elite solutions, in range [1, pop_size/2]. Default is 4.

    References
    ~~~~~~~~~~
    1. Rao, R. and Patel, V., 2012. An elitist teaching-learning-based optimization algorithm for solving
       complex constrained optimization problems. international journal of industrial engineering computations, 3(4), pp.535-560.
       https://doi.org/10.5267/j.ijiec.2012.03.007

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
    >>> model = TLO.ETLBO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Elitist Teaching Learning-based Optimization", year=2012, difficulty="medium", kind="variant")

    def __init__(self, epoch: int = 10000, pop_size: int = 100, elite_size: int = 4, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.elite_size = self.validator.check_int("elite_size", elite_size, [1, self.pop_size//2])
        self.set_parameters(["epoch", "pop_size", "elite_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # 1. Elitism: Keep the elite solutions
        _, elites, _ = self.get_special_agents(self.pop, n_best=self.elite_size, n_worst=1, minmax=self.problem.minmax, return_index=False)

        # 2. Teacher Phase
        mean_result = np.mean([agent.solution for agent in self.pop], axis=0)
        teacher, _ = self.get_best_agent(self.pop, self.problem.minmax)
        pop_new = []
        for idx in range(self.pop_size):
            T_F = np.round(1 + self.generator.random() * 1)  # T_F is either 1 or 2
            r_i = self.generator.random(self.problem.n_dims)
            # Calculate Difference_Mean
            new_solution = self.pop[idx].solution + r_i * (teacher.solution - T_F * mean_result)
            pos_new = self.correct_solution(new_solution)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        # Update fitness in parallel
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        # 3. Learner Phase
        pop_new = []
        for idx in range(self.pop_size):
            qdx = self.sample_indexes_exclude_one(self.generator, self.pop_size, exclude_idx=idx, n_samples=1)
            r_i = self.generator.random(self.problem.n_dims)
            if self.compare_target(self.pop[idx].target, self.pop[qdx].target, self.problem.minmax):
                new_solution = self.pop[idx].solution + r_i * (self.pop[idx].solution - self.pop[qdx].solution)
            else:
                new_solution = self.pop[idx].solution + r_i * (self.pop[qdx].solution - self.pop[idx].solution)
            pos_new = self.correct_solution(new_solution)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        # Update fitness in parallel
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        # 4. Replace worst solutions with elite solutions
        if self.elite_size > 0:
            _, _, worst_idx = self.get_special_agents(self.pop, n_best=1, n_worst=self.elite_size, minmax=self.problem.minmax, return_index=True)
            for idx, wdx in enumerate(worst_idx):
                self.pop[wdx] = elites[idx].copy()

        # 5. Modify duplicate solutions (Mutation to maintain diversity)
        pos_list = np.array([agent.solution for agent in self.pop])
        _, unique_indices = np.unique(pos_list, axis=0, return_index=True)
        duplicate_indices = np.setdiff1d(np.arange(self.pop_size), unique_indices)
        for idx in duplicate_indices:
            # Randomly mutate duplicate solutions within bounds
            pos_new = self.problem.lb + self.generator.random(self.problem.n_dims) * (self.problem.ub - self.problem.lb)
            self.pop[idx] = self.generate_agent(pos_new)


class OriginalTLO(Optimizer):
    """
    The original version of: Teaching Learning-based Optimization (TLO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.

    References
    ~~~~~~~~~~
    1. Rao, R.V., Savsani, V.J. and Vakharia, D.P., 2011. Teaching–learning-based optimization: a novel method
       for constrained mechanical design optimization problems. Computer-aided design, 43(3), pp.303-315.
       https://doi.org/10.1016/j.cad.2010.12.015

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
    """

    OPT_INFO = OptInfo(name="Teaching Learning-based Optimization", year=2011, difficulty="easy", kind="original")

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
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


class ImprovedTLO(OriginalTLO):
    """
    The original version of: Improved Teaching-Learning-based Optimization (ImprovedTLO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.
    n_teachers : int
        Number of teachers in class, in range [2, int(np.sqrt(pop_size) - 1)]. Default is 5.

    References
    ~~~~~~~~~~
    1. Rao, R.V. and Patel, V., 2013. An improved teaching-learning-based optimization algorithm
       for solving unconstrained optimization problems. Scientia Iranica, 20(3), pp.710-720.
       https://doi.org/10.1016/j.scient.2012.12.005

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
    """

    OPT_INFO = OptInfo(name="Improved Teaching-Learning-based Optimization", year=2013, difficulty="hard", kind="variant")

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
        sorted_pop, _ = self.get_sorted_population(self.pop, self.problem.minmax)
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
            team, _ = self.get_sorted_population(team, self.problem.minmax)
            self.teachers[id_teach] = team[0].copy()
            self.teams[id_teach] = team[1:]
        self.pop = self.teachers + reduce(lambda x, y: x + y, self.teams)
