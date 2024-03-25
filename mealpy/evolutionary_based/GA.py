#!/usr/bin/env python
# Created by "Thieu" at 09:33, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseGA(Optimizer):
    """
    The original version of: Genetic Algorithm (GA)

    Links:
        1. https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        2. https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        3. https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.95
        + pm (float): [0.01, 0.2], mutation probability, default = 0.025
        + selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
        + k_way (float): Optional, set it when use "tournament" selection, default = 0.2
        + crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
        + mutation_multipoints (bool): Optional, True or False, effect on mutation process, default = True
        + mutation (str): Optional, can be ["flip", "swap"] for multipoints and can be ["flip", "swap", "scramble", "inversion"] for one-point

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GA
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
    >>> model = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    >>>
    >>> model2 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, selection="tournament", k_way=0.4, crossover="multi_points")
    >>>
    >>> model3 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, crossover="one_point", mutation="scramble")
    >>>
    >>> model4 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, crossover="arithmetic", mutation_multipoints=True, mutation="swap")
    >>>
    >>> model5 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, selection="roulette", crossover="multi_points")
    >>>
    >>> model6 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, selection="random", mutation="inversion")
    >>>
    >>> model7 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, crossover="arithmetic", mutation="flip")

    References
    ~~~~~~~~~~
    [1] Whitley, D., 1994. A genetic algorithm tutorial. Statistics and computing, 4(2), pp.65-85.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, pc: float = 0.95, pm: float = 0.025, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            pc: cross-over probability, default = 0.95
            pm: mutation probability, default = 0.025
            selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
            k_way (float): Optional, set it when use "tournament" selection, default = 0.2
            crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
            mutation_multipoints (bool): Optional, True or False, effect on mutation process, default = False
            mutation (str): Optional, can be ["flip", "swap"] for multipoints and can be ["flip", "swap", "scramble", "inversion"] for one-point, default="flip"
        """
        super().__init__(**kwargs)

        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pc = self.validator.check_float("pc", pc, (0, 1.0))
        self.pm = self.validator.check_float("pm", pm, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "pc", "pm"])
        self.sort_flag = False
        self.selection = "tournament"
        self.k_way = 0.2
        self.crossover = "uniform"
        self.mutation = "flip"
        self.mutation_multipoints = True

        if "selection" in kwargs:
            self.selection = self.validator.check_str("selection", kwargs["selection"], ["tournament", "random", "roulette"])
        if "k_way" in kwargs:
            self.k_way = self.validator.check_float("k_way", kwargs["k_way"], (0, 1.0))
        if "crossover" in kwargs:
            self.crossover = self.validator.check_str("crossover", kwargs["crossover"], ["one_point", "multi_points", "uniform", "arithmetic"])
        if "mutation_multipoints" in kwargs:
            self.mutation_multipoints = self.validator.check_bool("mutation_multipoints", kwargs["mutation_multipoints"])
        if self.mutation_multipoints:
            if "mutation" in kwargs:
                self.mutation = self.validator.check_str("mutation", kwargs["mutation"], ["flip", "swap"])
        else:
            if "mutation" in kwargs:
                self.mutation = self.validator.check_str("mutation", kwargs["mutation"], ["flip", "swap", "scramble", "inversion"])

    def selection_process__(self, list_fitness):
        """
        Notes
        ~~~~~
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
        + Default selection strategy is Tournament with k% = 0.2.
        + Other strategy like "roulette" and "random" can be selected via Optional parameter "selection"

        Args:
            list_fitness (np.array): list of fitness values.

        Returns:
            list: The position of dad and mom
        """
        if self.selection == "roulette":
            id_c1 = self.get_index_roulette_wheel_selection(list_fitness)
            id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
            while id_c2 == id_c1:
                id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
        elif self.selection == "random":
            id_c1, id_c2 = self.generator.choice(range(self.pop_size), 2, replace=False)
        else:   ## tournament
            id_c1, id_c2 = self.get_index_kway_tournament_selection(self.pop, k_way=self.k_way, output=2)
        return self.pop[id_c1].solution, self.pop[id_c2].solution

    def selection_process_00__(self, pop_selected):
        """
        Notes
        ~~~~~
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
        + Default selection strategy is Tournament with k% = 0.2.
        + Other strategy like "roulette" and "random" can be selected via Optional parameter "selection"

        Args:
            pop_selected (np.array): a population that will be selected

        Returns:
            list: The position of dad and mom
        """
        if self.selection == "roulette":
            list_fitness = np.array([agent.target.fitness for agent in pop_selected])
            id_c1 = self.get_index_roulette_wheel_selection(list_fitness)
            id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
            while id_c2 == id_c1:
                id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
        elif self.selection == "random":
            id_c1, id_c2 = self.generator.choice(range(len(pop_selected)), 2, replace=False)
        else:   ## tournament
            id_c1, id_c2 = self.get_index_kway_tournament_selection(pop_selected, k_way=self.k_way, output=2)
        return pop_selected[id_c1].solution, pop_selected[id_c2].solution

    def selection_process_01__(self, pop_dad, pop_mom):
        """
        Notes
        ~~~~~
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
        + Default selection strategy is Tournament with k% = 0.2.
        + Other strategy like "roulette" and "random" can be selected via Optional parameter "selection"

        Returns:
            list: The position of dad and mom
        """
        if self.selection == "roulette":
            list_fit_dad = np.array([agent.target.fitness for agent in pop_dad])
            list_fit_mom = np.array([agent.target.fitness for agent in pop_mom])
            id_c1 = self.get_index_roulette_wheel_selection(list_fit_dad)
            id_c2 = self.get_index_roulette_wheel_selection(list_fit_mom)
        elif self.selection == "random":
            id_c1 = self.generator.choice(range(len(pop_dad)))
            id_c2 = self.generator.choice(range(len(pop_mom)))
        else:   ## tournament
            id_c1 = self.get_index_kway_tournament_selection(pop_dad, k_way=self.k_way, output=1)[0]
            id_c2 = self.get_index_kway_tournament_selection(pop_mom, k_way=self.k_way, output=1)[0]
        return pop_dad[id_c1].solution, pop_mom[id_c2].solution

    def crossover_process__(self, dad, mom):
        """
        Notes
        ~~~~~
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm
        + Default crossover strategy is "uniform"
        + Other strategy like "arithmetic", "one_point", "multi_points" can be selected via parameter: crossover

        Args:
            dad (np.array): The position of dad
            mom (np.array): The position of mom

        Returns:
            list: The position of child 1 and child 2
        """
        if self.crossover == "arithmetic":
            w1, w2 = self.crossover_arithmetic(dad, mom)
        elif self.crossover == "one_point":
            cut = self.generator.integers(1, self.problem.n_dims-1)
            w1 = np.concatenate([dad[:cut], mom[cut:]])
            w2 = np.concatenate([mom[:cut], dad[cut:]])
        elif self.crossover == "multi_points":
            idxs = self.generator.choice(range(1, self.problem.n_dims-1), 2, replace=False)
            cut1, cut2 = np.min(idxs), np.max(idxs)
            w1 = np.concatenate([dad[:cut1], mom[cut1:cut2], dad[cut2:]])
            w2 = np.concatenate([mom[:cut1], dad[cut1:cut2], mom[cut2:]])
        else:           # uniform
            flip = self.generator.integers(0, 2, self.problem.n_dims)
            w1 = dad * flip + mom * (1 - flip)
            w2 = mom * flip + dad * (1 - flip)
        return w1, w2

    def mutation_process__(self, child):
        """
        Notes
        ~~~~~
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm
        + There are 2 strategies that effects by the mutation probability: Mutated on single point or the whole vector.
            + Multiple points (whole vector) has 2 strategies selected via parameter: mutation
                + flip --> (default in this case) should set the pm small such as: [0.01 -> 0.2]
                + swap --> should set the pm small such as: [0.01 -> 0.2]
            + Single point has 4 strategies:
                + flip --> should set the pm large such as: [0.5 -> 0.9]
                + swap --> same as flip: pm in range [0.5 -> 0.9]
                + scramble --> should set the pm small enough such as: [0.4 -> 0.6]
                + inversion --> like scramble [0.4 -> 0.6]

        Args:
            child (np.array): The position of the child

        Returns:
            np.array: The mutated vector of the child
        """

        if self.mutation_multipoints:
            if self.mutation == "swap":
                for idx in range(self.problem.n_dims):
                    idx_swap = self.generator.choice(list(set(range(0, self.problem.n_dims)) - {idx}))
                    child[idx], child[idx_swap] = child[idx_swap], child[idx]
                    return child
            else:       # "flip"
                mutation_child = self.problem.generate_solution()
                flag_child = self.generator.uniform(0, 1, self.problem.n_dims) < self.pm
                return np.where(flag_child, mutation_child, child)
        else:
            if self.mutation == "swap":
                idx1, idx2 = self.generator.choice(range(0, self.problem.n_dims), 2, replace=False)
                child[idx1], child[idx2] = child[idx2], child[idx1]
                return child
            elif self.mutation == "inversion":
                cut1, cut2 = self.generator.choice(range(0, self.problem.n_dims), 2, replace=False)
                temp = child[cut1:cut2]
                temp = temp[::-1]
                child[cut1:cut2] = temp
                return child
            elif self.mutation == "scramble":
                cut1, cut2 = self.generator.choice(range(0, self.problem.n_dims), 2, replace=False)
                temp = child[cut1:cut2]
                self.generator.shuffle(temp)
                child[cut1:cut2] = temp
                return child
            else:   # "flip"
                idx = self.generator.integers(0, self.problem.n_dims)
                child[idx] = self.generator.uniform(self.problem.lb[idx], self.problem.ub[idx])
                return child

    def survivor_process__(self, pop, pop_child):
        """
        The current survivor process is select the worst solution out of k-way solutions (tournament selection) and
        compare with child solutions. The better solution will be kept for the next generation.

        Args:
            pop: The old population
            pop_child: The new population

        Returns:
            The new population
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            id_child = self.get_index_kway_tournament_selection(pop, k_way=0.1, output=1, reverse=True)[0]
            pop_new.append(self.get_better_agent(pop_child[idx], pop[id_child], self.problem.minmax))
        return pop_new

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        list_fitness = np.array([agent.target.fitness for agent in self.pop])
        pop_new = []
        for i in range(0, int(self.pop_size/2)):
            ### Selection
            child1, child2 = self.selection_process__(list_fitness)

            ### Crossover
            if self.generator.random() < self.pc:
                child1, child2 = self.crossover_process__(child1, child2)

            ### Mutation
            child1 = self.mutation_process__(child1)
            child2 = self.mutation_process__(child2)

            child1 = self.correct_solution(child1)
            child2 = self.correct_solution(child2)

            agent1 = self.generate_empty_agent(child1)
            agent2 = self.generate_empty_agent(child2)

            pop_new.append(agent1)
            pop_new.append(agent2)

            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-2].target = self.get_target(child1)
                pop_new[-1].target = self.get_target(child2)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        ### Survivor Selection
        self.pop = self.survivor_process__(self.pop, pop_new)


class SingleGA(BaseGA):
    """
    The developed single-point mutation of: Genetic Algorithm (GA)

    Links:
        1. https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        2. https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        3. https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.95
        + pm (float): [0.01, 0.2], mutation probability, default = 0.025
        + selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
        + crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
        + mutation (str): Optional, can be ["flip", "swap", "scramble", "inversion"] for one-point
        + k_way (float): Optional, set it when use "tournament" selection, default = 0.2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GA
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
    >>> model = GA.SingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection = "roulette", crossover = "uniform", mutation = "swap")
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    >>>
    >>> model2 = GA.SingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="tournament", k_way=0.4, crossover="multi_points")
    >>>
    >>> model3 = GA.SingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="one_point", mutation="scramble")
    >>>
    >>> model4 = GA.SingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="arithmetic", mutation="swap")
    >>>
    >>> model5 = GA.SingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="roulette", crossover="multi_points")
    >>>
    >>> model6 = GA.SingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="random", mutation="inversion")
    >>>
    >>> model7 = GA.SingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="arithmetic", mutation="flip")

    References
    ~~~~~~~~~~
    [1] Whitley, D., 1994. A genetic algorithm tutorial. Statistics and computing, 4(2), pp.65-85.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, pc: float = 0.95, pm: float = 0.8, selection: str = "roulette",
                 crossover: str = "uniform", mutation: str = "swap", k_way: float = 0.2, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            pc: cross-over probability, default = 0.95
            pm: mutation probability, default = 0.8
            selection: Optional, can be ["roulette", "tournament", "random"], default = "tournament"
            crossover: Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
            mutation: Optional, can be ["flip", "swap", "scramble", "inversion"], default="flip"
            k_way: Optional, set it when use "tournament" selection, default = 0.2
        """
        super().__init__(epoch, pop_size, pc, pm, **kwargs)
        self.selection = self.validator.check_str("selection", selection, ["tournament", "random", "roulette"])
        self.crossover = self.validator.check_str("crossover", crossover, ["one_point", "multi_points", "uniform", "arithmetic"])
        self.mutation = self.validator.check_str("mutation", mutation, ["flip", "swap", "scramble", "inversion"])
        self.k_way = self.validator.check_float("k_way", k_way, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "pc", "pm", "selection", "crossover", "mutation", "k_way"])
        self.sort_flag = False

    def mutation_process__(self, child):
        """
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm
        + The mutation process is effected by parameter: pm
            + flip --> should set the pm large such as: [0.5 -> 0.9]
            + swap --> same as flip: pm in range [0.5 -> 0.9]
            + scramble --> should set the pm small enough such as: [0.4 -> 0.6]
            + inversion --> like scramble [0.4 -> 0.6]

        Args:
            child (np.array): The position of the child

        Returns:
            np.array: The mutated vector of the child
        """
        if self.mutation == "swap":
            idx1, idx2 = self.generator.choice(range(0, self.problem.n_dims), 2, replace=False)
            child[idx1], child[idx2] = child[idx2], child[idx1]
            return child
        elif self.mutation == "inversion":
            cut1, cut2 = self.generator.choice(range(0, self.problem.n_dims), 2, replace=False)
            temp = child[cut1:cut2]
            temp = temp[::-1]
            child[cut1:cut2] = temp
            return child
        elif self.mutation == "scramble":
            cut1, cut2 = self.generator.choice(range(0, self.problem.n_dims), 2, replace=False)
            temp = child[cut1:cut2]
            self.generator.shuffle(temp)
            child[cut1:cut2] = temp
            return child
        else:   # "flip"
            idx = self.generator.integers(0, self.problem.n_dims)
            child[idx] = self.generator.uniform(self.problem.lb[idx], self.problem.ub[idx])
            return child


class EliteSingleGA(SingleGA):
    """
    The developed elite single-point mutation of: Genetic Algorithm (GA)

    Links:
        1. https://www.baeldung.com/cs/elitism-in-evolutionary-algorithms

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.95
        + pm (float): [0.01, 0.2], mutation probability, default = 0.025
        + selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
        + crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
        + mutation (str): Optional, can be ["flip", "swap", "scramble", "inversion"] for one-point
        + k_way (float): Optional, set it when use "tournament" selection, default = 0.2
        + elite_best (float/int): Optional, can be float (percentage of the best in elite group), or int (the number of best elite), default = 0.1
        + elite_worst (float/int): Opttional, can be float (percentage of the worst in elite group), or int (the number of worst elite), default = 0.3
        + strategy (int): Optional, can be 0 or 1. If = 0, the selection is select parents from (elite_worst + non_elite_group).
            Else, the selection will select dad from elite_worst and mom from non_elite_group.
        + pop_size = elite_group (elite_best + elite_worst) + non_elite_group

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GA
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
    >>> model = GA.EliteSingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection = "roulette", crossover = "uniform",
    >>>                         mutation = "swap", elite_best = 0.1, elite_worst = 0.3, strategy = 0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    >>>
    >>> model2 = GA.EliteSingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="tournament", k_way=0.4, crossover="multi_points")
    >>>
    >>> model3 = GA.EliteSingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="one_point", mutation="scramble")
    >>>
    >>> model4 = GA.EliteSingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="arithmetic", mutation="swap")
    >>>
    >>> model5 = GA.EliteSingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="roulette", crossover="multi_points")
    >>>
    >>> model6 = GA.EliteSingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="random", mutation="inversion")
    >>>
    >>> model7 = GA.EliteSingleGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="arithmetic", mutation="flip")

    References
    ~~~~~~~~~~
    [1] Whitley, D., 1994. A genetic algorithm tutorial. Statistics and computing, 4(2), pp.65-85.
    """

    def __init__(self, epoch=10000, pop_size=100, pc=0.95, pm=0.8, selection="roulette",
                 crossover="uniform", mutation="swap", k_way=0.2,
                 elite_best=0.1, elite_worst=0.3, strategy=0, **kwargs):
        super().__init__(epoch, pop_size, pc, pm, selection, crossover, mutation, k_way, **kwargs)
        self.elite_best = self.validator.check_is_int_and_float("elite_best", elite_best, [1, int(self.pop_size / 2)-1], (0, 0.5))
        self.n_elite_best = int(self.elite_best * self.pop_size) if self.elite_best < 1 else self.elite_best
        if self.n_elite_best < 1:
            self.n_elite_best = 1

        self.elite_worst = self.validator.check_is_int_and_float("elite_worst", elite_worst, [1, int(self.pop_size / 2)-1], (0, 0.5))
        self.n_elite_worst = int(self.elite_worst * self.pop_size) if self.elite_worst < 1 else self.elite_worst
        if self.n_elite_worst < 1:
            self.n_elite_worst = 1

        self.strategy = self.validator.check_int("strategy", strategy, [0, 1])
        self.set_parameters(["epoch", "pop_size", "pc", "pm", "selection", "crossover", "mutation", "k_way",
                             "elite_best", "elite_worst", "strategy"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = self.pop[:self.n_elite_best]
        if self.strategy == 0:
            pop_old = self.pop[self.n_elite_best:]
            for idx in range(self.n_elite_best, self.pop_size):
                ### Selection
                child1, child2 = self.selection_process_00__(pop_old)
                ### Crossover
                if self.generator.uniform() < self.pc:
                    child1, child2 = self.crossover_process__(child1, child2)
                child = child1 if self.generator.random() <= 0.5 else child2
                ### Mutation
                child = self.mutation_process__(child)
                ### Survivor Selection
                pos_new = self.correct_solution(child)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1].target = self.get_target(pos_new)
            self.pop = self.update_target_for_population(pop_new)
        else:
            pop_dad = self.pop[self.n_elite_best:self.n_elite_best+self.n_elite_worst]
            pop_mom = self.pop[self.n_elite_best+self.n_elite_worst:]
            for idx in range(self.n_elite_best, self.pop_size):
                ### Selection
                child1, child2 = self.selection_process_01__(pop_dad, pop_mom)
                ### Crossover
                if self.generator.uniform() < self.pc:
                    child1, child2 = self.crossover_process__(child1, child2)
                child = child1 if self.generator.random() <= 0.5 else child2
                ### Mutation
                child = self.mutation_process__(child)
                ### Survivor Selection
                pos_new = self.correct_solution(child)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1].target = self.get_target(pos_new)
            self.pop = self.update_target_for_population(pop_new)


class MultiGA(BaseGA):
    """
    The developed multipoints-mutation version of: Genetic Algorithm (GA)

    Links:
        1. https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        2. https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        3. https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.95
        + pm (float): [0.01, 0.2], mutation probability, default = 0.025
        + selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
        + k_way (float): Optional, set it when use "tournament" selection, default = 0.2
        + crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
        + mutation (str): Optional, can be ["flip", "swap"] for multipoints

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GA
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
    >>> model = GA.MultiGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection = "roulette", crossover = "uniform", mutation = "swap", k_way=0.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    >>>
    >>> model2 = GA.MultiGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="tournament", k_way=0.4, crossover="multi_points")
    >>>
    >>> model3 = GA.MultiGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="one_point", mutation="flip")
    >>>
    >>> model4 = GA.MultiGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="arithmetic", mutation_multipoints=True, mutation="swap")
    >>>
    >>> model5 = GA.MultiGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="roulette", crossover="multi_points")
    >>>
    >>> model6 = GA.MultiGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, selection="random", mutation="swap")
    >>>
    >>> model7 = GA.MultiGA(epoch=1000, pop_size=50, pc=0.9, pm=0.8, crossover="arithmetic", mutation="flip")

    References
    ~~~~~~~~~~
    [1] Whitley, D., 1994. A genetic algorithm tutorial. Statistics and computing, 4(2), pp.65-85.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, pc: float = 0.95, pm: float = 0.025,
                 selection: str = "roulette", crossover: str = "arithmetic", mutation: str = "flip", k_way: float = 0.2, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            pc: cross-over probability, default = 0.95
            pm: mutation probability, default = 0.025
            selection: Optional, can be ["roulette", "tournament", "random"], default = "tournament"
            crossover: Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
            mutation: Optional, can be ["flip", "swap"] for multipoints
            k_way: Optional, set it when use "tournament" selection, default = 0.2
        """
        super().__init__(epoch, pop_size, pc, pm, **kwargs)
        self.selection = self.validator.check_str("selection", selection, ["tournament", "random", "roulette"])
        self.crossover = self.validator.check_str("crossover", crossover, ["one_point", "multi_points", "uniform", "arithmetic"])
        self.mutation = self.validator.check_str("mutation", mutation, ["flip", "swap"])
        self.k_way = self.validator.check_float("k_way", k_way, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "pc", "pm", "selection", "crossover", "mutation", "k_way"])

    def mutation_process__(self, child):
        """
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm
        + Mutated on the whole vector is effected by parameter: pm
            + flip --> (default in this case) should set the pm small such as: [0.01 -> 0.2]
            + swap --> should set the pm small such as: [0.01 -> 0.2]

        Args:
            child (np.array): The position of the child

        Returns:
            np.array: The mutated vector of the child
        """
        if self.mutation == "swap":
            for idx in range(self.problem.n_dims):
                idx_swap = self.generator.choice(list(set(range(0, self.problem.n_dims)) - {idx}))
                child[idx], child[idx_swap] = child[idx_swap], child[idx]
                return child
        else:       # "flip"
            mutation_child = self.problem.generate_solution()
            flag_child = self.generator.uniform(0, 1, self.problem.n_dims) < self.pm
            return np.where(flag_child, mutation_child, child)


class EliteMultiGA(MultiGA):
    """
    The developed elite multipoints-mutation version of: Genetic Algorithm (GA)

    Links:
        1. https://www.baeldung.com/cs/elitism-in-evolutionary-algorithms

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.95
        + pm (float): [0.01, 0.2], mutation probability, default = 0.025
        + selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
        + k_way (float): Optional, set it when use "tournament" selection, default = 0.2
        + crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
        + mutation (str): Optional, can be ["flip", "swap"] for multipoints
        + elite_best (float/int): Optional, can be float (percentage of the best in elite group), or int (the number of best elite), default = 0.1
        + elite_worst (float/int): Opttional, can be float (percentage of the worst in elite group), or int (the number of worst elite), default = 0.3
        + strategy (int): Optional, can be 0 or 1. If = 0, the selection is select parents from (elite_worst + non_elite_group).
            Else, the selection will select dad from elite_worst and mom from non_elite_group.
        + pop_size = elite_group (elite_best + elite_worst) + non_elite_group

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GA
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
    >>> model = GA.EliteMultiGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, selection = "roulette", crossover = "uniform", mutation = "swap")
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Whitley, D., 1994. A genetic algorithm tutorial. Statistics and computing, 4(2), pp.65-85.
    """

    def __init__(self, epoch=10000, pop_size=100, pc=0.95, pm=0.8, selection="roulette",
                 crossover="uniform", mutation="swap", k_way=0.2,
                 elite_best=0.1, elite_worst=0.3, strategy=0, **kwargs):
        super().__init__(epoch, pop_size, pc, pm, selection, crossover, mutation, k_way, **kwargs)
        self.elite_best = self.validator.check_is_int_and_float("elite_best", elite_best, [1, int(self.pop_size / 2) - 1], (0, 0.5))
        self.n_elite_best = int(self.elite_best * self.pop_size) if self.elite_best < 1 else self.elite_best
        if self.n_elite_best < 1:
            self.n_elite_best = 1

        self.elite_worst = self.validator.check_is_int_and_float("elite_worst", elite_worst, [1, int(self.pop_size / 2) - 1], (0, 0.5))
        self.n_elite_worst = int(self.elite_worst * self.pop_size) if self.elite_worst < 1 else self.elite_worst
        if self.n_elite_worst < 1:
            self.n_elite_worst = 1

        self.strategy = self.validator.check_int("strategy", strategy, [0, 1])
        self.set_parameters(["epoch", "pop_size", "pc", "pm", "selection", "crossover", "mutation", "k_way",
                             "elite_best", "elite_worst", "strategy"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = self.pop[:self.n_elite_best]
        if self.strategy == 0:
            pop_old = self.pop[self.n_elite_best:]
            for idx in range(self.n_elite_best, self.pop_size):
                ### Selection
                child1, child2 = self.selection_process_00__(pop_old)
                ### Crossover
                if self.generator.uniform() < self.pc:
                    child1, child2 = self.crossover_process__(child1, child2)
                child = child1 if self.generator.random() <= 0.5 else child2
                ### Mutation
                child = self.mutation_process__(child)
                ### Survivor Selection
                pos_new = self.correct_solution(child)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1].target = self.get_target(pos_new)
            self.pop = self.update_target_for_population(pop_new)
        else:
            pop_dad = self.pop[self.n_elite_best:self.n_elite_best+self.n_elite_worst]
            pop_mom = self.pop[self.n_elite_best+self.n_elite_worst:]
            for idx in range(self.n_elite_best, self.pop_size):
                ### Selection
                child1, child2 = self.selection_process_01__(pop_dad, pop_mom)
                ### Crossover
                if self.generator.uniform() < self.pc:
                    child1, child2 = self.crossover_process__(child1, child2)
                child = child1 if self.generator.random() <= 0.5 else child2
                ### Mutation
                child = self.mutation_process__(child)
                ### Survivor Selection
                pos_new = self.correct_solution(child)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1].target = self.get_target(pos_new)
            self.pop = self.update_target_for_population(pop_new)
