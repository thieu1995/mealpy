# !/usr/bin/env python
# Created by "Thieu" at 14:51, 13/10/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.utils.logger import Logger
from mealpy.utils.visualize import export_convergence_chart, export_explore_exploit_chart, \
    export_diversity_chart, export_objectives_chart, export_trajectory_chart


class History:
    """
    A History class is responsible for saving each iteration's output.

    Notes
    ~~~~~
    + Access to variables in this class:
        + list_global_best: List of global best SOLUTION found so far in all previous generations
        + list_current_best: List of current best SOLUTION in each previous generations
        + list_epoch_time: List of runtime for each generation
        + list_global_best_fit: List of global best FITNESS found so far in all previous generations
        + list_current_best_fit: List of current best FITNESS in each previous generations
        + list_diversity: List of DIVERSITY of swarm in all generations
        + list_exploitation: List of EXPLOITATION percentages for all generations
        + list_exploration: List of EXPLORATION percentages for all generations
        + list_population: List of POPULATION in each generations
        + **Warning**, the last variable 'list_population' can cause the error related to 'memory' when saving model.
            Better to set parameter 'save_population' to False in the input problem dictionary to not using it.

    + There are 8 methods to draw available in this class:
        + save_global_best_fitness_chart()
        + save_local_best_fitness_chart()
        + save_global_objectives_chart()
        + save_local_objectives_chart()
        + save_exploration_exploitation_chart()
        + save_diversity_chart()
        + save_runtime_chart()
        + save_trajectory_chart()

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.PSO import BasePSO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>>     "save_population": False        # Then you can't draw the trajectory chart
    >>> }
    >>> model = BasePSO(problem_dict, epoch=1000, pop_size=50)
    >>>
    >>> model.history.save_global_objectives_chart(filename="hello/goc")
    >>> model.history.save_local_objectives_chart(filename="hello/loc")
    >>> model.history.save_global_best_fitness_chart(filename="hello/gbfc")
    >>> model.history.save_local_best_fitness_chart(filename="hello/lbfc")
    >>> model.history.save_runtime_chart(filename="hello/rtc")
    >>> model.history.save_exploration_exploitation_chart(filename="hello/eec")
    >>> model.history.save_diversity_chart(filename="hello/dc")
    >>> model.history.save_trajectory_chart(list_agent_idx=[3, 5], selected_dimensions=[3], filename="hello/tc")
    >>>
    >>> ## Get list of global best solution after all generations
    >>> print(model.history.list_global_best)
    """

    def __init__(self, **kwargs):
        self.list_global_best = []  # List of global best solution found so far in all previous generations
        self.list_current_best = []  # List of current best solution in each previous generations
        self.list_epoch_time = []  # List of runtime for each generation
        self.list_global_best_fit = []  # List of global best fitness found so far in all previous generations
        self.list_current_best_fit = []  # List of current best fitness in each previous generations
        self.list_population = []  # List of population in each generation
        self.list_diversity = []  # List of diversity of swarm in all generations
        self.list_exploitation = []  # List of exploitation percentages for all generations
        self.list_exploration = []  # List of exploration percentages for all generations
        self.epoch, self.log_to, self.log_file = None, None, None
        self.__set_keyword_arguments(kwargs)
        self.logger = Logger(self.log_to, log_file=self.log_file).create_logger(name=f"{__name__}.{__class__.__name__}",
            format_str='%(asctime)s, %(levelname)s, %(name)s [line: %(lineno)d]: %(message)s')

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def store_initial_best(self, best_agent):
        self.list_global_best = [deepcopy(best_agent)]
        self.list_current_best = [deepcopy(best_agent)]

    def get_global_repeated_times(self, id_fitness, id_target, epsilon):
        count = 0
        for i in range(0, len(self.list_global_best) - 1):
            temp = np.abs(self.list_global_best[i][id_fitness][id_target] - self.list_global_best[i + 1][id_fitness][id_target])
            if temp <= epsilon:
                count += 1
            else:
                count = 0
        return count

    def save_global_best_fitness_chart(self, title='Global Best Fitness', legend=None, linestyle='-', color='b', x_label="#Iteration",
                                       y_label="Function Value", filename="global-best-fitness-chart", exts=(".png", ".pdf"), verbose=True):
        # Draw global best fitness found so far in previous generations
        export_convergence_chart(data=self.list_global_best_fit, title=title, legend=legend, linestyle=linestyle,
                                 color=color, x_label=x_label, y_label=y_label, filename=filename, exts=exts, verbose=verbose)

    def save_local_best_fitness_chart(self, title='Local Best Fitness', legend=None, linestyle='-', color='b', x_label="#Iteration",
                                      y_label="Function Value", filename="local-best-fitness-chart", exts=(".png", ".pdf"), verbose=True):
        # Draw current best fitness in each previous generation
        export_convergence_chart(self.list_current_best_fit, title=title, legend=legend, linestyle=linestyle, color=color,
                                 x_label=x_label, y_label=y_label, filename=filename, exts=exts, verbose=verbose)

    def save_runtime_chart(self, title='Runtime chart', legend=None, linestyle='-', color='b', x_label="#Iteration",
                           y_label='Second', filename="runtime-chart", exts=(".png", ".pdf"), verbose=True):
        # Draw runtime for each generation
        export_convergence_chart(self.list_epoch_time, title=title, legend=legend, linestyle=linestyle, color=color,
                                 x_label=x_label, y_label=y_label, filename=filename, exts=exts, verbose=verbose)

    ## The paper: On the exploration and exploitation in popular swarm-based metaheuristic algorithms
    def save_exploration_exploitation_chart(self, title="Exploration vs Exploitation Percentages", list_colors=('blue', 'orange'),
                                            filename="exploration-exploitation-chart", verbose=True):
        # This exploration/exploitation chart should draws for single algorithm and single fitness function
        # Draw exploration and exploitation chart
        export_explore_exploit_chart(data=[self.list_exploration, self.list_exploitation], title=title,
                                     list_colors=list_colors, filename=filename, verbose=verbose)

    def save_diversity_chart(self, title='Diversity Measurement Chart', algorithm_name='Algorithm',
                             filename="diversity-chart", verbose=True):
        # This diversity chart should draws for multiple algorithms for a single fitness function at the same time
        # to compare the diversity spreading
        export_diversity_chart(data=[self.list_diversity], title=title, list_legends=[algorithm_name],
                               filename=filename, verbose=verbose)

    ## Because convergence chart is formulated from objective values and weights,
    ## thus we also want to draw objective charts to understand the convergence
    ## Need a little bit more pre-processing

    def save_global_objectives_chart(self, title='Global Objectives Chart', x_label="#Iteration", y_labels=None,
                                     filename="global-objectives-chart", verbose=True):
        # 2D array / matrix 2D
        global_obj_list = np.array([agent[1][-1] for agent in self.list_global_best])
        # Make each obj_list as a element in array for drawing
        global_obj_list = [global_obj_list[:, idx] for idx in range(0, len(global_obj_list[0]))]
        export_objectives_chart(global_obj_list, title=title, x_label=x_label, y_labels=y_labels, filename=filename, verbose=verbose)

    def save_local_objectives_chart(self, title='Local Objectives Chart', x_label="#Iteration", y_labels=None,
                                    filename="local-objectives-chart", verbose=True):
        current_obj_list = np.array([agent[1][-1] for agent in self.list_current_best])
        # Make each obj_list as a element in array for drawing
        current_obj_list = [current_obj_list[:, idx] for idx in range(0, len(current_obj_list[0]))]
        export_objectives_chart(current_obj_list, title=title, x_label=x_label, y_labels=y_labels,
                                filename=filename, verbose=verbose)

    def save_trajectory_chart(self, title="Trajectory of some agents",
                              list_agent_idx=(1, 2, 3), selected_dimensions=(1, 2),
                              filename="trajectory-chart", verbose=True):
        if len(self.list_population) < 2:
            self.logger.error("Can't draw the trajectory because 'save_population' is set to False or the number of epochs is too small.")
            exit(0)
        ## Drawing trajectory of some agents in the first and second dimensions
        # Need a little bit more pre-processing
        list_agent_idx = set(list_agent_idx)
        selected_dimensions = set(selected_dimensions)
        list_agent_idx = sorted(list_agent_idx)
        selected_dimensions = sorted(selected_dimensions)
        n_dim = len(selected_dimensions)

        if n_dim not in [1, 2]:
            self.logger.error("Trajectory chart for more than 2 dimensions is not supported.")
            exit(0)
        if len(list_agent_idx) < 1 or len(list_agent_idx) > 10:
            self.logger.error("Trajectory chart for more than 10 agents is not supported.")
            exit(0)
        if list_agent_idx[-1] > len(self.list_population[0]) or list_agent_idx[0] < 1:
            self.logger.error(f"Can't draw trajectory chart, the index of selected agents should be in range of [1, {len(self.list_population[0])}]")
            exit(0)
        if selected_dimensions[-1] > len(self.list_population[0][0][0]) or selected_dimensions[0] < 1:
            self.logger.error(f"Can't draw trajectory chart, the index of selected dimensions should be in range of [1, {len(self.list_population[0][0][0])}]")
            exit(0)

        pos_list = []
        list_legends = []

        # pop[0]: Get the first solution
        # pop[0][0]: Get the position of the first solution
        # pop[0][0][0]: Get the first dimension of the position of the first solution
        if n_dim == 1:
            y_label = f"x{selected_dimensions[0]}"
            for idx, id_agent in enumerate(list_agent_idx):
                x = [pop[id_agent - 1][0][selected_dimensions[0] - 1] for pop in self.list_population]
                pos_list.append(x)
                list_legends.append(f"Agent {id_agent}")
            export_trajectory_chart(pos_list, n_dimensions=n_dim, title=title, list_legends=list_legends,
                                    y_label=y_label, filename=filename, verbose=verbose)
        elif n_dim == 2:
            x_label = f"x{selected_dimensions[0]}"
            y_label = f"x{selected_dimensions[1]}"
            for idx1, id_agent in enumerate(list_agent_idx):
                pos_temp = []
                for idx2, id_dim in enumerate(selected_dimensions):
                    x = [pop[id_agent - 1][0][id_dim - 1] for pop in self.list_population]
                    pos_temp.append(x)
                pos_list.append(pos_temp)
                list_legends.append(f"Agent {id_agent}")
            export_trajectory_chart(pos_list, n_dimensions=n_dim, title=title, list_legends=list_legends, x_label=x_label,
                                    y_label=y_label, filename=filename, verbose=verbose)
